"""Leakage-safe architecture search + rigorous validation for DCT-FNO.

Screens candidate (width, modes) configurations at a reduced epoch/seed
budget (100 epochs x 3 screening seeds, averaged validation score) to avoid
overfitting the 9-sample validation set to a single seed's luck -- the same
concern that made the earlier hand-picked invention hyperparameters
untrustworthy. Only the single winning configuration is then handed to the
full 10-seed x 300-epoch protocol already used for every other model in
`docs/EXTENDED_MULTISEED_STUDY_2026-07-17.md`, so the final comparison
against vanilla FNO is exactly as rigorous as everything else, not inflated
by having tried many configurations.
"""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = ROOT / "scripts" / "train_smoke.py"
DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v1.h5"

# Screening phase: cheap, multi-seed averaged validation score, never touches test.
SCREEN_SEEDS = [101, 102, 103]
SCREEN_EPOCHS = 100
WIDTHS = [16, 24, 32]
MODES = [8, 12, 16]
LAYERS = 4

# Final phase: identical protocol to every other model in the extended study.
FINAL_SEEDS = list(range(1, 11))
FINAL_EPOCHS = 300


def run_name_for(width: int, modes: int, seed: int, epochs: int) -> str:
    suffix = f"_seed{seed}_ep{epochs}"
    return f"dct_fno_w{width}_mx{modes}_my{modes}_l{LAYERS}{suffix}"


def train(width: int, modes: int, seed: int, epochs: int) -> Path:
    out_dir = ROOT / "outputs" / "smoke" / DATA.stem / run_name_for(width, modes, seed, epochs)
    results_path = out_dir / "results.json"
    if results_path.exists():
        print(f"[skip] {out_dir.name}")
        return results_path
    cmd = [
        PY, str(TRAIN),
        "--data", str(DATA),
        "--model", "dct_fno",
        "--width", str(width),
        "--modes-x", str(modes),
        "--modes-y", str(modes),
        "--layers", str(LAYERS),
        "--epochs", str(epochs),
        "--allow-extended-training",
        "--seed", str(seed),
    ]
    print(f"[run] {out_dir.name}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return results_path


def val_score(results_path: Path) -> float:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    best_l2 = data["best_val_relative_l2"]
    for h in data["history"]:
        if abs(h["val_relative_l2"] - best_l2) < 1e-9:
            return h["val_relative_l2"] + 2.0 * h["val_hotspot_mae"]
    h = data["history"][-1]
    return h["val_relative_l2"] + 2.0 * h["val_hotspot_mae"]


def screen() -> tuple[int, int]:
    candidates = list(product(WIDTHS, MODES))
    scored = []
    for width, modes in candidates:
        scores = [val_score(train(width, modes, seed, SCREEN_EPOCHS)) for seed in SCREEN_SEEDS]
        mean_score = mean(scores)
        print(f"width={width} modes={modes}: mean_val_score={mean_score:.4f} (scores={[round(s, 4) for s in scores]})")
        scored.append((mean_score, width, modes))
    scored.sort(key=lambda t: t[0])
    best_score, best_width, best_modes = scored[0]
    print(f"[selected] width={best_width} modes={best_modes} mean_val_score={best_score:.4f}")
    screen_summary = {
        "screen_seeds": SCREEN_SEEDS,
        "screen_epochs": SCREEN_EPOCHS,
        "candidates": [{"width": w, "modes": m, "mean_val_score": s} for s, w, m in scored],
        "selected": {"width": best_width, "modes": best_modes},
    }
    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "dct_fno_architecture_search.json"
    out_path.write_text(json.dumps(screen_summary, indent=2), encoding="utf-8")
    return best_width, best_modes


def validate(width: int, modes: int) -> dict:
    runs = []
    for seed in FINAL_SEEDS:
        results_path = train(width, modes, seed, FINAL_EPOCHS)
        data = json.loads(results_path.read_text(encoding="utf-8"))
        test = data["test"]
        runs.append(
            {
                "seed": seed,
                "relative_l2": test["relative_l2"],
                "hotspot_mae": test["hotspot_mae"],
                "equilibrium_residual": test["equilibrium_residual"],
            }
        )

    fno_by_seed = {}
    for seed in FINAL_SEEDS:
        fno_path = ROOT / "outputs" / "smoke" / DATA.stem / f"fno_seed{seed}_ep{FINAL_EPOCHS}" / "results.json"
        fno_by_seed[seed] = json.loads(fno_path.read_text(encoding="utf-8"))["test"]

    diffs_l2 = [r["relative_l2"] - fno_by_seed[r["seed"]]["relative_l2"] for r in runs]
    diffs_hot = [r["hotspot_mae"] - fno_by_seed[r["seed"]]["hotspot_mae"] for r in runs]
    diffs_eq = [r["equilibrium_residual"] - fno_by_seed[r["seed"]]["equilibrium_residual"] for r in runs]

    def paired_stats(diffs: list[float]) -> dict:
        n = len(diffs)
        m = mean(diffs)
        s = stdev(diffs) if n > 1 else 0.0
        se = s / (n ** 0.5) if n > 1 else 0.0
        t = m / se if se > 0 else float("inf")
        return {"mean_diff": m, "sd_diff": s, "t": t, "n": n}

    summary = {
        "model": "dct_fno",
        "selected_architecture": {"width": width, "modes_x": modes, "modes_y": modes, "layers": LAYERS},
        "n_seeds": len(runs),
        "relative_l2_mean": mean(r["relative_l2"] for r in runs),
        "relative_l2_std": stdev(r["relative_l2"] for r in runs),
        "hotspot_mae_mean": mean(r["hotspot_mae"] for r in runs),
        "hotspot_mae_std": stdev(r["hotspot_mae"] for r in runs),
        "equilibrium_residual_mean": mean(r["equilibrium_residual"] for r in runs),
        "equilibrium_residual_std": stdev(r["equilibrium_residual"] for r in runs),
        "runs": runs,
        "paired_vs_vanilla_fno": {
            "relative_l2": paired_stats(diffs_l2),
            "hotspot_mae": paired_stats(diffs_hot),
            "equilibrium_residual": paired_stats(diffs_eq),
        },
    }
    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "dct_fno_validated_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["paired_vs_vanilla_fno"], indent=2))
    return summary


def main() -> None:
    width, modes = screen()
    validate(width, modes)


if __name__ == "__main__":
    main()

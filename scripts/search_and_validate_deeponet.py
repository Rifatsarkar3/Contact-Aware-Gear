"""Leakage-safe architecture search + rigorous validation for DeepONet.

Same protocol as `search_and_validate_dwm_fno.py` (and its CCM-FNO/HGM-FNO/
DCT-FNO siblings): 3 screening seeds disjoint from the 10 final-validation
seeds, screened at 100 epochs, final validation at 300 epochs against the
cached vanilla-FNO default results. DeepONet (`src/gearstress/models.py`,
new class) is a from-scratch baseline built to close the "DeepONet" portion
of MSSP's submission gate (`docs/journal/MSSP_AUTHOR_REQUIREMENTS.md`, item
3) -- unlike U-Net, it has no prior implementation or tuning in this
project, so a real leakage-safe search over its two architecture
hyperparameters (branch/trunk width, shared basis dimension) is required for
the comparison to be "fairly tuned," not just "run."
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

SCREEN_SEEDS = [101, 102, 103]
SCREEN_EPOCHS = 100
WIDTHS = [16, 24, 32]
BASIS_DIMS = [32, 64, 96]

FINAL_SEEDS = list(range(1, 11))
FINAL_EPOCHS = 300


def run_name_for(width: int, basis_dim: int, seed: int, epochs: int) -> str:
    suffix = f"_seed{seed}_ep{epochs}"
    return f"deeponet_w{width}_mx12_my12_l4_bd{basis_dim}{suffix}"


def train(width: int, basis_dim: int, seed: int, epochs: int) -> Path:
    out_dir = ROOT / "outputs" / "smoke" / DATA.stem / run_name_for(width, basis_dim, seed, epochs)
    results_path = out_dir / "results.json"
    if results_path.exists():
        print(f"[skip] {out_dir.name}")
        return results_path
    cmd = [
        PY, str(TRAIN),
        "--data", str(DATA),
        "--model", "deeponet",
        "--width", str(width),
        "--basis-dim", str(basis_dim),
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
    candidates = list(product(WIDTHS, BASIS_DIMS))
    scored = []
    for width, basis_dim in candidates:
        scores = [val_score(train(width, basis_dim, seed, SCREEN_EPOCHS)) for seed in SCREEN_SEEDS]
        mean_score = mean(scores)
        print(f"width={width} basis_dim={basis_dim}: mean_val_score={mean_score:.4f} (scores={[round(s, 4) for s in scores]})")
        scored.append((mean_score, width, basis_dim))
    scored.sort(key=lambda t: t[0])
    best_score, best_width, best_basis_dim = scored[0]
    print(f"[selected] width={best_width} basis_dim={best_basis_dim} mean_val_score={best_score:.4f}")
    summary = {
        "screen_seeds": SCREEN_SEEDS,
        "screen_epochs": SCREEN_EPOCHS,
        "candidates": [{"width": w, "basis_dim": bd, "mean_val_score": s} for s, w, bd in scored],
        "selected": {"width": best_width, "basis_dim": best_basis_dim},
    }
    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "deeponet_architecture_search.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return best_width, best_basis_dim


def validate(width: int, basis_dim: int) -> dict:
    runs = []
    for seed in FINAL_SEEDS:
        results_path = train(width, basis_dim, seed, FINAL_EPOCHS)
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
        "model": "deeponet",
        "selected_architecture": {"width": width, "basis_dim": basis_dim},
        "n_seeds": len(runs),
        "relative_l2_mean": mean(r["relative_l2"] for r in runs),
        "relative_l2_std": stdev(r["relative_l2"] for r in runs),
        "hotspot_mae_mean": mean(r["hotspot_mae"] for r in runs),
        "hotspot_mae_std": stdev(r["hotspot_mae"] for r in runs),
        "equilibrium_residual_mean": mean(r["equilibrium_residual"] for r in runs),
        "equilibrium_residual_std": stdev(r["equilibrium_residual"] for r in runs),
        "runs": runs,
        "paired_vs_vanilla_fno_default": {
            "relative_l2": paired_stats(diffs_l2),
            "hotspot_mae": paired_stats(diffs_hot),
            "equilibrium_residual": paired_stats(diffs_eq),
        },
    }
    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "deeponet_validated_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["paired_vs_vanilla_fno_default"], indent=2))
    return summary


def main() -> None:
    width, basis_dim = screen()
    validate(width, basis_dim)


if __name__ == "__main__":
    main()

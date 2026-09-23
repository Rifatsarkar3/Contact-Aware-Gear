"""Leakage-safe architecture search + rigorous validation for HGM-FNO.

Same protocol as `scripts/search_and_validate_ccm_fno.py` (global branch
held at vanilla FNO's default width/modes/layers throughout the search;
local-branch hyperparameters only are searched; screening seeds distinct
from final validation seeds). HGM-FNO fixes CCM-FNO's diagnosed flaw: the
local window is now centered on a soft-argmax of the global branch's own
predicted stress magnitude, not the (measurably wrong) input contact-map
centroid. `softmax_temperature` is left at its default (not searched, to
keep scope bounded).
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
LOCAL_WINDOWS = [0.25, 0.375, 0.5]
LOCAL_WIDTHS = [8, 16, 24]
LOCAL_SIZE = 16

FINAL_SEEDS = list(range(1, 11))
FINAL_EPOCHS = 300


def run_name_for(window: float, width: int, seed: int, epochs: int) -> str:
    suffix = f"_seed{seed}_ep{epochs}"
    return f"hgm_fno_lw{width}_lwin{window:g}_lsz{LOCAL_SIZE}{suffix}"


def train(window: float, width: int, seed: int, epochs: int) -> Path:
    out_dir = ROOT / "outputs" / "smoke" / DATA.stem / run_name_for(window, width, seed, epochs)
    results_path = out_dir / "results.json"
    if results_path.exists():
        print(f"[skip] {out_dir.name}")
        return results_path
    cmd = [
        PY, str(TRAIN),
        "--data", str(DATA),
        "--model", "hgm_fno",
        "--local-window", str(window),
        "--local-width", str(width),
        "--local-size", str(LOCAL_SIZE),
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


def screen() -> tuple[float, int]:
    candidates = list(product(LOCAL_WINDOWS, LOCAL_WIDTHS))
    scored = []
    for window, width in candidates:
        scores = [val_score(train(window, width, seed, SCREEN_EPOCHS)) for seed in SCREEN_SEEDS]
        mean_score = mean(scores)
        print(f"window={window} local_width={width}: mean_val_score={mean_score:.4f} (scores={[round(s, 4) for s in scores]})")
        scored.append((mean_score, window, width))
    scored.sort(key=lambda t: t[0])
    best_score, best_window, best_width = scored[0]
    print(f"[selected] window={best_window} local_width={best_width} mean_val_score={best_score:.4f}")
    summary = {
        "screen_seeds": SCREEN_SEEDS,
        "screen_epochs": SCREEN_EPOCHS,
        "candidates": [{"window": w, "local_width": lw, "mean_val_score": s} for s, w, lw in scored],
        "selected": {"window": best_window, "local_width": best_width, "local_size": LOCAL_SIZE},
    }
    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "hgm_fno_architecture_search.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return best_window, best_width


def validate(window: float, width: int) -> dict:
    runs = []
    for seed in FINAL_SEEDS:
        results_path = train(window, width, seed, FINAL_EPOCHS)
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
        "model": "hgm_fno",
        "selected_architecture": {"local_window": window, "local_width": width, "local_size": LOCAL_SIZE},
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
    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "hgm_fno_validated_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["paired_vs_vanilla_fno_default"], indent=2))
    return summary


def main() -> None:
    window, width = screen()
    validate(window, width)


if __name__ == "__main__":
    main()

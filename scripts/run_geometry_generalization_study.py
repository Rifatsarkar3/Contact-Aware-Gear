"""Does the vanilla FNO baseline generalize across tooth geometry, or only the
single fixed geometry (module=2.0mm, teeth=24, pressure_angle_deg=20.0,
root_clearance_module=0.25) used by every other dataset in this project?

``gear_pair_geometry_v1.h5`` (see ``generate_gear_pair_geometry_dataset.py``)
randomizes pressure_angle_deg in [17.5, 22.5] deg and root_clearance_module
in [0.20, 0.32] per case. This script trains the same vanilla FNO under two
split conditions on that one dataset, 10 seeds each (same protocol convention
as ``run_v2_multiseed_study.py``):

- ``random_split``: an ordinary random train/val/test split (grouped_trajectory
  split with each case its own trivial group) -- train and test geometries
  are drawn from the same distribution.
- ``geometry_holdout``: the top pressure-angle tercile [20.8333, 22.5] deg is
  withheld entirely for test -- the model never sees that geometry band while
  training. This is the genuine out-of-distribution generalization test.

A paired t-test (n=10, matched by seed) between the two conditions' test
relative_l2 is the headline result: if geometry_holdout is significantly
worse than random_split, that is honest evidence of a real geometry-
generalization gap; if not, that is honest evidence the FNO's spectral basis
generalizes across this geometry range without needing to see it directly.
Either outcome is reported as found, not adjusted toward either conclusion.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = ROOT / "scripts" / "train_smoke.py"
DATA = ROOT / "data" / "processed" / "gear_pair_geometry_v1.h5"

SEEDS = list(range(1, 11))
EPOCHS = 300
HOLDOUT_LOW = 20.8333
HOLDOUT_HIGH = 22.5

CONDITIONS = ["random_split", "geometry_holdout"]


def run_name_for(condition: str, seed: int) -> str:
    suffix = f"_seed{seed}_ep{EPOCHS}"
    if condition == "geometry_holdout":
        return f"fno_geomholdout{HOLDOUT_LOW:g}-{HOLDOUT_HIGH:g}{suffix}"
    return f"fno{suffix}"


def train(condition: str, seed: int) -> Path:
    run_name = run_name_for(condition, seed)
    out_dir = ROOT / "outputs" / "smoke" / DATA.stem / run_name
    results_path = out_dir / "results.json"
    if results_path.exists():
        print(f"[skip] {run_name}")
        return results_path
    cmd = [
        PY, str(TRAIN),
        "--data", str(DATA),
        "--model", "fno",
        "--epochs", str(EPOCHS),
        "--allow-extended-training",
        "--seed", str(seed),
    ]
    if condition == "geometry_holdout":
        cmd += [
            "--split-scheme", "geometry_holdout",
            "--holdout-low", str(HOLDOUT_LOW),
            "--holdout-high", str(HOLDOUT_HIGH),
        ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return results_path


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    m = mean(diffs)
    s = stdev(diffs) if n > 1 else 0.0
    se = s / (n ** 0.5) if n > 1 else 0.0
    t = m / se if se > 0 else float("inf")
    return {"mean_diff": m, "sd_diff": s, "t": t, "n": n}


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found -- run generate_gear_pair_geometry_dataset.py first")

    for condition in CONDITIONS:
        for seed in SEEDS:
            train(condition, seed)

    runs_by_condition: dict[str, list[dict]] = {}
    for condition in CONDITIONS:
        runs = []
        for seed in SEEDS:
            results_path = ROOT / "outputs" / "smoke" / DATA.stem / run_name_for(condition, seed) / "results.json"
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
        runs_by_condition[condition] = runs

    baseline_by_seed = {r["seed"]: r for r in runs_by_condition["random_split"]}

    summary = {}
    for condition in CONDITIONS:
        runs = runs_by_condition[condition]
        entry = {
            "n_seeds": len(runs),
            "relative_l2_mean": mean(r["relative_l2"] for r in runs),
            "relative_l2_std": stdev(r["relative_l2"] for r in runs),
            "hotspot_mae_mean": mean(r["hotspot_mae"] for r in runs),
            "hotspot_mae_std": stdev(r["hotspot_mae"] for r in runs),
            "equilibrium_residual_mean": mean(r["equilibrium_residual"] for r in runs),
            "equilibrium_residual_std": stdev(r["equilibrium_residual"] for r in runs),
            "runs": runs,
        }
        if condition != "random_split":
            diffs_l2 = [r["relative_l2"] - baseline_by_seed[r["seed"]]["relative_l2"] for r in runs]
            diffs_hot = [r["hotspot_mae"] - baseline_by_seed[r["seed"]]["hotspot_mae"] for r in runs]
            diffs_eq = [r["equilibrium_residual"] - baseline_by_seed[r["seed"]]["equilibrium_residual"] for r in runs]
            entry["paired_vs_random_split"] = {
                "relative_l2": paired_stats(diffs_l2),
                "hotspot_mae": paired_stats(diffs_hot),
                "equilibrium_residual": paired_stats(diffs_eq),
            }
        summary[condition] = entry

    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "geometry_generalization_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary written] {out_path}")
    for condition, stats in summary.items():
        print(
            f"{condition}: relative_l2 {stats['relative_l2_mean']:.4f} +/- {stats['relative_l2_std']:.4f}, "
            f"hotspot_mae {stats['hotspot_mae_mean']:.4f} +/- {stats['hotspot_mae_std']:.4f}, "
            f"equilibrium {stats['equilibrium_residual_mean']:.2f} +/- {stats['equilibrium_residual_std']:.2f}"
        )
        if "paired_vs_random_split" in stats:
            print(json.dumps(stats["paired_vs_random_split"], indent=2))


if __name__ == "__main__":
    main()

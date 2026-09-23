"""Add DeepONet's capacity-matched control on v1, alongside its already-searched result.

`scripts/search_and_validate_deeponet.py` already produced DeepONet's real,
leakage-safely-searched final-validation result on v1 (width=32, basis_dim=32
-- 70,371 parameters, chosen purely by validation score). This script adds
one more real number: a capacity-matched control (width=98, basis_dim=128
-- 663,093 parameters, within 0.7% of vanilla FNO's 667,515-param default,
chosen by parameter counting, not validation-loss search), following this
project's own established practice (`docs/DCT_FNO_STUDY_2026-07-17.md`,
also applied to the U-Net baseline in `docs/UNET_BASELINE_STUDY_2026-08-30.md`)
of checking whether a baseline's loss could be dismissed as an unfair
capacity handicap. Same run-naming scheme as `search_and_validate_deeponet.py`,
so the already-computed searched-config runs are reused (skip-if-exists),
not retrained.
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
DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v1.h5"

SEEDS = list(range(1, 11))
EPOCHS = 300

MODELS = [
    {"key": "fno", "model": "fno", "width": None, "basis_dim": None},
    {"key": "deeponet_searched", "model": "deeponet", "width": 32, "basis_dim": 32},
    {"key": "deeponet_capacity_matched", "model": "deeponet", "width": 98, "basis_dim": 128},
]


def run_suffix(seed: int) -> str:
    return f"_seed{seed}_ep{EPOCHS}"


def run_name_for(spec: dict, seed: int) -> str:
    if spec["width"] is None:
        return f"{spec['model']}{run_suffix(seed)}"
    return f"deeponet_w{spec['width']}_mx12_my12_l4_bd{spec['basis_dim']}{run_suffix(seed)}"


def train(spec: dict, seed: int) -> Path:
    run_name = run_name_for(spec, seed)
    out_dir = ROOT / "outputs" / "smoke" / DATA.stem / run_name
    results_path = out_dir / "results.json"
    if results_path.exists():
        print(f"[skip] {run_name}")
        return results_path
    cmd = [
        PY, str(TRAIN),
        "--data", str(DATA),
        "--model", spec["model"],
        "--epochs", str(EPOCHS),
        "--allow-extended-training",
        "--seed", str(seed),
    ]
    if spec["width"] is not None:
        cmd += ["--width", str(spec["width"]), "--basis-dim", str(spec["basis_dim"])]
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
    for seed in SEEDS:
        for spec in MODELS:
            train(spec, seed)

    runs_by_key: dict[str, list[dict]] = {}
    for spec in MODELS:
        runs = []
        for seed in SEEDS:
            results_path = ROOT / "outputs" / "smoke" / DATA.stem / run_name_for(spec, seed) / "results.json"
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
        runs_by_key[spec["key"]] = runs

    baseline_by_seed = {r["seed"]: r for r in runs_by_key["fno"]}

    summary = {}
    for spec in MODELS:
        key = spec["key"]
        runs = runs_by_key[key]
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
        if key != "fno":
            diffs_l2 = [r["relative_l2"] - baseline_by_seed[r["seed"]]["relative_l2"] for r in runs]
            diffs_hot = [r["hotspot_mae"] - baseline_by_seed[r["seed"]]["hotspot_mae"] for r in runs]
            diffs_eq = [r["equilibrium_residual"] - baseline_by_seed[r["seed"]]["equilibrium_residual"] for r in runs]
            entry["paired_vs_fno_default"] = {
                "relative_l2": paired_stats(diffs_l2),
                "hotspot_mae": paired_stats(diffs_hot),
                "equilibrium_residual": paired_stats(diffs_eq),
            }
        summary[key] = entry

    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "deeponet_baseline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary written] {out_path}")
    for key, stats in summary.items():
        print(
            f"{key}: relative_l2 {stats['relative_l2_mean']:.4f} +/- {stats['relative_l2_std']:.4f}, "
            f"hotspot_mae {stats['hotspot_mae_mean']:.4f} +/- {stats['hotspot_mae_std']:.4f}, "
            f"equilibrium {stats['equilibrium_residual_mean']:.2f} +/- {stats['equilibrium_residual_std']:.2f}"
        )
        if "paired_vs_fno_default" in stats:
            print(json.dumps(stats["paired_vs_fno_default"], indent=2))


if __name__ == "__main__":
    main()

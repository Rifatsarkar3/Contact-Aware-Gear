"""Extend an active-learning arm's seed count to reach adequate statistical power for DWM-FNO/AEE-FNO.

Generalizes scripts/extend_pool_based_seeds.py to any arm -- written
after ensemble_surrogate's own DWM-FNO comparison turned out underpowered
too (40.5% retrospective power at n=10, same issue pool_based had),
while its AEE-FNO comparison was already adequately powered (86.1%).
Reuses run_al_misranking_comparison.py's model construction and training
loop directly -- seeds already present for a given arm are skipped.

Usage:
    python scripts/extend_al_arm_seeds.py --arm ensemble_surrogate --up-to 34
    python scripts/extend_al_arm_seeds.py --arm pool_based --up-to 34
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_al_misranking_comparison as base  # noqa: E402

MODEL_KEYS = ["fno", "aee_fno", "dwm_fno"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("pool_based", "ensemble_surrogate", "random_baseline"))
    parser.add_argument("--up-to", type=int, required=True, help="extend seeds 1..N (existing seeds are skipped)")
    args = parser.parse_args()

    dataset_path = ROOT / "data" / "processed" / f"gear_pair_al_{args.arm}_extended.h5"
    extended_seeds = list(range(1, args.up_to + 1))

    base.cap_gpu_memory()
    cfg = yaml.safe_load(base.CONFIG.read_text(encoding="utf-8"))
    v1_split_ids = base.v1_fixed_split_trajectory_ids()

    with h5py.File(dataset_path, "r") as handle:
        inputs_all = np.asarray(handle["inputs"], dtype=np.float32)
        targets_raw = np.asarray(handle["targets"], dtype=np.float32)
        trajectory_ids = np.asarray(handle["trajectory_ids"])
        grid_dx = float(handle.attrs["grid_dx_mm"])
        grid_dy = float(handle.attrs["grid_dy_mm"])

    train_id_set = v1_split_ids["train"] | (
        set(int(t) for t in trajectory_ids.tolist()) - v1_split_ids["train"] - v1_split_ids["val"] - v1_split_ids["test"]
    )
    val_idx = np.flatnonzero(np.isin(trajectory_ids, list(v1_split_ids["val"])))
    test_idx = np.flatnonzero(np.isin(trajectory_ids, list(v1_split_ids["test"])))
    train_idx = np.flatnonzero(np.isin(trajectory_ids, list(train_id_set)))

    runs_by_model: dict[str, dict[int, dict]] = {}
    for model_key in MODEL_KEYS:
        by_seed = {}
        for seed in extended_seeds:
            path = base.run_one(inputs_all, targets_raw, train_idx, val_idx, test_idx, model_key, seed, cfg, grid_dx, grid_dy, args.arm)
            data = json.loads(path.read_text(encoding="utf-8"))
            by_seed[seed] = data["test"]
        runs_by_model[model_key] = by_seed

    baseline = runs_by_model["fno"]
    summary: dict = {"n_train": int(train_idx.size), "seeds": extended_seeds}
    for model_key in MODEL_KEYS:
        by_seed = runs_by_model[model_key]
        entry = {
            "relative_l2_mean": mean(v["relative_l2"] for v in by_seed.values()),
            "relative_l2_std": stdev(v["relative_l2"] for v in by_seed.values()),
        }
        if model_key != "fno":
            diffs = [by_seed[s]["relative_l2"] - baseline[s]["relative_l2"] for s in extended_seeds]
            n = len(diffs)
            m = mean(diffs)
            sd = stdev(diffs)
            se = sd / (n**0.5)
            t = m / se if se > 0 else float("inf")
            entry["paired_vs_fno_relative_l2"] = {"mean_diff": m, "sd_diff": sd, "t": t, "n": n}
        summary[model_key] = entry

    out_path = base.OUT_ROOT / f"{args.arm}_extended_seeds_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[written] {out_path}")
    for model_key in ("aee_fno", "dwm_fno"):
        p = summary[model_key]["paired_vs_fno_relative_l2"]
        print(f"{model_key} (n={p['n']}): diff={p['mean_diff']:+.4f} t={p['t']:+.2f}")


if __name__ == "__main__":
    main()

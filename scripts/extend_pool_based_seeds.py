"""Extend pool_based's seed count to reach adequate statistical power for DWM-FNO/AEE-FNO.

Per docs/AL_MISRANKING_COMBINED_STUDY_2026-08-04.md's power analysis:
pool_based's paired-seed noise is roughly double random_baseline's,
leaving DWM-FNO's n=10 comparison underpowered (33.3% retrospective
power; ~34 seeds needed for 80%) and AEE-FNO's right at the edge (80.6%;
~10 seeds needed, i.e. already met but with no margin). This trains
seeds 11-34 (24 more) for fno/aee_fno/dwm_fno on pool_based only --
random_baseline is not extended since it is already well-powered at
n=10 for both mechanisms. Reuses run_al_misranking_comparison.py's
model construction and training loop directly (no duplication) -- seeds
1-10 are already cached there and will be skipped.

Usage (once launched):
    python scripts/extend_pool_based_seeds.py
"""

from __future__ import annotations

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

DATASET_PATH = ROOT / "data" / "processed" / "gear_pair_al_pool_based_extended.h5"
EXTENDED_SEEDS = list(range(1, 35))  # 1-34; seeds 1-10 already exist and will be skipped
MODEL_KEYS = ["fno", "aee_fno", "dwm_fno"]


def main() -> None:
    base.cap_gpu_memory()
    cfg = yaml.safe_load(base.CONFIG.read_text(encoding="utf-8"))
    v1_split_ids = base.v1_fixed_split_trajectory_ids()

    with h5py.File(DATASET_PATH, "r") as handle:
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
        for seed in EXTENDED_SEEDS:
            path = base.run_one(inputs_all, targets_raw, train_idx, val_idx, test_idx, model_key, seed, cfg, grid_dx, grid_dy, "pool_based")
            data = json.loads(path.read_text(encoding="utf-8"))
            by_seed[seed] = data["test"]
        runs_by_model[model_key] = by_seed

    baseline = runs_by_model["fno"]
    summary: dict = {"n_train": int(train_idx.size), "seeds": EXTENDED_SEEDS}
    for model_key in MODEL_KEYS:
        by_seed = runs_by_model[model_key]
        entry = {
            "relative_l2_mean": mean(v["relative_l2"] for v in by_seed.values()),
            "relative_l2_std": stdev(v["relative_l2"] for v in by_seed.values()),
        }
        if model_key != "fno":
            diffs = [by_seed[s]["relative_l2"] - baseline[s]["relative_l2"] for s in EXTENDED_SEEDS]
            n = len(diffs)
            m = mean(diffs)
            sd = stdev(diffs)
            se = sd / (n**0.5)
            t = m / se if se > 0 else float("inf")
            entry["paired_vs_fno_relative_l2"] = {"mean_diff": m, "sd_diff": sd, "t": t, "n": n}
        summary[model_key] = entry

    out_path = base.OUT_ROOT / "pool_based_extended_seeds_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[written] {out_path}")
    for model_key in ("aee_fno", "dwm_fno"):
        p = summary[model_key]["paired_vs_fno_relative_l2"]
        print(f"{model_key} (n={p['n']}): diff={p['mean_diff']:+.4f} t={p['t']:+.2f}")


if __name__ == "__main__":
    main()

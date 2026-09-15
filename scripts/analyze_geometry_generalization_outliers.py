"""Recompute robust (median-based) test metrics for the geometry-generalization
study and rewrite its summary with both mean- and median-based statistics.

The raw mean-of-per-sample-ratios relative_l2 (as computed identically to
every other study in this project, see ``gearstress.losses.relative_l2``) is
outlier-inflated here in a way it is not elsewhere: this dataset's stress
magnitude varies ~60x across sampled geometries (pressure_angle_deg strongly
affects peak contact stress), versus a much narrower range in every
fixed-geometry dataset. A handful of near-zero-target-norm test cases (weakly
loaded, low-stress geometry/loading combinations, not solver failures --
their FEM solves converged normally) produce enormous per-sample ratios that
dominate the mean. This script quantifies that precisely: it reports the
outlier cases explicitly (not silently dropped) and adds the median -- robust
to this specific failure mode -- as the primary comparison statistic,
alongside the original mean for full transparency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.models import FNO2d  # noqa: E402
from gearstress.splits import geometry_holdout_split, grouped_trajectory_split  # noqa: E402

DATA = ROOT / "data" / "processed" / "gear_pair_geometry_v1.h5"
OUT_DIR = ROOT / "outputs" / "smoke" / DATA.stem
HOLDOUT_LOW, HOLDOUT_HIGH = 20.8333, 22.5
SEEDS = list(range(1, 11))
CONDITIONS = ["random_split", "geometry_holdout"]
OUTLIER_THRESHOLD = 3.0


def run_name_for(condition: str, seed: int) -> str:
    if condition == "geometry_holdout":
        return f"fno_geomholdout{HOLDOUT_LOW:g}-{HOLDOUT_HIGH:g}_seed{seed}_ep300"
    return f"fno_seed{seed}_ep300"


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    m = mean(diffs)
    s = stdev(diffs) if n > 1 else 0.0
    se = s / (n ** 0.5) if n > 1 else 0.0
    t = m / se if se > 0 else float("inf")
    return {"mean_diff": m, "sd_diff": s, "t": t, "n": n}


def main() -> None:
    with h5py.File(DATA, "r") as f:
        inputs = np.asarray(f["inputs"], dtype=np.float32)
        targets = np.asarray(f["targets"], dtype=np.float32)
        traj = np.asarray(f["trajectory_ids"])
        pa = np.asarray(f["pressure_angle_deg"], dtype=np.float32)

    per_condition: dict[str, list[dict]] = {}
    outlier_log: list[dict] = []
    for condition in CONDITIONS:
        runs = []
        for seed in SEEDS:
            if condition == "random_split":
                split = grouped_trajectory_split(traj, train_fraction=0.625, val_fraction=0.1875, seed=42)
            else:
                split = geometry_holdout_split(
                    pa, holdout_low=HOLDOUT_LOW, holdout_high=HOLDOUT_HIGH, val_fraction=0.1875, seed=42
                )
            target_scale = max(float(np.max(np.abs(targets[split.train]))), 1e-8)
            targets_n = targets / target_scale

            model = FNO2d(width=24, modes_x=12, modes_y=12, layers=4)
            model_path = OUT_DIR / run_name_for(condition, seed) / "model.pt"
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            test_idx = split.test
            with torch.no_grad():
                pred = model(torch.from_numpy(inputs[test_idx]))
                tgt = torch.from_numpy(targets_n[test_idx])
                num = torch.linalg.vector_norm((pred - tgt).flatten(1), dim=1)
                den = torch.linalg.vector_norm(tgt.flatten(1), dim=1).clamp_min(1e-8)
                per_sample = (num / den).numpy()

            outlier_mask = per_sample > OUTLIER_THRESHOLD
            for local_idx in np.flatnonzero(outlier_mask):
                outlier_log.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "rel_l2": float(per_sample[local_idx]),
                        "pressure_angle_deg": float(pa[test_idx[local_idx]]),
                        "normalized_target_norm": float(den.numpy()[local_idx]),
                    }
                )
            runs.append(
                {
                    "seed": seed,
                    "mean_rel_l2": float(per_sample.mean()),
                    "median_rel_l2": float(np.median(per_sample)),
                    "n_outliers_gt3": int(outlier_mask.sum()),
                    "n_test": int(len(per_sample)),
                }
            )
        per_condition[condition] = runs

    summary = json.loads((OUT_DIR / "geometry_generalization_summary.json").read_text(encoding="utf-8"))
    for condition in CONDITIONS:
        medians = [r["median_rel_l2"] for r in per_condition[condition]]
        means = [r["mean_rel_l2"] for r in per_condition[condition]]
        summary[condition]["relative_l2_median_mean"] = mean(medians)
        summary[condition]["relative_l2_median_std"] = stdev(medians)
        summary[condition]["per_seed_median_rel_l2"] = per_condition[condition]
        assert abs(mean(means) - summary[condition]["relative_l2_mean"]) < 1e-4, "mean mismatch vs original run"

    random_medians = {r["seed"]: r["median_rel_l2"] for r in per_condition["random_split"]}
    holdout_medians = {r["seed"]: r["median_rel_l2"] for r in per_condition["geometry_holdout"]}
    diffs = [holdout_medians[s] - random_medians[s] for s in SEEDS]
    summary["geometry_holdout"]["paired_vs_random_split"]["relative_l2_median"] = paired_stats(diffs)
    summary["outlier_disclosure"] = {
        "note": (
            "Raw target stress magnitude varies ~60x across this dataset's sampled geometries "
            "(pressure_angle_deg strongly affects peak contact stress; see the FEM sanity check in "
            "GEOMETRY_BASELINE_STUDY_2026-08-31.md), far wider than any fixed-geometry dataset in "
            "this project. A handful of near-zero-target-norm test cases -- weakly loaded, "
            "low-stress geometry/loading draws whose FEM solves converged normally, not failures -- "
            "produce very large per-sample relative_l2 ratios (denominator near zero) that inflate "
            "the mean. The median is reported alongside the mean for this reason and is the primary "
            "comparison statistic for this study; every other table in this paper uses the mean "
            "because no other dataset has this magnitude of dynamic range."
        ),
        "outlier_threshold": OUTLIER_THRESHOLD,
        "outlier_cases": outlier_log,
    }
    (OUT_DIR / "geometry_generalization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[updated] {OUT_DIR / 'geometry_generalization_summary.json'}")
    for condition in CONDITIONS:
        print(
            f"{condition}: median_rel_l2 {summary[condition]['relative_l2_median_mean']:.4f} "
            f"+/- {summary[condition]['relative_l2_median_std']:.4f} "
            f"(mean {summary[condition]['relative_l2_mean']:.4f}, outlier-inflated)"
        )
    print(json.dumps(summary["geometry_holdout"]["paired_vs_random_split"]["relative_l2_median"], indent=2))


if __name__ == "__main__":
    main()

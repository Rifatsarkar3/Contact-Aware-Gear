"""Check whether a trained arm's predicted uncertainty is actually calibrated.

**Do NOT run this until GPU is free and run_active_learning_study.py has
produced saved checkpoints for the requested arm/round** -- see
docs/superpowers/specs/2026-08-04-uncertainty-guided-active-learning-design.md.

For each test sample: compute the ensemble mean prediction and the
cross-seed predictive variance (the arm's uncertainty signal -- for
mc_dropout, "cross-seed" here means across the 10 independently seeded
dropout models saved at that checkpoint, matching the accuracy-evaluation
protocol, not a single model's MC samples), and the true squared error of
the mean prediction against the FEM ground truth. A well-calibrated
uncertainty should be positively correlated with true error across test
samples: a Spearman rank correlation (computed from scratch, no new scipy
dependency, matching this project's existing pure-numpy approach in
scripts/run_posthoc_power_analysis.py) plus reliability-diagram numbers
(mean predicted variance vs. mean true error within variance quintiles).

Usage (once launched):
    python scripts/run_uq_calibration_check.py --method ensemble_surrogate --round 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.models import FNO2d  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

V1_DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v1.h5"
CONFIG = ROOT / "configs" / "smoke.yaml"
OUT_ROOT = ROOT / "outputs" / "active_learning"
SEEDS = list(range(1, 11))

# Same rationale/value as run_active_learning_study.py -- another job may
# already be resident on the GPU; cap our footprint to 20% of total device
# memory so we can never grow into memory it needs.
GPU_MEMORY_FRACTION = 0.20


def cap_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation from scratch -- Pearson correlation of the rank-transformed arrays."""
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def load_test_set():
    with h5py.File(V1_DATA, "r") as handle:
        inputs = np.asarray(handle["inputs"], dtype=np.float32)
        targets = np.asarray(handle["targets"], dtype=np.float32)
        trajectory_ids = np.asarray(handle["trajectory_ids"])
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    split = grouped_trajectory_split(
        trajectory_ids, train_fraction=float(cfg["train_fraction"]), val_fraction=float(cfg["val_fraction"]), seed=int(cfg["seed"]),
    )
    return inputs[split.test], targets[split.test], cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("ensemble_surrogate", "mc_dropout", "pool_based"))
    parser.add_argument("--round", type=int, required=True, choices=(2, 4, 6))
    args = parser.parse_args()

    cap_gpu_memory()
    checkpoint_dir = OUT_ROOT / args.method / f"round_{args.round}_models"
    results_path = OUT_ROOT / args.method / f"round_{args.round}_results.json"
    if not checkpoint_dir.exists() or not results_path.exists():
        raise FileNotFoundError(
            f"No saved checkpoints for method={args.method} round={args.round} -- "
            f"run scripts/run_active_learning_study.py --method {args.method} --rounds {args.round} first."
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    target_scales = {r["seed"]: r["target_scale"] for r in results["per_seed_results"]}
    dropout_rate = results.get("dropout_rate", 0.0)

    test_inputs, test_targets, cfg = load_test_set()
    model_cfg = dict(cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed_predictions = []
    for seed in SEEDS:
        model = FNO2d(width=model_cfg["width"], modes_x=model_cfg["modes_x"], modes_y=model_cfg["modes_y"], layers=model_cfg["layers"], dropout=dropout_rate)
        model.load_state_dict(torch.load(checkpoint_dir / f"seed{seed}.pt", map_location="cpu"))
        model.to(device).eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(test_inputs).float().to(device)).cpu().numpy()
        per_seed_predictions.append(prediction * target_scales[seed])  # de-normalize back to native units

    stacked = np.stack(per_seed_predictions, axis=0)  # (n_seeds, n_test, 3, H, W)
    ensemble_mean = stacked.mean(axis=0)
    predicted_variance_per_sample = stacked.var(axis=0, ddof=1).mean(axis=(1, 2, 3))  # (n_test,)
    true_squared_error_per_sample = ((ensemble_mean - test_targets) ** 2).mean(axis=(1, 2, 3))  # (n_test,)

    correlation = spearman_correlation(predicted_variance_per_sample, true_squared_error_per_sample)

    quintile_edges = np.quantile(predicted_variance_per_sample, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    reliability = []
    for lo, hi in zip(quintile_edges[:-1], quintile_edges[1:]):
        mask = (predicted_variance_per_sample >= lo) & (predicted_variance_per_sample <= hi)
        if mask.sum() == 0:
            continue
        reliability.append({
            "predicted_variance_mean": float(predicted_variance_per_sample[mask].mean()),
            "true_error_mean": float(true_squared_error_per_sample[mask].mean()),
            "n_samples": int(mask.sum()),
        })

    out = {
        "method": args.method,
        "round": args.round,
        "spearman_correlation_variance_vs_error": correlation,
        "reliability_diagram_quintiles": reliability,
        "n_test_samples": int(test_targets.shape[0]),
    }
    out_path = OUT_ROOT / args.method / f"round_{args.round}_calibration.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()

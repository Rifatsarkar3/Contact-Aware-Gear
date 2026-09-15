"""POD/reduced-basis regression: a fundamentally different paradigm from every FNO variant tried.

Flagged repeatedly (`docs/CCM_FNO_STUDY_2026-07-24.md`,
`docs/DATASET_SCALE_STUDY_2026-07-24.md`, `memory.md`) as an unattempted,
structurally different alternative: instead of a deep operator network,
fit a low-dimensional linear (POD) basis to the training stress fields and
regress each mode's coefficient directly from the scalar physical
condition vector (indentation, friction, Young's modulus, contact
fraction, phase, time, speed) with ridge regression. No spatial network at
all -- this is the classical reduced-order-model approach from computational
mechanics, evaluated with the exact same metrics and split used throughout
this project so it is directly comparable to every FNO-family result.

Unlike the FNO family, this baseline has no training-seed randomness (POD
via SVD and ridge regression are both deterministic given the data and a
fixed regularization strength), so there is a single number per metric,
not a 10-seed mean +/- std. The number of POD modes and the ridge alpha
are both selected on the validation split only, mirroring the leakage-safe
protocol used for every architecture search in this project.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.losses import equilibrium_residual, hotspot_mae, relative_l2  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

N_MODES_GRID = (5, 10, 20, 40, 89)
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def fit_pod_basis(train_targets_flat: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    mean_field = train_targets_flat.mean(axis=0)
    centered = train_targets_flat - mean_field
    # economy SVD: centered = U @ diag(S) @ Vt, modes are Vt's leading rows
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    modes = vt[:n_modes]  # (n_modes, n_features)
    return mean_field, modes


def evaluate_metrics(
    prediction_flat: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    dx: float,
    dy: float,
    shape: tuple[int, int, int],
) -> dict[str, float]:
    prediction = torch.from_numpy(prediction_flat.reshape((-1, *shape)).astype(np.float32))
    target_t = torch.from_numpy(targets.astype(np.float32))
    mask_t = torch.from_numpy(masks.astype(np.float32))
    return {
        "relative_l2": float(relative_l2(prediction, target_t)),
        "hotspot_mae": float(hotspot_mae(prediction, target_t)),
        "equilibrium_residual": float(equilibrium_residual(prediction, mask_t, dx=dx, dy=dy)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v2.h5")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs" / "smoke" / args.data.stem / "pod_baseline_results.json"

    with h5py.File(args.data, "r") as handle:
        inputs = np.asarray(handle["inputs"], dtype=np.float32)
        targets = np.asarray(handle["targets"], dtype=np.float32)
        conditions = np.asarray(handle["conditions"], dtype=np.float64)
        trajectory_ids = np.asarray(handle["trajectory_ids"])
        grid_dx = float(handle.attrs.get("grid_dx_mm", 1.0))
        grid_dy = float(handle.attrs.get("grid_dy_mm", 1.0))

    split = grouped_trajectory_split(trajectory_ids, train_fraction=0.625, val_fraction=0.1875, seed=42)
    target_scale = max(float(np.max(np.abs(targets[split.train]))), 1e-8)
    targets_norm = targets / target_scale
    shape = targets_norm.shape[1:]  # (3, 64, 64)
    n_features = int(np.prod(shape))

    train_flat = targets_norm[split.train].reshape(len(split.train), n_features).astype(np.float64)
    val_flat = targets_norm[split.val].reshape(len(split.val), n_features).astype(np.float64)
    test_flat = targets_norm[split.test].reshape(len(split.test), n_features).astype(np.float64)

    # Standardize the condition vector using train statistics only.
    cond_mean = conditions[split.train].mean(axis=0)
    cond_std = conditions[split.train].std(axis=0)
    cond_std[cond_std < 1e-8] = 1.0
    cond_train = (conditions[split.train] - cond_mean) / cond_std
    cond_val = (conditions[split.val] - cond_mean) / cond_std
    cond_test = (conditions[split.test] - cond_mean) / cond_std

    masks = inputs[:, :1]

    search = []
    best = None
    for n_modes in N_MODES_GRID:
        mean_field, pod_modes = fit_pod_basis(train_flat, n_modes)
        coeff_train = (train_flat - mean_field) @ pod_modes.T
        coeff_val_target = (val_flat - mean_field) @ pod_modes.T
        for alpha in ALPHA_GRID:
            regressor = Ridge(alpha=alpha)
            regressor.fit(cond_train, coeff_train)
            coeff_val_pred = regressor.predict(cond_val)
            val_recon = mean_field + coeff_val_pred @ pod_modes
            val_metrics = evaluate_metrics(val_recon, targets_norm[split.val], masks[split.val], grid_dx, grid_dy, shape)
            score = val_metrics["relative_l2"] + 2.0 * val_metrics["hotspot_mae"]
            search.append({"n_modes": n_modes, "alpha": alpha, "val_score": score, "val_metrics": val_metrics})
            print(f"n_modes={n_modes} alpha={alpha}: val_score={score:.4f} {val_metrics}")
            if best is None or score < best["val_score"]:
                best = {"n_modes": n_modes, "alpha": alpha, "val_score": score}

    print(f"[selected] n_modes={best['n_modes']} alpha={best['alpha']}")
    mean_field, pod_modes = fit_pod_basis(train_flat, best["n_modes"])
    coeff_train = (train_flat - mean_field) @ pod_modes.T
    regressor = Ridge(alpha=best["alpha"])
    regressor.fit(cond_train, coeff_train)

    coeff_test_pred = regressor.predict(cond_test)
    test_recon = mean_field + coeff_test_pred @ pod_modes
    test_metrics = evaluate_metrics(test_recon, targets_norm[split.test], masks[split.test], grid_dx, grid_dy, shape)

    explained_variance = float(1.0 - np.sum((train_flat - (mean_field + coeff_train @ pod_modes)) ** 2) / np.sum((train_flat - mean_field) ** 2))

    summary = {
        "model": "pod_ridge_baseline",
        "selected_n_modes": best["n_modes"],
        "selected_alpha": best["alpha"],
        "train_pod_explained_variance": explained_variance,
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "test": test_metrics,
        "search": search,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"selected": {"n_modes": best["n_modes"], "alpha": best["alpha"]}, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()

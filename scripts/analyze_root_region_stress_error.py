"""Does the headline v2 relative-L2 result also hold on a design-relevant
metric, or only on a generic field-error norm?

An ARS Stage-3 domain review flagged that this paper's metrics (relative L2,
hotspot MAE) are generic field-error norms, not the root-fillet stress
concentration a gear designer actually checks (ISO 6336 / AGMA-convention
root bending stress lives at the tooth root fillet specifically). This
dataset's target channels are (sigma_xx, sigma_yy, sigma_xy) -- not a full
3D stress tensor, so a true ISO 6336 root-bending-stress number isn't
available here. What *is* directly computable from the existing predictions:
a 2D (plane) von Mises-equivalent stress proxy,
sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2), restricted to the root-fillet band
of the grid (the region right above the fixed geometry's root radius), and
its peak value -- the single number a fillet-stress check would actually
look at.

This script reuses the already-trained v2 checkpoints (fno, aee_fno,
dwm_fno; 10 model-init seeds each, split seed 42) -- no retraining -- runs
inference once on the held-out test split, and reports the relative error of
predicted vs. true peak root-region von Mises stress per case, aggregated
per model and paired against vanilla FNO exactly like the headline
relative_l2 comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median, stdev

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.inventions import AdaptiveEquilibriumExchangeFNO, DualWindowMultiscaleFNO  # noqa: E402
from gearstress.models import FNO2d  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v2.h5"
CFG = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
SEEDS = list(range(1, 11))

# Fixed geometry used by every case in this dataset (module=2.0mm, teeth=24,
# root_clearance_module=0.25) -- see fem.py:project_case_to_grid for the
# same formulas applied per-case for the geometry-varying dataset.
MODULE_MM = 2.0
TEETH = 24.0
ROOT_CLEARANCE_MODULE = 0.25
PITCH_RADIUS_MM = 0.5 * MODULE_MM * TEETH
ROOT_RADIUS_MM = PITCH_RADIUS_MM - MODULE_MM * (1.0 + ROOT_CLEARANCE_MODULE)
OUTER_RADIUS_MM = PITCH_RADIUS_MM + MODULE_MM
MARGIN_BELOW_MM = 1.0 * MODULE_MM
MARGIN_ABOVE_MM = 0.10 * MODULE_MM
Y_AXIS = np.linspace(ROOT_RADIUS_MM - MARGIN_BELOW_MM, OUTER_RADIUS_MM + MARGIN_ABOVE_MM, 64, dtype=np.float32)
ROOT_BAND_MM = MODULE_MM  # root-fillet band: root radius up to +1 module
ROOT_ROWS = Y_AXIS <= (ROOT_RADIUS_MM + ROOT_BAND_MM)


def von_mises_2d(sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2, 0.0, None))


def build_model(name: str, dx: float, dy: float) -> torch.nn.Module:
    w = CFG["model"]
    if name == "fno":
        return FNO2d(width=w["width"], modes_x=w["modes_x"], modes_y=w["modes_y"], layers=w["layers"])
    if name == "aee_fno":
        return AdaptiveEquilibriumExchangeFNO(
            width=w["width"], modes_x=w["modes_x"], modes_y=w["modes_y"], layers=w["layers"], dx=dx, dy=dy
        )
    if name == "dwm_fno":
        return DualWindowMultiscaleFNO(
            width=w["width"], modes_x=w["modes_x"], modes_y=w["modes_y"], layers=w["layers"],
            local_width=8, local_window=0.5, local_size=16, gate_sigma_fraction=0.25, softmax_temperature=0.1,
        )
    raise ValueError(name)


def run_dir(model: str, seed: int) -> Path:
    suffix = f"_seed{seed}_ep300"
    base = "dwm_fno_lw8_lwin0.5_lsz16" if model == "dwm_fno" else model
    return ROOT / "outputs" / "smoke" / DATA.stem / f"{base}{suffix}"


def main() -> None:
    with h5py.File(DATA, "r") as f:
        inputs = np.asarray(f["inputs"], dtype=np.float32)
        targets_raw = np.asarray(f["targets"], dtype=np.float32)  # physical units, unnormalized
        trajectory_ids = np.asarray(f["trajectory_ids"])
        grid_dx_mm = float(f.attrs["grid_dx_mm"])
        grid_dy_mm = float(f.attrs["grid_dy_mm"])

    split = grouped_trajectory_split(
        trajectory_ids, train_fraction=float(CFG["train_fraction"]), val_fraction=float(CFG["val_fraction"]), seed=42
    )
    test_idx = split.test
    mask = inputs[test_idx, 0] > 0.5  # (n_test, 64, 64)
    region = mask & ROOT_ROWS[None, :, None]

    true_sxx, true_syy, true_sxy = targets_raw[test_idx, 0], targets_raw[test_idx, 1], targets_raw[test_idx, 2]
    true_vm = von_mises_2d(true_sxx, true_syy, true_sxy)
    true_vm_masked = np.where(region, true_vm, -np.inf)
    true_peak = true_vm_masked.max(axis=(1, 2))  # (n_test,)

    device = torch.device("cpu")
    results: dict[str, dict] = {}
    for model_name in ("fno", "aee_fno", "dwm_fno"):
        per_seed_rel_err = []
        for seed in SEEDS:
            rd = run_dir(model_name, seed)
            result = json.loads((rd / "results.json").read_text(encoding="utf-8"))
            target_scale = float(result["target_scale_mpa_or_native"])
            model = build_model(model_name, grid_dx_mm, grid_dy_mm)
            model.load_state_dict(torch.load(rd / "model.pt", map_location=device))
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(inputs[test_idx])).numpy() * target_scale
            pred_sxx, pred_syy, pred_sxy = pred[:, 0], pred[:, 1], pred[:, 2]
            pred_vm = von_mises_2d(pred_sxx, pred_syy, pred_sxy)
            pred_vm_masked = np.where(region, pred_vm, -np.inf)
            pred_peak = pred_vm_masked.max(axis=(1, 2))
            rel_err = np.abs(pred_peak - true_peak) / np.maximum(np.abs(true_peak), 1e-6)
            per_seed_rel_err.append(float(np.mean(rel_err)))
        results[model_name] = {
            "per_seed_mean_rel_err": per_seed_rel_err,
            "mean": mean(per_seed_rel_err),
            "median": median(per_seed_rel_err),
            "std": stdev(per_seed_rel_err),
        }

    baseline = results["fno"]["per_seed_mean_rel_err"]
    for model_name in ("aee_fno", "dwm_fno"):
        diffs = [a - b for a, b in zip(results[model_name]["per_seed_mean_rel_err"], baseline)]
        m = mean(diffs)
        s = stdev(diffs)
        se = s / (len(diffs) ** 0.5)
        t = m / se if se > 0 else float("inf")
        results[model_name]["paired_vs_fno"] = {"mean_diff": m, "sd_diff": s, "t": t, "n": len(diffs)}

    results["_metadata"] = {
        "n_test_cases": int(len(test_idx)),
        "root_radius_mm": ROOT_RADIUS_MM,
        "root_band_mm": ROOT_BAND_MM,
        "root_band_upper_mm": ROOT_RADIUS_MM + ROOT_BAND_MM,
        "n_root_rows_of_64": int(ROOT_ROWS.sum()),
        "metric": "relative error of peak 2D von Mises-equivalent stress (sigma_xx, sigma_yy, sigma_xy only; sigma_zz not in target channels) within the root-fillet grid band, vs. FNO baseline",
    }

    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "root_region_stress_error_summary.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[summary written] {out_path}")
    for model_name in ("fno", "aee_fno", "dwm_fno"):
        r = results[model_name]
        print(f"{model_name}: mean {r['mean']:.4f} median {r['median']:.4f} std {r['std']:.4f}")
        if "paired_vs_fno" in r:
            print(f"  paired vs fno: {r['paired_vs_fno']}")


if __name__ == "__main__":
    main()

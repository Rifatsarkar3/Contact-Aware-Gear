"""Re-validate MICE-LPM's contact/hotspot weights and equilibrium budget on a
given dataset's own validation split, instead of reusing the values locked on
the 21-case dataset (`docs/GEAR_PAIR_TRANSIENT_QS_SMOKE_2026-07-16.md` found
that reuse produced a mixed, non-dominant tradeoff on the 63-case dataset).

Selection rule matches ``evaluate_mice_smoke.py``'s documented rule: minimize
validation ``relative_l2 + 2 * hotspot_mae`` subject to the achieved
validation equilibrium residual not exceeding the budget.
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

from gearstress.inventions import LoadPathModulatedFNO, MinimumInterventionContactEquilibrium  # noqa: E402
from gearstress.losses import equilibrium_residual, hotspot_mae, relative_l2  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "gear_pair_smoke_v1.h5")
    parser.add_argument("--weights", type=float, nargs="+", default=(0.0, 2.0, 4.0, 8.0))
    parser.add_argument("--budgets", type=float, nargs="+", default=(50.0, 100.0, 150.0, 250.0, 400.0))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    with h5py.File(args.data, "r") as handle:
        inputs = np.asarray(handle["inputs"], dtype=np.float32)
        targets = np.asarray(handle["targets"], dtype=np.float32)
        trajectory_ids = np.asarray(handle["trajectory_ids"])
        dx = float(handle.attrs.get("grid_dx_mm", 1.0))
        dy = float(handle.attrs.get("grid_dy_mm", 1.0))
    split = grouped_trajectory_split(
        trajectory_ids,
        train_fraction=float(cfg["train_fraction"]),
        val_fraction=float(cfg["val_fraction"]),
        seed=int(cfg["seed"]),
    )
    target_scale = max(float(np.max(np.abs(targets[split.train]))), 1e-8)
    targets = targets / target_scale

    model_cfg = cfg["model"]
    predictor = LoadPathModulatedFNO(
        width=model_cfg["width"], modes_x=model_cfg["modes_x"], modes_y=model_cfg["modes_y"], layers=model_cfg["layers"]
    )
    checkpoint = ROOT / "outputs" / "smoke" / args.data.stem / "lpm_fno" / "model.pt"
    predictor.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    predictor.eval()

    with torch.no_grad():
        val_inputs = torch.from_numpy(inputs[split.val])
        val_targets = torch.from_numpy(targets[split.val])
        val_raw = predictor(val_inputs)
        test_inputs = torch.from_numpy(inputs[split.test])
        test_targets = torch.from_numpy(targets[split.test])
        test_raw = predictor(test_inputs)

    def evaluate(repair: MinimumInterventionContactEquilibrium, raw, batch_inputs, batch_targets) -> dict[str, float]:
        with torch.no_grad():
            prediction = repair(raw, batch_inputs[:, :1], batch_inputs[:, 7:8])
        return {
            "relative_l2": float(relative_l2(prediction, batch_targets)),
            "hotspot_mae": float(hotspot_mae(prediction, batch_targets)),
            "equilibrium_residual": float(equilibrium_residual(prediction, batch_inputs[:, :1], dx=dx, dy=dy)),
        }

    grid = []
    best = None
    for contact_weight in args.weights:
        for hotspot_weight in args.weights:
            for budget in args.budgets:
                repair = MinimumInterventionContactEquilibrium(
                    dx, dy, equilibrium_budget=budget, contact_weight=contact_weight, hotspot_weight=hotspot_weight
                )
                val = evaluate(repair, val_raw, val_inputs, val_targets)
                score = val["relative_l2"] + 2.0 * val["hotspot_mae"]
                meets_budget = val["equilibrium_residual"] <= budget * 1.001
                record = {
                    "contact_weight": contact_weight,
                    "hotspot_weight": hotspot_weight,
                    "equilibrium_budget": budget,
                    "val": val,
                    "selection_score": score,
                    "meets_budget": meets_budget,
                }
                grid.append(record)
                if meets_budget and (best is None or score < best["selection_score"]):
                    best = record

    if best is None:
        raise RuntimeError("no grid point met its own equilibrium budget")

    best_repair = MinimumInterventionContactEquilibrium(
        dx, dy,
        equilibrium_budget=best["equilibrium_budget"],
        contact_weight=best["contact_weight"],
        hotspot_weight=best["hotspot_weight"],
    )
    best_test = evaluate(best_repair, test_raw, test_inputs, test_targets)

    result = {
        "data": str(args.data),
        "selection_rule": "minimize validation relative_l2 + 2 * hotspot_mae subject to equilibrium <= budget",
        "weights_tried": list(args.weights),
        "budgets_tried": list(args.budgets),
        "grid_points": len(grid),
        "selected": {
            "contact_weight": best["contact_weight"],
            "hotspot_weight": best["hotspot_weight"],
            "equilibrium_budget": best["equilibrium_budget"],
            "validation": best["val"],
            "test": best_test,
        },
        "full_grid": grid,
    }
    out_dir = ROOT / "outputs" / "smoke" / args.data.stem / "mice_lpm_revalidated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()

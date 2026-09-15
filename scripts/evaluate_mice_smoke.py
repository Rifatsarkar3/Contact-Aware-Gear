"""Evaluate the validation-selected MICE layer on the locked scientific smoke split."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "processed" / "gear_pair_smoke_v1.h5"
    )
    parser.add_argument("--equilibrium-budget", type=float, default=100.0)
    parser.add_argument("--contact-weight", type=float, default=8.0)
    parser.add_argument("--hotspot-weight", type=float, default=2.0)
    parser.add_argument(
        "--lpm-run-name",
        type=str,
        default="lpm_fno",
        help="subdirectory under outputs/smoke/<data-stem>/ holding the frozen LPM-FNO checkpoint to repair",
    )
    parser.add_argument(
        "--output-run-name",
        type=str,
        default="mice_lpm",
        help="subdirectory under outputs/smoke/<data-stem>/ to write this run's results.json into",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    with h5py.File(args.data, "r") as handle:
        inputs = np.asarray(handle["inputs"], dtype=np.float32)
        targets = np.asarray(handle["targets"], dtype=np.float32)
        trajectory_ids = np.asarray(handle["trajectory_ids"])
        evidence_status = str(handle.attrs.get("evidence_status", "software smoke only"))
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
        width=model_cfg["width"],
        modes_x=model_cfg["modes_x"],
        modes_y=model_cfg["modes_y"],
        layers=model_cfg["layers"],
    )
    checkpoint = ROOT / "outputs" / "smoke" / args.data.stem / args.lpm_run_name / "model.pt"
    predictor.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    predictor.eval()
    repair = MinimumInterventionContactEquilibrium(
        dx,
        dy,
        equilibrium_budget=args.equilibrium_budget,
        contact_weight=args.contact_weight,
        hotspot_weight=args.hotspot_weight,
    )

    def evaluate(indices: np.ndarray) -> dict[str, float | list[float]]:
        batch_inputs = torch.from_numpy(inputs[indices])
        batch_targets = torch.from_numpy(targets[indices])
        with torch.no_grad():
            start = time.perf_counter()
            raw = predictor(batch_inputs)
            prediction = repair(raw, batch_inputs[:, :1], batch_inputs[:, 7:8])
            elapsed = time.perf_counter() - start
        assert repair.last_repair_fraction is not None
        fractions = repair.last_repair_fraction.cpu().tolist()
        return {
            "relative_l2": float(relative_l2(prediction, batch_targets)),
            "hotspot_mae": float(hotspot_mae(prediction, batch_targets)),
            "equilibrium_residual": float(
                equilibrium_residual(prediction, batch_inputs[:, :1], dx=dx, dy=dy)
            ),
            "repair_fraction_mean": float(np.mean(fractions)),
            "repair_fraction_min": float(np.min(fractions)),
            "repair_fraction_max": float(np.max(fractions)),
            "repair_fractions": fractions,
            "wall_seconds": elapsed,
        }

    result = {
        "evidence_status": evidence_status,
        "method": "MICE-LPM",
        "seed": int(cfg["seed"]),
        "selection_rule": (
            "minimize validation relative_l2 + 2 * hotspot_mae subject to "
            f"equilibrium <= {args.equilibrium_budget:g}"
        ),
        "equilibrium_budget": args.equilibrium_budget,
        "contact_weight": args.contact_weight,
        "hotspot_weight": args.hotspot_weight,
        "split_samples": {
            "train": len(split.train),
            "validation": len(split.val),
            "test": len(split.test),
        },
        "target_scale_mpa_or_native": target_scale,
        "grid_spacing_mm": {"dx": dx, "dy": dy},
        "validation": evaluate(split.val),
        "test": evaluate(split.test),
    }
    output = ROOT / "outputs" / "smoke" / args.data.stem / args.output_run_name
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Generate a geometry-varying mating-tooth dataset (pressure angle + root
clearance/fillet depth), to test whether the FNO baseline generalizes across
tooth geometry rather than only the single fixed geometry used everywhere
else in this project.

Every prior dataset (``gear_pair_smoke_v1.h5``,
``gear_pair_transient_quasistatic_v1/v2.h5``) fixes module_mm=2.0, teeth=24,
pressure_angle_deg=20.0, root_clearance_module=0.25 and only randomizes
loading/kinematic parameters (friction, indentation, Young's modulus, phase).
This script instead randomizes ``pressure_angle_deg`` (flank profile) and
``root_clearance_module`` (root fillet depth, the classic gear root-stress-
concentration parameter) per case, holding module/teeth/Young's modulus/
contact phase fixed so the case remains within the mesh/grid pipeline's
validated regime and so geometry is the isolated variable of interest.

Each case is an independent single static solve (no phase-dense trajectory
sub-sampling) -- transient/phase behavior at fixed geometry is already
covered by the *_transient_quasistatic_* datasets; this dataset's only job
is geometry diversity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.fem import (  # noqa: E402
    GearPairCase,
    build_gear_pair_case,
    contact_element_counts,
    project_case_to_grid,
    run_calculix,
)

PRESSURE_ANGLE_RANGE = (17.5, 22.5)
ROOT_CLEARANCE_RANGE = (0.20, 0.32)
PRESSURE_ANGLE_CENTER = sum(PRESSURE_ANGLE_RANGE) / 2.0
PRESSURE_ANGLE_HALF_RANGE = (PRESSURE_ANGLE_RANGE[1] - PRESSURE_ANGLE_RANGE[0]) / 2.0
ROOT_CLEARANCE_CENTER = sum(ROOT_CLEARANCE_RANGE) / 2.0
ROOT_CLEARANCE_HALF_RANGE = (ROOT_CLEARANCE_RANGE[1] - ROOT_CLEARANCE_RANGE[0]) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=120)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--mesh-size", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "processed" / "gear_pair_geometry_v1.h5"
    )
    parser.add_argument("--case-root", type=Path, default=ROOT / "outputs" / "gear_pair_geometry_dataset")
    parser.add_argument("--reuse-solved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument(
        "--ccx",
        type=Path,
        default=ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe",
    )
    args = parser.parse_args()
    if args.cases > args.max_cases:
        raise ValueError(f"requested {args.cases} cases exceeds --max-cases {args.max_cases}")
    if args.mesh_size <= 0:
        raise ValueError("mesh size must be positive")

    rng = np.random.default_rng(args.seed)
    inputs, targets, trajectory_ids, conditions = [], [], [], []
    pressure_angles, root_clearances = [], []
    records = []
    y_spans = []
    for case_index in range(args.cases):
        pressure_angle_deg = float(rng.uniform(*PRESSURE_ANGLE_RANGE))
        root_clearance_module = float(rng.uniform(*ROOT_CLEARANCE_RANGE))
        friction = float(rng.uniform(0.04, 0.12))
        indentation = float(rng.uniform(0.018, 0.030))

        case = GearPairCase(
            pressure_angle_deg=pressure_angle_deg,
            root_clearance_module=root_clearance_module,
            friction=friction,
            indentation_mm=indentation,
            contact_fraction=0.52,
            mesh_size_mm=args.mesh_size,
        )
        case_dir = args.case_root / f"case_{case_index:04d}"
        deck = case_dir / "case.inp"
        metadata_path = case_dir / "case.json"
        reusable = args.reuse_solved and deck.with_suffix(".frd").exists() and (case_dir / "solver.log").exists() and metadata_path.exists()
        if reusable:
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            reusable = stored.get("model") == "deformable_involute_tooth_pair" and all(
                np.isclose(float(stored.get(key, np.nan)), float(value), rtol=1e-10, atol=1e-12)
                for key, value in asdict(case).items()
            )
        if not reusable:
            deck = build_gear_pair_case(case, case_dir)
            result = run_calculix(deck, args.ccx, timeout_seconds=600)
            if result.returncode != 0:
                raise RuntimeError(f"gear-pair solve failed at case {case_index}")
        counts = contact_element_counts(case_dir / "solver.log")
        if not counts or max(counts) <= 0:
            raise RuntimeError(f"inactive contact at case {case_index}")

        grid = project_case_to_grid(case_dir, args.grid_size)
        x_span = max(float(np.ptp(grid["x"])), 1e-6)
        y_span = max(float(np.ptp(grid["y"])), 1e-6)
        y_spans.append(y_span)
        x_norm = (grid["x"] - grid["x"].mean()) / x_span
        y_norm = (grid["y"] - grid["y"].mean()) / y_span
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        contact_x, contact_y = metadata["contact_point_mm"]
        contact_sigma_mm = 0.30
        contact_map = np.exp(
            -((grid["x"] - contact_x) ** 2 + (grid["y"] - contact_y) ** 2) / (2.0 * contact_sigma_mm**2)
        ).astype(np.float32)
        contact_map *= grid["mask"]
        scalar_condition = np.asarray(
            [
                (pressure_angle_deg - PRESSURE_ANGLE_CENTER) / PRESSURE_ANGLE_HALF_RANGE,
                (root_clearance_module - ROOT_CLEARANCE_CENTER) / ROOT_CLEARANCE_HALF_RANGE,
                indentation / 0.025,
                friction / 0.10,
            ],
            dtype=np.float32,
        )
        condition_maps = np.broadcast_to(
            scalar_condition[:, None, None],
            (4, args.grid_size, args.grid_size),
        )
        feature = np.concatenate(
            (
                grid["mask"][None],
                x_norm[None],
                y_norm[None],
                condition_maps,
                contact_map[None],
            ),
            axis=0,
        )
        condition = np.asarray(
            [pressure_angle_deg, root_clearance_module, indentation, friction, case.contact_fraction],
            dtype=np.float32,
        )
        inputs.append(feature.astype(np.float32))
        targets.append(grid["stress"])
        trajectory_ids.append(case_index)
        conditions.append(condition)
        pressure_angles.append(pressure_angle_deg)
        root_clearances.append(root_clearance_module)
        record = {
            "case_index": case_index,
            "case_dir": str(case_dir.relative_to(ROOT)),
            "mesh_size_mm": args.mesh_size,
            "pressure_angle_deg": pressure_angle_deg,
            "root_clearance_module": root_clearance_module,
            "friction": friction,
            "indentation_mm": indentation,
            "max_contact_elements": max(counts),
            "peak_abs_stress_mpa": float(np.max(np.abs(grid["stress"]))),
        }
        records.append(record)
        print(json.dumps(record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dx = 7.0 / max(args.grid_size - 1, 1)
    dy = float(np.mean(y_spans)) / max(args.grid_size - 1, 1)
    with h5py.File(args.output, "w") as handle:
        handle.attrs["evidence_status"] = (
            "deformable involute tooth-pair scientific smoke, geometry-varying "
            "(pressure_angle_deg, root_clearance_module); not full transient evidence"
        )
        handle.attrs["fem_model"] = "two deformable involute tooth sectors, plane strain, nonlinear frictional contact"
        handle.attrs["seed"] = args.seed
        handle.attrs["mesh_size_mm"] = args.mesh_size
        handle.attrs["grid_dx_mm"] = dx
        handle.attrs["grid_dy_mm"] = dy
        handle.attrs["grid_dy_mm_note"] = (
            "averaged across cases: the grid y-window scales with root_clearance_module, "
            f"so per-case dy varies by roughly +/-{100 * (max(y_spans) - min(y_spans)) / (2 * np.mean(y_spans)):.1f}% "
            "around this mean; equilibrium_residual is therefore an approximate diagnostic on this "
            "dataset, not the primary metric (relative_l2 is unaffected, since it is dimensionless)"
        )
        handle.attrs["pressure_angle_range_deg"] = list(PRESSURE_ANGLE_RANGE)
        handle.attrs["root_clearance_range"] = list(ROOT_CLEARANCE_RANGE)
        handle.create_dataset("inputs", data=np.stack(inputs), compression="gzip", shuffle=True)
        handle.create_dataset("targets", data=np.stack(targets), compression="gzip", shuffle=True)
        handle.create_dataset("trajectory_ids", data=np.asarray(trajectory_ids, dtype=np.int32))
        handle.create_dataset("conditions", data=np.stack(conditions))
        handle.create_dataset("pressure_angle_deg", data=np.asarray(pressure_angles, dtype=np.float32))
        handle.create_dataset("root_clearance_module", data=np.asarray(root_clearances, dtype=np.float32))
    args.case_root.mkdir(parents=True, exist_ok=True)
    (args.case_root / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(inputs)} converged geometry-varying mating-tooth contact cases")


if __name__ == "__main__":
    main()

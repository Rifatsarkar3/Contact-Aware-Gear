"""Generate the capped seed-42 deformable mating-tooth smoke dataset."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=7)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--mesh-size", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "gear_pair_smoke_v1.h5")
    parser.add_argument("--case-root", type=Path, default=ROOT / "outputs" / "gear_pair_smoke_dataset")
    parser.add_argument("--reuse-solved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ccx",
        type=Path,
        default=ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe",
    )
    args = parser.parse_args()
    if args.trajectories * args.frames > 40:
        raise ValueError("the FEM smoke dataset is capped at 40 cases")
    if args.mesh_size <= 0:
        raise ValueError("mesh size must be positive")

    rng = np.random.default_rng(args.seed)
    inputs, targets, trajectory_ids, conditions = [], [], [], []
    records = []
    for trajectory in range(args.trajectories):
        friction = float(rng.uniform(0.04, 0.12))
        youngs = float(rng.uniform(195_000.0, 215_000.0))
        base_indent = float(rng.uniform(0.018, 0.030))
        direction = -1.0 if trajectory % 2 else 1.0
        for frame in range(args.frames):
            phase = frame / max(args.frames - 1, 1)
            directed_phase = phase if direction > 0 else 1.0 - phase
            contact_fraction = float(0.38 + 0.24 * directed_phase)
            indentation = float(base_indent * (0.85 + 0.30 * np.sin(np.pi * phase)))
            case = GearPairCase(
                friction=friction,
                youngs_modulus_mpa=youngs,
                indentation_mm=indentation,
                contact_fraction=contact_fraction,
                mesh_size_mm=args.mesh_size,
            )
            case_dir = args.case_root / f"trajectory_{trajectory:03d}" / f"frame_{frame:02d}"
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
                    raise RuntimeError(f"gear-pair solve failed at trajectory {trajectory}, frame {frame}")
            counts = contact_element_counts(case_dir / "solver.log")
            if not counts or max(counts) <= 0:
                raise RuntimeError(f"inactive contact at trajectory {trajectory}, frame {frame}")

            grid = project_case_to_grid(case_dir, args.grid_size)
            x_span = max(float(np.ptp(grid["x"])), 1e-6)
            y_span = max(float(np.ptp(grid["y"])), 1e-6)
            x_norm = (grid["x"] - grid["x"].mean()) / x_span
            y_norm = (grid["y"] - grid["y"].mean()) / y_span
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            contact_x, contact_y = metadata["contact_point_mm"]
            contact_sigma_mm = 0.30
            contact_map = np.exp(
                -((grid["x"] - contact_x) ** 2 + (grid["y"] - contact_y) ** 2)
                / (2.0 * contact_sigma_mm**2)
            ).astype(np.float32)
            contact_map *= grid["mask"]
            scalar_condition = np.asarray(
                [
                    indentation / 0.025,
                    friction / 0.10,
                    youngs / 210_000.0,
                    2.0 * directed_phase - 1.0,
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
                [indentation, friction, youngs, contact_fraction, directed_phase],
                dtype=np.float32,
            )
            inputs.append(feature.astype(np.float32))
            targets.append(grid["stress"])
            trajectory_ids.append(trajectory)
            conditions.append(condition)
            record = {
                "trajectory": trajectory,
                "frame": frame,
                "case_dir": str(case_dir.relative_to(ROOT)),
                "mesh_size_mm": args.mesh_size,
                "contact_fraction": contact_fraction,
                "center_distance_mm": metadata["center_distance_mm"],
                "max_contact_elements": max(counts),
                "peak_abs_stress_mpa": float(np.max(np.abs(grid["stress"]))),
            }
            records.append(record)
            print(json.dumps(record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dx = 7.0 / max(args.grid_size - 1, 1)
    dy = 6.7 / max(args.grid_size - 1, 1)
    with h5py.File(args.output, "w") as handle:
        handle.attrs["evidence_status"] = "deformable involute tooth-pair scientific smoke; not full transient evidence"
        handle.attrs["fem_model"] = "two deformable involute tooth sectors, plane strain, nonlinear frictional contact"
        handle.attrs["seed"] = args.seed
        handle.attrs["mesh_size_mm"] = args.mesh_size
        handle.attrs["grid_dx_mm"] = dx
        handle.attrs["grid_dy_mm"] = dy
        handle.create_dataset("inputs", data=np.stack(inputs), compression="gzip", shuffle=True)
        handle.create_dataset("targets", data=np.stack(targets), compression="gzip", shuffle=True)
        handle.create_dataset("trajectory_ids", data=np.asarray(trajectory_ids, dtype=np.int32))
        handle.create_dataset("conditions", data=np.stack(conditions))
    args.case_root.mkdir(parents=True, exist_ok=True)
    (args.case_root / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(inputs)} converged mating-tooth contact cases")


if __name__ == "__main__":
    main()

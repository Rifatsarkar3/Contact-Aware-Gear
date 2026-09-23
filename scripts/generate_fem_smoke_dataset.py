"""Generate the allowed 32-case, seed-42 FEM smoke dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.fem import (  # noqa: E402
    ContactCase,
    build_case,
    contact_element_counts,
    project_case_to_grid,
    run_calculix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "fem_smoke_v0.h5")
    parser.add_argument("--case-root", type=Path, default=ROOT / "outputs" / "fem_smoke_dataset")
    parser.add_argument(
        "--ccx",
        type=Path,
        default=ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe",
    )
    args = parser.parse_args()
    if args.trajectories * args.frames > 40:
        raise ValueError("the FEM smoke dataset is capped at 40 cases")
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
            contact_fraction = float(0.38 + 0.24 * (phase if direction > 0 else 1.0 - phase))
            indentation = float(base_indent * (0.85 + 0.30 * np.sin(np.pi * phase)))
            case = ContactCase(
                friction=friction,
                youngs_modulus_mpa=youngs,
                indentation_mm=indentation,
                contact_fraction=contact_fraction,
                mesh_size_mm=0.70,
            )
            case_dir = args.case_root / f"trajectory_{trajectory:03d}" / f"frame_{frame:02d}"
            deck = build_case(case, case_dir)
            result = run_calculix(deck, args.ccx)
            counts = contact_element_counts(case_dir / "solver.log")
            if result.returncode != 0 or not counts or max(counts) <= 0:
                raise RuntimeError(f"invalid contact solve at trajectory {trajectory}, frame {frame}")
            grid = project_case_to_grid(case_dir, args.grid_size)
            x_norm = (grid["x"] - grid["x"].mean()) / max(float(np.ptp(grid["x"])), 1e-6)
            y_norm = (grid["y"] - grid["y"].mean()) / max(float(np.ptp(grid["y"])), 1e-6)
            condition = np.asarray(
                [indentation, friction, youngs / 210_000.0, contact_fraction, np.sin(np.pi * phase)],
                dtype=np.float32,
            )
            condition_maps = np.broadcast_to(condition[:, None, None], (5, args.grid_size, args.grid_size))
            feature = np.concatenate((grid["mask"][None], x_norm[None], y_norm[None], condition_maps), axis=0)
            inputs.append(feature.astype(np.float32))
            targets.append(grid["stress"])
            trajectory_ids.append(trajectory)
            conditions.append(condition)
            record = {"trajectory": trajectory, "frame": frame, "case_dir": str(case_dir.relative_to(ROOT)), "max_contact_elements": max(counts), "peak_abs_stress_mpa": float(np.max(np.abs(grid["stress"])))}
            records.append(record)
            print(json.dumps(record))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.attrs["evidence_status"] = "FEM software smoke only; local indenter approximation"
        handle.attrs["seed"] = args.seed
        handle.create_dataset("inputs", data=np.stack(inputs), compression="gzip", shuffle=True)
        handle.create_dataset("targets", data=np.stack(targets), compression="gzip", shuffle=True)
        handle.create_dataset("trajectory_ids", data=np.asarray(trajectory_ids, dtype=np.int32))
        handle.create_dataset("conditions", data=np.stack(conditions))
    (args.case_root / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(inputs)} converged contact cases")


if __name__ == "__main__":
    main()


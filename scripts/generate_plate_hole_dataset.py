"""Generate the plate-with-elliptical-hole dataset: the second, independent
domain used to test whether the gear-tooth study's central small-sample
architecture-misranking finding generalizes beyond gear-tooth contact.

Each case varies the hole's aspect ratio and absolute size (its shape) and
the applied far-field tension (its load), all held fixed everywhere else
(steel, E=207 GPa, nu=0.30 -- see plate_hole_fem.py's module docstring for
why material properties don't need to vary for this problem to be a
meaningful learning task). The FEM backend itself is validated separately in
scripts/validate_plate_hole_fem.py against Inglis's (1913) closed-form
stress-concentration formula (0.07-1.09% error across three aspect ratios).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.fem import run_calculix  # noqa: E402
from gearstress.plate_hole_fem import (  # noqa: E402
    PlateHoleCase,
    build_plate_hole_case,
    project_plate_hole_case_to_grid,
)

CCX = ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cases", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--window-mm", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    inputs, targets, conditions, records = [], [], [], []
    for i in range(args.n_cases):
        b_mm = float(rng.uniform(2.0, 4.0))
        aspect_ratio = float(rng.uniform(1.0, 3.0))
        a_mm = aspect_ratio * b_mm
        far_field_stress_mpa = float(rng.uniform(60.0, 160.0))
        case = PlateHoleCase(a_mm=a_mm, b_mm=b_mm, far_field_stress_mpa=far_field_stress_mpa)

        case_dir = args.case_root / f"case_{i:04d}"
        deck = build_plate_hole_case(case, case_dir)
        result = run_calculix(deck, CCX, timeout_seconds=120)
        if result.returncode != 0 or not (case_dir / "case.frd").exists():
            raise RuntimeError(f"case {i} FEM solve failed (returncode={result.returncode})")

        grid = project_plate_hole_case_to_grid(case_dir, args.grid_size, args.window_mm)
        window = args.window_mm
        axis = np.linspace(-window, window, args.grid_size, dtype=np.float32)
        x_norm = np.broadcast_to(axis[None, :], (args.grid_size, args.grid_size)) / window
        y_norm = np.broadcast_to(axis[:, None], (args.grid_size, args.grid_size)) / window

        hole_sigma_mm = 0.5
        location_map = np.exp(
            -((grid["x"] - a_mm) ** 2 + grid["y"] ** 2) / (2.0 * hole_sigma_mm**2)
        ) + np.exp(-((grid["x"] + a_mm) ** 2 + grid["y"] ** 2) / (2.0 * hole_sigma_mm**2))
        location_map = (location_map * grid["mask"]).astype(np.float32)

        scalar_condition = np.asarray(
            [
                aspect_ratio / 3.0,
                b_mm / 4.0,
                far_field_stress_mpa / 160.0,
                case.poisson_ratio,
            ],
            dtype=np.float32,
        )
        condition_maps = np.broadcast_to(scalar_condition[:, None, None], (4, args.grid_size, args.grid_size))
        feature = np.concatenate(
            (grid["mask"][None], x_norm[None], y_norm[None], condition_maps, location_map[None]),
            axis=0,
        )
        inputs.append(feature.astype(np.float32))
        targets.append(grid["stress"])
        conditions.append(np.asarray([a_mm, b_mm, aspect_ratio, far_field_stress_mpa], dtype=np.float32))
        record = {
            "case": i,
            "a_mm": a_mm,
            "b_mm": b_mm,
            "aspect_ratio": aspect_ratio,
            "far_field_stress_mpa": far_field_stress_mpa,
            "peak_abs_stress_mpa": float(np.max(np.abs(grid["stress"]))),
        }
        records.append(record)
        print(json.dumps(record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.attrs["evidence_status"] = (
            "second, independent domain (plate with elliptical hole, uniaxial tension, no contact) "
            "used to test generalization of the gear-tooth study's small-sample architecture-"
            "misranking finding; FEM backend validated against Inglis (1913) closed-form Kt, "
            "see scripts/validate_plate_hole_fem.py"
        )
        handle.attrs["fem_model"] = "linear elastic, plane strain, CPE6, single *STATIC step, no contact"
        handle.attrs["seed"] = args.seed
        handle.attrs["window_mm"] = args.window_mm
        handle.attrs["condition_columns"] = "a_mm, b_mm, aspect_ratio, far_field_stress_mpa"
        handle.create_dataset("inputs", data=np.stack(inputs), compression="gzip", shuffle=True)
        handle.create_dataset("targets", data=np.stack(targets), compression="gzip", shuffle=True)
        handle.create_dataset("conditions", data=np.stack(conditions))

    args.case_root.mkdir(parents=True, exist_ok=True)
    (args.case_root / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(inputs)} cases")


if __name__ == "__main__":
    main()

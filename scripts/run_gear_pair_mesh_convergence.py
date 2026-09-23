"""Mesh-convergence audit for the deformable involute tooth-pair smoke model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.fem import (  # noqa: E402
    GearPairCase,
    build_gear_pair_case,
    contact_element_counts,
    parse_final_nodal_stress,
    run_calculix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-sizes", type=float, nargs="+", default=(0.35, 0.25, 0.18, 0.12, 0.08))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "gear_pair_mesh_convergence")
    parser.add_argument("--reuse-solved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ccx",
        type=Path,
        default=ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe",
    )
    args = parser.parse_args()
    mesh_sizes = sorted(set(args.mesh_sizes), reverse=True)
    if any(size <= 0 for size in mesh_sizes):
        raise ValueError("mesh sizes must be positive")

    records = []
    for mesh_size in mesh_sizes:
        case = GearPairCase(mesh_size_mm=mesh_size)
        case_dir = args.output / f"h_{mesh_size:.3f}"
        deck = case_dir / "case.inp"
        reusable = args.reuse_solved and deck.with_suffix(".frd").exists() and (case_dir / "solver.log").exists()
        if not reusable:
            deck = build_gear_pair_case(case, case_dir)
            result = run_calculix(deck, args.ccx, timeout_seconds=1800)
            if result.returncode != 0:
                raise RuntimeError(f"mesh-convergence solve failed for h={mesh_size}")
        counts = contact_element_counts(case_dir / "solver.log")
        if not counts or max(counts) <= 0:
            raise RuntimeError(f"inactive contact for h={mesh_size}")
        stress = parse_final_nodal_stress(case_dir / "case.frd")
        mesh = np.load(case_dir / "mesh_data.npz")
        node_ids = mesh["node_ids"]
        coordinates = mesh["coordinates"]
        driver_ids = mesh["driver_node_ids"]
        coordinate_map = {int(tag): coordinates[i] for i, tag in enumerate(node_ids)}
        driver_stress = np.stack([stress[int(tag)] for tag in driver_ids])
        sxx, syy, szz, sxy, syz, szx = driver_stress.T
        von_mises = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy**2 + syz**2 + szx**2)
        )
        driver_xy = np.stack([coordinate_map[int(tag)][:2] for tag in driver_ids])
        radius = np.linalg.norm(driver_xy, axis=1)
        root_mask = (driver_xy[:, 1] <= case.base_radius_mm + 0.25) & (
            radius >= case.root_radius_mm + 0.05
        )
        hotspot_threshold = np.quantile(von_mises, 0.95)
        root_threshold = np.quantile(von_mises[root_mask], 0.95)
        metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        record = {
            "mesh_size_mm": mesh_size,
            "nodes": metadata["nodes"],
            "elements": metadata["elements"],
            "max_contact_elements": max(counts),
            "driver_field_p99_von_mises_mpa": float(np.quantile(von_mises, 0.99)),
            "driver_hotspot_top5_mean_von_mises_mpa": float(von_mises[von_mises >= hotspot_threshold].mean()),
            "driver_root_top5_mean_von_mises_mpa": float(von_mises[root_mask][von_mises[root_mask] >= root_threshold].mean()),
            "driver_root_peak_von_mises_mpa": float(von_mises[root_mask].max()),
        }
        records.append(record)
        print(json.dumps(record))

    finest = records[-1]
    metrics = (
        "driver_field_p99_von_mises_mpa",
        "driver_hotspot_top5_mean_von_mises_mpa",
        "driver_root_top5_mean_von_mises_mpa",
        "driver_root_peak_von_mises_mpa",
    )
    for record in records:
        for metric in metrics:
            record[f"{metric}_relative_difference_vs_finest"] = abs(record[metric] - finest[metric]) / finest[metric]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

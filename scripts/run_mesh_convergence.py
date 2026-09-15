"""Run the minimum mesh-convergence audit for the local contact smoke model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.fem import (  # noqa: E402
    ContactCase,
    build_case,
    contact_element_counts,
    parse_final_nodal_stress,
    run_calculix,
)


def main() -> None:
    ccx = ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe"
    output = ROOT / "outputs" / "mesh_convergence_smoke"
    records = []
    for mesh_size in (1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08):
        case_dir = output / f"h_{mesh_size:.2f}"
        deck = build_case(ContactCase(mesh_size_mm=mesh_size), case_dir)
        result = run_calculix(deck, ccx)
        counts = contact_element_counts(case_dir / "solver.log")
        if result.returncode != 0 or not counts or max(counts) <= 0:
            raise RuntimeError(f"mesh-convergence solve failed for h={mesh_size}")
        stress = parse_final_nodal_stress(case_dir / "case.frd")
        peak = max(float(np.abs(values[[0, 1, 3]]).max()) for values in stress.values())
        metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        mesh = np.load(case_dir / "mesh_data.npz")
        node_ids = mesh["node_ids"]
        coordinates = mesh["coordinates"]
        tooth_ids = mesh["tooth_node_ids"]
        coordinate_map = {int(tag): coordinates[i] for i, tag in enumerate(node_ids)}
        tooth_stress = np.stack([stress[int(tag)] for tag in tooth_ids])
        sxx, syy, szz, sxy, syz, szx = tooth_stress.T
        von_mises = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy**2 + syz**2 + szx**2)
        )
        tooth_xy = np.stack([coordinate_map[int(tag)][:2] for tag in tooth_ids])
        tooth_y = tooth_xy[:, 1]
        tooth_radius = np.linalg.norm(tooth_xy, axis=1)
        reference = ContactCase()
        # Exclude the fixed inner boundary, whose corner reactions create a
        # discretization singularity unrelated to the tooth-root hotspot.
        root_mask = (tooth_y <= reference.base_radius_mm + 0.25) & (
            tooth_radius >= reference.root_radius_mm + 0.05
        )
        root_peak = float(von_mises[root_mask].max())
        field_p99 = float(np.quantile(von_mises, 0.99))
        records.append(
            {
                "mesh_size_mm": mesh_size,
                "nodes": metadata["nodes"],
                "elements": metadata["elements"],
                "max_contact_elements": max(counts),
                "peak_abs_nodal_stress_mpa": peak,
                "root_peak_von_mises_mpa": root_peak,
                "tooth_field_p99_von_mises_mpa": field_p99,
            }
        )
    finest = records[-1]
    for record in records:
        for metric in ("peak_abs_nodal_stress_mpa", "root_peak_von_mises_mpa", "tooth_field_p99_von_mises_mpa"):
            record[f"{metric}_relative_difference_vs_finest"] = abs(record[metric] - finest[metric]) / finest[metric]
    (output / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

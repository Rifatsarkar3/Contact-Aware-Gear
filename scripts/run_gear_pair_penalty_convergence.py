"""Contact-penalty-stiffness sensitivity audit for the deformable involute tooth-pair model.

Stage E (`docs/`/`memory.md`) established mesh convergence at mesh_size=0.12mm
for the shared `build_gear_pair_case` contact formulation (still used
unchanged by both `gear_pair_transient_quasistatic_v1` and `v2`), but
explicitly flagged contact-penalty sensitivity as "not yet qualified". The
normal contact-penalty stiffness has been hardcoded at 50x the material's
Young's modulus (a numerical regularization parameter of the linear
penalty-overclosure contact formulation, not a physical quantity) since the
original 21-case dataset. This script sweeps it directly, mirroring
`run_gear_pair_mesh_convergence.py`'s structure and metrics exactly, to
check whether the stress field the operator is trained on is numerically
converged with respect to this parameter, the same way it already is with
respect to mesh size.
"""

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
    parser.add_argument("--penalty-factors", type=float, nargs="+", default=(10.0, 25.0, 50.0, 100.0, 200.0))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "gear_pair_penalty_convergence")
    parser.add_argument("--reuse-solved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ccx",
        type=Path,
        default=ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe",
    )
    args = parser.parse_args()
    penalty_factors = sorted(set(args.penalty_factors))
    if any(factor <= 0 for factor in penalty_factors):
        raise ValueError("penalty factors must be positive")

    records = []
    for factor in penalty_factors:
        case = GearPairCase(contact_penalty_factor=factor)
        case_dir = args.output / f"pf_{factor:.1f}"
        deck = case_dir / "case.inp"
        reusable = args.reuse_solved and deck.with_suffix(".frd").exists() and (case_dir / "solver.log").exists()
        if not reusable:
            deck = build_gear_pair_case(case, case_dir)
            result = run_calculix(deck, args.ccx, timeout_seconds=1800)
            if result.returncode != 0:
                raise RuntimeError(f"penalty-convergence solve failed for factor={factor}")
        counts = contact_element_counts(case_dir / "solver.log")
        if not counts or max(counts) <= 0:
            raise RuntimeError(f"inactive contact for factor={factor}")
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
            "contact_penalty_factor": factor,
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

    finest = records[-1]  # highest penalty stiffness approximates the rigid-contact limit
    metrics = (
        "driver_field_p99_von_mises_mpa",
        "driver_hotspot_top5_mean_von_mises_mpa",
        "driver_root_top5_mean_von_mises_mpa",
        "driver_root_peak_von_mises_mpa",
    )
    for record in records:
        for metric in metrics:
            record[f"{metric}_relative_difference_vs_stiffest"] = abs(record[metric] - finest[metric]) / finest[metric]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))

    default_record = next((r for r in records if r["contact_penalty_factor"] == 50.0), None)
    if default_record is not None:
        print("\n[decision-relevant] default factor=50.0 relative differences vs stiffest tested:")
        for metric in metrics:
            print(f"  {metric}: {default_record[f'{metric}_relative_difference_vs_stiffest']:.4%}")


if __name__ == "__main__":
    main()

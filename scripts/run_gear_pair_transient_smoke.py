"""One-case smoke check for the transient dynamic gear-pair FEM formulation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.fem import (  # noqa: E402
    GearPairTransientCase,
    build_gear_pair_transient_case,
    contact_element_counts,
    parse_nodal_force_history,
    parse_nodal_stress_history,
    run_calculix,
)

CCX = ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe"


def main() -> None:
    case = GearPairTransientCase(n_time_steps=6, mesh_size_mm=0.12)
    case_dir = ROOT / "outputs" / "gear_pair_transient_smoke" / "case_000"
    deck = build_gear_pair_transient_case(case, case_dir)
    result = run_calculix(deck, CCX, timeout_seconds=1800)
    if result.returncode != 0:
        raise RuntimeError(f"transient smoke solve failed, see {case_dir / 'solver.log'}")

    counts = contact_element_counts(case_dir / "solver.log")
    if not counts or min(counts[-case.n_time_steps :]) <= 0:
        raise RuntimeError("contact went inactive during the transient sweep")

    stress_history = parse_nodal_stress_history(case_dir / "case.frd")
    force_history = parse_nodal_force_history(case_dir / "case.frd")
    if len(stress_history) != case.n_time_steps or len(force_history) != case.n_time_steps:
        raise RuntimeError(
            f"expected {case.n_time_steps} output blocks, got "
            f"{len(stress_history)} stress / {len(force_history)} force"
        )

    mesh = np.load(case_dir / "mesh_data.npz")
    driver_root_ids = mesh["driver_root_ids"]
    node_ids = mesh["node_ids"]
    coordinates = mesh["coordinates"]
    coordinate_map = {int(tag): coordinates[i, :2] for i, tag in enumerate(node_ids)}

    torque_history = []
    for forces in force_history:
        torque = sum(
            coordinate_map[int(tag)][0] * forces[int(tag)][1]
            - coordinate_map[int(tag)][1] * forces[int(tag)][0]
            for tag in driver_root_ids
            if int(tag) in forces
        )
        torque_history.append(float(torque))

    driver_ids = mesh["driver_node_ids"]
    von_mises_history = []
    for stress in stress_history:
        driver_stress = np.stack([stress[int(tag)] for tag in driver_ids])
        sxx, syy, szz, sxy, syz, szx = driver_stress.T
        von_mises = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy**2 + syz**2 + szx**2)
        )
        von_mises_history.append(float(np.quantile(von_mises, 0.99)))

    if not np.all(np.isfinite(torque_history)):
        raise RuntimeError("non-finite torque in history")
    if not np.all(np.isfinite(von_mises_history)):
        raise RuntimeError("non-finite stress in history")
    if max(von_mises_history) < 1.0:
        raise RuntimeError("stress field is suspiciously near zero across the whole sweep")
    if np.ptp(von_mises_history) / max(von_mises_history) < 0.01:
        raise RuntimeError("stress field barely varies across time samples; sweep may be a no-op")

    summary = {
        "time_s": json.loads((case_dir / "case.json").read_text())["time_samples_s"][1:],
        "torque_reaction_n_mm": torque_history,
        "driver_field_p99_von_mises_mpa": von_mises_history,
        "max_contact_elements_per_step": counts[-case.n_time_steps :],
    }
    (case_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

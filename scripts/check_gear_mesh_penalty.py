"""Contact-penalty sensitivity of the corrected gear-mesh model.

Re-solves three positions of the first test trajectory (s = +0.88, 0 and -0.88
mm) with the normal penalty at 25E, 50E (production), 100E and 200E per mm,
0.12 mm elements and the 80-vertex flank, and compares normal load, peak
contact pressure, peak root tensile stress and the 64 x 64 field with the
200E solution.

Usage: python scripts/check_gear_mesh_penalty.py
Output: outputs/surrogate/gear_mesh_v4_penalty.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gearstress.fem as fem  # noqa: E402
from generate_gear_mesh_dataset import contact_metrics, job_list  # noqa: E402
from analyze_root_region_stress_error import ROOT_ROWS  # noqa: E402
from measure_label_uncertainty import find_ccx  # noqa: E402

FACTORS = (25.0, 50.0, 100.0, 200.0)
FRAMES = (1, 4, 7)
NEAR_MM = 1.0


def solve(job):
    idx, factor = job
    rows, *_ = job_list()
    r = rows[idx]
    case = fem.GearMeshCase(line_of_action_mm=r["line_of_action_mm"], approach_mm=r["approach_mm"],
                            friction=r["friction"], youngs_modulus_mpa=r["youngs_modulus_mpa"],
                            contact_penalty_factor=factor)
    with tempfile.TemporaryDirectory(prefix="pen_") as w:
        d = Path(w) / "case"
        deck = fem.build_gear_mesh_case(case, d)
        if fem.run_calculix(deck, find_ccx(None), timeout_seconds=3600).returncode != 0:
            raise RuntimeError(f"solve failed: case {idx}, {factor}E")
        meta = json.loads((d / "case.json").read_text(encoding="utf-8"))
        m = contact_metrics(deck, meta, case)
        g = fem.project_case_to_grid(d, 64)
    return idx, factor, m, g, meta["contact_point_mm"]


def sigma1(s):
    return 0.5 * (s[0] + s[1]) + np.sqrt(0.25 * (s[0] - s[1]) ** 2 + s[2] ** 2)


def main() -> None:
    rows, traj, test, _ = job_list()
    idxs = [test[f] for f in FRAMES]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(solve, [(i, f) for i in idxs for f in FACTORS]))
    by = {(i, f): (m, g, cp) for i, f, m, g, cp in res}
    out = {"factors": FACTORS, "cases": []}
    for i in idxs:
        _, g_ref, cp = by[(i, FACTORS[-1])]
        mask = g_ref["mask"] > 0.5
        gx, gy = g_ref["x"], g_ref["y"]
        far = mask & (np.hypot(gx - cp[0], gy - cp[1]) > NEAR_MM)
        band = mask & ROOT_ROWS[:, None]            # root circle to one module above it, as in the analysis
        ref = g_ref["stress"]
        root_ref = float(sigma1(ref)[band].max())
        entry = {"index": i, "line_of_action_mm": rows[i]["line_of_action_mm"], "by_factor": []}
        for f in FACTORS:
            m, g, _ = by[(i, f)]
            s = g["stress"]
            entry["by_factor"].append({
                "factor": f, "normal_force_N_per_mm": m["normal_force_N_per_mm"],
                "fe_peak_pressure_mpa": m["fe_peak_pressure_mpa"], "hertz_peak_pressure_mpa": m["hertz_peak_pressure_mpa"],
                "root_sigma1_mpa": float(sigma1(s)[band].max()),
                "root_change_vs_200E": float(sigma1(s)[band].max() / root_ref - 1),
                "field_rel_l2_vs_200E": float(np.linalg.norm((s - ref)[:, mask]) / np.linalg.norm(ref[:, mask])),
                "body_rel_l2_vs_200E": float(np.linalg.norm((s - ref)[:, far]) / np.linalg.norm(ref[:, far]))})
        out["cases"].append(entry)
        for e in entry["by_factor"]:
            print(f"case {i} s={entry['line_of_action_mm']:+.2f} {e['factor']:5.0f}E: F'={e['normal_force_N_per_mm']:.1f} "
                  f"p_FE/p_H={e['fe_peak_pressure_mpa'] / e['hertz_peak_pressure_mpa']:.3f} root {e['root_sigma1_mpa']:.1f} "
                  f"({100 * e['root_change_vs_200E']:+.2f}%) field {e['field_rel_l2_vs_200E']:.4f} body {e['body_rel_l2_vs_200E']:.4f}")
    (ROOT / "outputs" / "surrogate" / "gear_mesh_v4_penalty.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

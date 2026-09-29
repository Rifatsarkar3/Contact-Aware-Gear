"""Solver time of the corrected gear-mesh cases on an otherwise idle machine.

The dataset solves ran six at a time, which inflates CalculiX's reported time
through contention. This script re-solves one test trajectory (nine positions)
sequentially, one solve at a time, and records the time CalculiX reports.

Usage: python scripts/time_gear_mesh_solves.py
Output: outputs/surrogate/gear_mesh_solve_time_idle.json
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gearstress.fem as fem  # noqa: E402
from generate_gear_mesh_dataset import job_list  # noqa: E402
from measure_label_uncertainty import find_ccx  # noqa: E402


def main() -> None:
    rows, traj, test, jobs = job_list()
    ccx = find_ccx(None)
    times = []
    for i in test[:9]:
        r = rows[i]
        case = fem.GearMeshCase(line_of_action_mm=r["line_of_action_mm"], approach_mm=r["approach_mm"],
                                friction=r["friction"], youngs_modulus_mpa=r["youngs_modulus_mpa"])
        with tempfile.TemporaryDirectory(prefix="tm_") as w:
            d = Path(w) / "c"
            deck = fem.build_gear_mesh_case(case, d)
            if fem.run_calculix(deck, ccx, timeout_seconds=1800).returncode != 0:
                raise RuntimeError(f"solve failed: {i}")
            log = (d / "solver.log").read_text(encoding="utf-8", errors="ignore")
            t = float(log.rsplit("Total CalculiX Time:", 1)[1].split()[0])
        times.append(t)
        print(f"case {i}: {t:.1f} s", flush=True)
    out = {"cases": test[:9], "seconds": times, "median_s": statistics.median(times), "mean_s": statistics.fmean(times),
           "method": "one test trajectory re-solved sequentially on an idle machine (CalculiX single-threaded)"}
    (ROOT / "outputs" / "surrogate" / "gear_mesh_solve_time_idle.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"median {out['median_s']:.1f} s")


if __name__ == "__main__":
    main()

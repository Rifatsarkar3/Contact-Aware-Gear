"""Discretization uncertainty of the training labels, in the surrogate's own
metrics. Reproduces the headline label-uncertainty figures.

The v2 labels are CalculiX solutions at a mesh-convergent element size of
0.12 mm, not exact solutions. For eight of the 144 v2 dataset cases, sampled
evenly through the case list, this script re-solves each case at 0.12 mm and at a finer 0.08 mm,
projects both onto the same 64x64 grid used for training, and compares them in
exactly the metrics used to score the surrogate: relative L2, hotspot MAE, and
peak root-fillet von Mises stress. The 0.08 mm solution is the reference.

A surrogate error at or below this discrepancy cannot be distinguished from the
labels' own mesh error, which is why field accuracy is reported against this
floor rather than against zero.

A case whose solve at either mesh exceeds the solver time limit is recorded
as dropped, with the mesh that timed out, rather than silently skipped.
Solver scratch files are deleted after projection.

Needs CalculiX; pass its path with --ccx if it is not on PATH. Gmsh is used
through the Python package installed with this project.

Usage:
    python scripts/measure_label_uncertainty.py [--ccx /path/to/ccx]   # about 20 minutes

Writes outputs/surrogate/label_uncertainty.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_root_region_stress_error import ROOT_ROWS  # noqa: E402
from gearstress.fem import (GearPairCase, build_gear_pair_case,  # noqa: E402
                            project_case_to_grid, run_calculix)
from gearstress.losses import hotspot_mae, relative_l2  # noqa: E402

LOCAL_CCX = ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe"
CASES = ROOT / "outputs" / "gear_pair_transient_quasistatic_v2_dataset"
DATASET = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v2.h5"
OUT = ROOT / "outputs" / "surrogate" / "label_uncertainty.json"
PRODUCTION_MM, REFERENCE_MM = 0.120, 0.080
N_CASES = 8
TIMEOUT_S = 900


def von_mises(a: np.ndarray) -> np.ndarray:
    x, y, z = a[0], a[1], a[2]
    return np.sqrt(np.maximum(x * x - x * y + y * y + 3 * z * z, 0.0))


def find_ccx(given: Path | None) -> Path:
    """The CalculiX executable: --ccx, else the bundled Windows build, else PATH."""
    for cand in (given, LOCAL_CCX, shutil.which("ccx"), shutil.which("ccx_static")):
        if cand and Path(cand).exists():
            return Path(cand)
    raise FileNotFoundError("CalculiX not found: pass --ccx /path/to/ccx or put ccx on PATH")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discretization uncertainty of the training labels, in the surrogate's own metrics.")
    parser.add_argument("--ccx", type=Path, default=None, help="path to the CalculiX executable")
    ccx = find_ccx(parser.parse_args().ccx)

    # Sample only cases that are in the v2 dataset. The case folder also holds
    # an aborted, unused trajectory (trajectory_016) that never entered the
    # dataset; drawing from it would measure labels the model never saw.
    with h5py.File(DATASET, "r") as f:
        in_dataset = {f"trajectory_{int(t):03d}" for t in np.unique(f["trajectory_ids"][:])}
    sources = sorted(p for p in CASES.glob("trajectory_*/frame_*/case.json")
                     if p.parent.parent.name in in_dataset)
    if not sources:
        raise FileNotFoundError(f"no v2 dataset case definitions under {CASES}")
    picks = [sources[i] for i in np.linspace(0, len(sources) - 1, N_CASES).astype(int)]
    valid = {f.name for f in fields(GearPairCase)}

    rows, dropped = [], []
    with tempfile.TemporaryDirectory(prefix="label_uncertainty_") as work:
        for k, case_json in enumerate(picks):
            meta = json.loads(case_json.read_text(encoding="utf-8"))
            base = {key: meta[key] for key in meta if key in valid}
            grids, secs, reason = {}, {}, None
            for h in (PRODUCTION_MM, REFERENCE_MM):
                d = Path(work) / f"case{k:02d}_h{h:.3f}"
                d.mkdir(parents=True)
                try:
                    deck = build_gear_pair_case(GearPairCase(**{**base, "mesh_size_mm": h}), d)
                    result = run_calculix(deck, ccx, timeout_seconds=TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    reason = f"h={h} solve exceeded {TIMEOUT_S} s"
                    break
                if result.returncode != 0:
                    reason = f"h={h} solver returned {result.returncode}"
                    break
                grids[h] = project_case_to_grid(d, 64)
                log = (d / "solver.log").read_text(encoding="utf-8", errors="ignore")
                secs[h] = float(log.rsplit("Total CalculiX Time:", 1)[1].split()[0]) \
                    if "Total CalculiX Time:" in log else None
                shutil.rmtree(d)                         # drop the ~100 MB .frd
            label = str(case_json.parent.relative_to(ROOT))
            if reason:
                dropped.append({"case": label, "reason": reason})
                print(f"  [{k + 1}/{N_CASES}] dropped: {reason}", flush=True)
                continue

            ref, lab = grids[REFERENCE_MM], grids[PRODUCTION_MM]
            a = torch.from_numpy(ref["stress"])[None]
            b = torch.from_numpy(lab["stress"])[None]
            region = (ref["mask"] > 0.5) & (lab["mask"] > 0.5) & ROOT_ROWS[:, None]
            pa = np.where(region, von_mises(ref["stress"]), -np.inf).max()
            pb = np.where(region, von_mises(lab["stress"]), -np.inf).max()
            row = {"case": label,
                   "contact_fraction": meta.get("contact_fraction"),
                   "indentation_mm": meta.get("indentation_mm"),
                   "relative_l2": float(relative_l2(b, a)),
                   "hotspot_mae": float(hotspot_mae(b, a)),
                   "root_peak_relative": float(abs(pb - pa) / pa),
                   "solve_seconds": secs}
            rows.append(row)
            print(f"  [{k + 1}/{N_CASES}] rel_L2 {row['relative_l2']:.4f}  "
                  f"root-peak {row['root_peak_relative']:.4f}", flush=True)

    def stats(key):
        v = np.array([r[key] for r in rows])
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "min": float(v.min()), "max": float(v.max()),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else None}

    summary = {"production_mesh_mm": PRODUCTION_MM, "reference_mesh_mm": REFERENCE_MM,
               "population": f"{len(sources)} v2 dataset cases",
               "n_attempted": N_CASES, "n_completed": len(rows), "dropped": dropped,
               "relative_l2": stats("relative_l2"),
               "root_peak_relative": stats("root_peak_relative"),
               "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    r, p = summary["relative_l2"], summary["root_peak_relative"]
    print(f"\nn = {len(rows)} of {N_CASES}   relative L2 mean {r['mean']:.4f}  median {r['median']:.4f}  "
          f"range [{r['min']:.4f}, {r['max']:.4f}]")
    print(f"root peak mean {p['mean']:.4f}  median {p['median']:.4f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

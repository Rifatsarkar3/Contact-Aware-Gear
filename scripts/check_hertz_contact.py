"""Check the finite-element tooth contact against Hertz line-contact theory.

For a few v2 test cases, re-solve at the production (0.12 mm) and reference
(0.08 mm) element sizes and read the final contact state from the CalculiX
result file: normal contact pressure and frictional shear at every slave
surface node, and the reaction force at the fixed driver root. Hertz theory for
two cylinders with the involute radii of curvature at the contact point then
gives the half-width b and peak pressure p0 for the same force per unit face
width. The script reports both, the ratio of frictional shear to pressure where
the surface slides (which should equal the Coulomb coefficient), and how many
elements and 64x64 grid pixels span the contact width 2b.

Usage:
    python scripts/check_hertz_contact.py [--ccx PATH]
Output: outputs/surrogate/hertz_contact_check.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gearstress.fem import (GearPairCase, _parse_nodal_blocks, build_gear_pair_case,  # noqa: E402
                            parse_final_nodal_force, run_calculix)
from gearstress.splits import grouped_trajectory_split  # noqa: E402
from measure_label_uncertainty import find_ccx  # noqa: E402

CASES = ROOT / "outputs" / "gear_pair_transient_quasistatic_v2_dataset"
DATASET = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v2.h5"
OUT = ROOT / "outputs" / "surrogate" / "hertz_contact_check.json"
MESHES = (0.120, 0.080)
GRID_DX_MM = 0.1111


def read_inp(deck: Path) -> tuple[dict[int, np.ndarray], dict[str, list[int]]]:
    """Node coordinates and node sets from a CalculiX input deck."""
    nodes, sets, mode, cur = {}, {}, None, None
    for line in deck.read_text(encoding="ascii", errors="replace").splitlines():
        if line.startswith("*"):
            mode = None
            if line.upper().startswith("*NODE") and "FILE" not in line.upper() and "PRINT" not in line.upper():
                mode = "node"
            m = re.match(r"\*NSET,\s*NSET=(\w+)", line, re.I)
            if m:
                mode, cur = "nset", m.group(1).upper()
                sets[cur] = []
            continue
        if mode == "node":
            p = [s.strip() for s in line.split(",")]
            nodes[int(p[0])] = np.array([float(p[1]), float(p[2])])
        elif mode == "nset":
            sets[cur] += [int(s) for s in line.split(",") if s.strip()]
    return nodes, sets


def involute_radius_of_curvature(point: np.ndarray, center: np.ndarray, base_radius: float) -> float:
    r = float(np.linalg.norm(point - center))
    return math.sqrt(max(r * r - base_radius * base_radius, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ccx", type=Path, default=None)
    ccx = find_ccx(parser.parse_args().ccx)
    with h5py.File(DATASET, "r") as f:
        traj = np.asarray(f["trajectory_ids"])
    split = grouped_trajectory_split(traj, train_fraction=0.625, val_fraction=0.1875, seed=42)
    t0 = int(traj[sorted(split.test)[0]])
    picks = [(t0, fr) for fr in (1, 4, 7)]       # early, mid and late in one test trajectory
    valid = {fl.name for fl in fields(GearPairCase)}
    rows = []
    for t, fr in picks:
        meta = json.loads((CASES / f"trajectory_{t:03d}" / f"frame_{fr:02d}" / "case.json").read_text(encoding="utf-8"))
        base = {k: meta[k] for k in meta if k in valid}
        case = GearPairCase(**base)
        rb = 0.5 * case.module_mm * case.teeth * math.cos(math.radians(case.pressure_angle_deg))
        cp = np.asarray(meta["contact_point_mm"])
        rho1 = involute_radius_of_curvature(cp, np.zeros(2), rb)
        rho2 = involute_radius_of_curvature(cp, np.asarray(meta["mating_center_mm"]), rb)
        r_eff = rho1 * rho2 / (rho1 + rho2)
        e_star = case.youngs_modulus_mpa / (2.0 * (1.0 - case.poisson_ratio ** 2))
        for h in MESHES:
            with tempfile.TemporaryDirectory(prefix="hertz_") as work:
                d = Path(work)
                deck = build_gear_pair_case(GearPairCase(**{**base, "mesh_size_mm": h}), d)
                if run_calculix(deck, ccx, timeout_seconds=1200).returncode != 0:
                    raise RuntimeError(f"solve failed: traj {t} frame {fr} h {h}")
                frd = deck.with_suffix(".frd")
                nodes, sets = read_inp(deck)
                contact = _parse_nodal_blocks(frd, " -4  CONTACT", 6)[-1]
                force = parse_final_nodal_force(frd)
            f_vec = sum(force[n] for n in sets["DRIVERROOT"] if n in force)
            f_line = float(np.linalg.norm(f_vec[:2]))                   # N per mm face width
            b = math.sqrt(4.0 * f_line * r_eff / (math.pi * e_star))
            p0 = 2.0 * f_line / (math.pi * b)
            press = {n: v[3] for n, v in contact.items() if v[3] > 0}
            shear = {n: math.hypot(v[4], v[5]) for n, v in contact.items()}
            pmax = max(press.values())
            active = [n for n, p in press.items() if p > 0.01 * pmax]
            pts = np.array([nodes[n] for n in active])
            tangent = np.array([-meta["contact_normal"][1], meta["contact_normal"][0]])
            s = pts @ tangent
            width_fe = float(s.max() - s.min()) if len(s) > 1 else 0.0
            sliding = [shear[n] / press[n] for n in active if press[n] > 0.2 * pmax]
            row = {"trajectory": t, "frame": fr, "mesh_mm": h, "friction": case.friction,
                   "indentation_mm": case.indentation_mm, "force_per_mm_N": f_line,
                   "rho_driver_mm": rho1, "rho_mating_mm": rho2, "r_eff_mm": r_eff,
                   "hertz_half_width_mm": b, "hertz_peak_pressure_mpa": p0,
                   "fe_peak_pressure_mpa": float(pmax), "fe_contact_width_mm": width_fe,
                   "fe_active_contact_nodes": len(active),
                   "shear_over_pressure_median": float(np.median(sliding)) if sliding else None,
                   "elements_across_2b": 2 * b / h, "grid_pixels_across_2b": 2 * b / GRID_DX_MM}
            rows.append(row)
            print(f"traj {t} frame {fr} h={h}: F'={f_line:.1f} N/mm  Hertz b={b:.3f} mm p0={p0:.0f} MPa | "
                  f"FE pmax={pmax:.0f} MPa width={width_fe:.3f} mm nodes={len(active)} | "
                  f"tau/p={row['shear_over_pressure_median']} mu={case.friction:.3f} | "
                  f"2b/h={2 * b / h:.2f} 2b/pixel={2 * b / GRID_DX_MM:.2f}", flush=True)
    OUT.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

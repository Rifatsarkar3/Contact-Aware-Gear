"""Contact load path and a boundary-refined reference solution.

Two job families, solved in parallel (CalculiX runs single-threaded):

* load path: every v2 case at its own friction coefficient, and every test case
  at mu = 0.04 and 0.12, with the production discretization (0.12 mm elements,
  80-vertex flank). For each solve: the contact force per mm of face width (from
  the driver-root reactions) split along the true flank normal and tangent at the
  pressure-weighted contact centroid, the angle between that normal and the
  direction in which the mating rim is displaced (the deck displaces it along
  the line joining the closest vertices of the two flank polygons), the normal
  approach and tangential slip that this displacement imposes, the peak contact
  pressure, the Hertz half-width and peak pressure for the normal load, and the
  ratio of shear to mu times pressure over the loaded contact nodes.
* refined: the 27 test cases, and the three Hertz-table positions at
  mu = 0.04 and 0.12, with every length scale refined by 1.5: 0.08 mm elements
  AND a 120-vertex flank polygon (the production decks use 80 vertices, whose
  0.047 mm facets fix the boundary nodes at both 0.12 and 0.08 mm). Stress is
  projected onto the same 64 x 64 grid as the labels.

Usage:
    python scripts/analyze_contact_load_path.py --workers 6 --worker-id k   (k = 0..5)
    python scripts/analyze_contact_load_path.py --assemble
Outputs: outputs/surrogate/contact_load_path/*.json, outputs/surrogate/refined_flank_test/*.npz,
         outputs/surrogate/contact_load_path.json (assembled)
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gearstress.fem as fem  # noqa: E402
from check_hertz_contact import involute_radius_of_curvature, read_inp  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402
from measure_label_uncertainty import find_ccx  # noqa: E402

CASES = ROOT / "outputs" / "gear_pair_transient_quasistatic_v2_dataset"
V2 = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v2.h5"
JOBS = ROOT / "outputs" / "surrogate" / "contact_load_path"
REFINED = ROOT / "outputs" / "surrogate" / "refined_flank_test"
OUT = ROOT / "outputs" / "surrogate" / "contact_load_path.json"
FLANK_PROD, FLANK_REFINED = 80, 120
H_PROD, H_REFINED = 0.12, 0.08
ORIG_POLYGON = fem.gear_tooth_polygon


def job_list():
    with h5py.File(V2, "r") as f:
        traj = np.asarray(f["trajectory_ids"])
    split = grouped_trajectory_split(traj, train_fraction=0.625, val_fraction=0.1875, seed=42)
    test = sorted(int(i) for i in split.test)
    jobs = [("load", i, None) for i in range(len(traj))]
    jobs += [("load", i, mu) for i in test for mu in (0.04, 0.12)]
    jobs += [("refined", i, None) for i in test]
    jobs += [("refined", test[fr], mu) for fr in (1, 4, 7) for mu in (0.04, 0.12)]
    return traj, test, jobs


def meta_of(traj, idx):
    t = int(traj[idx])
    frame = int(idx - np.flatnonzero(traj == t).min())
    return t, frame, json.loads((CASES / f"trajectory_{t:03d}" / f"frame_{frame:02d}" / "case.json").read_text(encoding="utf-8"))


def name_of(kind, idx, mu):
    return f"{kind}_{idx:03d}_" + ("own" if mu is None else f"mu{mu:.2f}")


def contact_quantities(deck: Path, meta: dict, case) -> dict:
    frd = deck.with_suffix(".frd")
    nodes, sets = read_inp(deck)
    contact = fem._parse_nodal_blocks(frd, " -4  CONTACT", 6)[-1]
    force = fem.parse_final_nodal_force(frd)
    f_vec = -sum(force[n] for n in sets["DRIVERROOT"] if n in force)[:2]   # contact force on the driver tooth
    cp = np.asarray(meta["contact_point_mm"])
    press_all = {k: v[3] for k, v in contact.items() if v[3] > 0}
    pk = max(press_all.values())
    loaded_all = [k for k, p in press_all.items() if p > 0.01 * pk]
    centroid = sum(press_all[k] * nodes[k] for k in loaded_all) / sum(press_all[k] for k in loaded_all)
    # true outward flank normal: normal of the driver-polygon segment nearest the pressure centroid
    poly = fem.gear_tooth_polygon(case)
    seg_a, seg_b = poly, np.roll(poly, -1, axis=0)
    ab = seg_b - seg_a
    tt = np.clip(np.einsum("ij,ij->i", centroid - seg_a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
    k0 = int(np.argmin(np.linalg.norm(seg_a + tt[:, None] * ab - centroid, axis=1)))
    area2 = float(np.sum(seg_a[:, 0] * seg_b[:, 1] - seg_b[:, 0] * seg_a[:, 1]))   # > 0 for counter-clockwise
    d = ab[k0] / np.linalg.norm(ab[k0])
    n = np.array([d[1], -d[0]]) if area2 > 0 else np.array([-d[1], d[0]])       # outward from the driver
    t = np.array([-n[1], n[0]])
    if t @ (centroid / np.linalg.norm(centroid)) < 0:   # tangent pointing toward the tooth tip
        t = -t
    fn, ft = float(-(f_vec @ n)), float(f_vec @ t)
    n_disp = np.asarray(meta["contact_normal"], dtype=float)
    n_disp /= np.linalg.norm(n_disp)
    disp = -n_disp * case.indentation_mm                 # imposed displacement of the mating rim
    rb = case.pitch_radius_mm * math.cos(math.radians(case.pressure_angle_deg))
    rho1 = involute_radius_of_curvature(centroid, np.zeros(2), rb)
    rho2 = involute_radius_of_curvature(centroid, np.asarray(meta["mating_center_mm"]), rb)
    r_eff = rho1 * rho2 / (rho1 + rho2)
    e_star = case.youngs_modulus_mpa / (2.0 * (1.0 - case.poisson_ratio ** 2))
    b = math.sqrt(4.0 * fn * r_eff / (math.pi * e_star)) if fn > 0 else 0.0
    press = {k: v[3] for k, v in contact.items() if v[3] > 0}
    pmax = max(press.values()) if press else 0.0
    loaded = [k for k, p in press.items() if p > 0.2 * pmax]
    ratio = [math.hypot(contact[k][4], contact[k][5]) / (case.friction * press[k]) for k in loaded]
    return {"contact_radius_mm": float(np.linalg.norm(centroid)), "nominal_contact_radius_mm": float(np.linalg.norm(cp)),
            "force_vector_N_per_mm": f_vec.tolist(), "flank_normal": n.tolist(),
            "displacement_angle_to_flank_normal_deg": float(math.degrees(math.acos(min(1.0, abs(n_disp @ n))))),
            "imposed_normal_approach_mm": float(-(disp @ n)), "imposed_tangential_slip_toward_tip_mm": float(disp @ t),
            "normal_force_N_per_mm": fn,
            "tangential_force_toward_tip_N_per_mm": ft, "tangential_over_normal": ft / fn if fn else None,
            "r_eff_mm": r_eff, "hertz_half_width_mm": b,
            "hertz_peak_pressure_mpa": 2.0 * fn / (math.pi * b) if b else 0.0, "fe_peak_pressure_mpa": float(pmax),
            "active_nodes": sum(1 for p in press.values() if p > 0.01 * pmax),
            "shear_over_mu_pressure_min": float(min(ratio)) if ratio else None,
            "shear_over_mu_pressure_median": float(np.median(ratio)) if ratio else None}


def solve_one(kind, idx, mu, traj, ccx):
    t, frame, meta = meta_of(traj, idx)
    valid = {fl.name for fl in fields(fem.GearPairCase)}
    base = {k: meta[k] for k in meta if k in valid}
    if mu is not None:
        base["friction"] = mu
    if kind == "refined":
        base["mesh_size_mm"] = H_REFINED
        fem.gear_tooth_polygon = functools.partial(ORIG_POLYGON, flank_samples=FLANK_REFINED)
    else:
        base["mesh_size_mm"] = H_PROD
        fem.gear_tooth_polygon = ORIG_POLYGON
    case = fem.GearPairCase(**base)
    with tempfile.TemporaryDirectory(prefix="clp_") as work:
        d = Path(work) / "case"
        deck = fem.build_gear_pair_case(case, d)
        r = fem.run_calculix(deck, ccx, timeout_seconds=3600)
        if r.returncode != 0:
            raise RuntimeError(f"solve failed: {kind} {idx} {mu}")
        meta_new = json.loads((d / "case.json").read_text(encoding="utf-8"))
        row = {"kind": kind, "index": idx, "trajectory": t, "frame": frame, "friction": case.friction,
               "indentation_mm": case.indentation_mm, "mesh_mm": case.mesh_size_mm,
               "flank_vertices": FLANK_REFINED if kind == "refined" else FLANK_PROD,
               "nodes": meta_new.get("nodes"), **contact_quantities(deck, meta_new, case)}
        if kind == "refined":
            g = fem.project_case_to_grid(d, 64)
            REFINED.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(REFINED / f"{name_of(kind, idx, mu)}.npz", stress=g["stress"], mask=g["mask"])
    fem.gear_tooth_polygon = ORIG_POLYGON
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ccx", type=Path, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--assemble", action="store_true")
    a = p.parse_args()
    traj, test, jobs = job_list()
    JOBS.mkdir(parents=True, exist_ok=True)
    if a.assemble:
        rows = [json.loads((JOBS / f"{name_of(*j)}.json").read_text(encoding="utf-8")) for j in jobs]
        OUT.write_text(json.dumps({"test_indices": test, "rows": rows}, indent=2), encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} solves")
        return
    ccx = find_ccx(a.ccx)
    mine = [j for n, j in enumerate(jobs) if n % a.workers == a.worker_id]
    for n, j in enumerate(mine):
        out = JOBS / f"{name_of(*j)}.json"
        if out.exists():
            continue
        row = solve_one(*j, traj, ccx)
        out.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"[worker {a.worker_id}] {n + 1}/{len(mine)} {out.stem}: F'={row['normal_force_N_per_mm']:.1f} N/mm "
              f"Ft/Fn={row['tangential_over_normal']:+.3f} r={row['contact_radius_mm']:.2f}", flush=True)


if __name__ == "__main__":
    main()

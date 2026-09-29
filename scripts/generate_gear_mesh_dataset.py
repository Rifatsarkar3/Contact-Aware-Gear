"""Regenerate the gear-tooth datasets with kinematically consistent loading.

The v2 generator displaced the mating rim along the line joining the closest
vertices of the two faceted flank polygons, which deviates from the flank normal
by up to ~90 deg; the load and the friction direction then depended on the flank
discretization (see scripts/analyze_contact_load_path.py). This script builds
the same 16 trajectories x 9 mesh positions with ``fem.build_gear_mesh_case``:
conjugate placement, contact in the single-tooth-contact zone of the line of
action (crossing the pitch point), loading by a normal approach along the line of
action plus a tangential slip with the sign of the kinematic sliding. The contact
pair has no initial ADJUST, which in an earlier run of this script moved slave
nodes within 0.05 element sizes onto the master and widened the contact.

Per trajectory, friction coefficient and Young's modulus are taken unchanged from
v2; the approach amplitude is v2's indentation amplitude mapped linearly from
[0.018, 0.030] mm to [0.004, 0.012] mm (normal loads of roughly 100 to 500 N/mm);
the mesh position runs through the single-contact zone in v2's frame order.
Every case is solved at its own friction and at mu = 0.04 and 0.12.

Also solved: the 27 test cases with 0.08 mm elements (80-vertex flank) and with
0.08 mm elements and a 120-vertex flank, and the latter at mu = 0.04 and 0.12 for
three positions of the first test trajectory.

Usage:
    python scripts/generate_gear_mesh_dataset.py --workers 6 --worker-id k   (k = 0..5)
    python scripts/generate_gear_mesh_dataset.py --assemble
Outputs: outputs/gear_mesh_v4_dataset/*.npz|json,
         data/processed/gear_mesh_quasistatic_v4.h5 (144 cases) and _v4f.h5 (432 cases)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
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

V2 = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v2.h5"
SOLVED = ROOT / "outputs" / "gear_mesh_v4_dataset"
OUT4 = ROOT / "data" / "processed" / "gear_mesh_quasistatic_v4.h5"
OUT4F = ROOT / "data" / "processed" / "gear_mesh_quasistatic_v4f.h5"
MUS = (0.04, 0.12)
FRAMES = 9
APPROACH_SCALE, S_SCALE = 0.010, fem.GearMeshCase().single_contact_half_length_mm


def v2_parameters():
    with h5py.File(V2, "r") as f:
        c = np.asarray(f["conditions"])
        traj = np.asarray(f["trajectory_ids"])
    rows = []
    for i in range(len(traj)):
        t = int(traj[i])
        first = int(np.flatnonzero(traj == t).min())
        frame = i - first
        base_v2 = float(c[first, 0]) / 0.85                   # frame 0: indentation = 0.85 * base
        base = 0.004 + (base_v2 - 0.018) / 0.012 * 0.008
        approach = base * (0.85 + 0.30 * math.sin(math.pi * frame / (FRAMES - 1)))
        directed = float(c[i, 4])                              # 0..1 in v2's frame order
        s = -S_SCALE + 2.0 * S_SCALE * directed
        rows.append({"index": i, "trajectory": t, "frame": frame, "friction": float(c[i, 1]),
                     "youngs_modulus_mpa": float(c[i, 2]), "approach_mm": approach, "line_of_action_mm": s,
                     "directed_phase": directed})
    return rows, traj


def job_list():
    rows, traj = v2_parameters()
    split = grouped_trajectory_split(traj, train_fraction=0.625, val_fraction=0.1875, seed=42)
    test = sorted(int(i) for i in split.test)
    jobs = [("base", i, None) for i in range(len(rows))] + [("base", i, mu) for mu in MUS for i in range(len(rows))]
    jobs += [("fine", i, None) for i in test] + [("fineflank", i, None) for i in test]
    jobs += [("fineflank", test[fr], mu) for fr in (1, 4, 7) for mu in MUS]
    return rows, traj, test, jobs


def name_of(kind, idx, mu):
    return f"{kind}_{idx:03d}_" + ("own" if mu is None else f"mu{mu:.2f}")


def contact_metrics(deck: Path, meta: dict, case: fem.GearMeshCase) -> dict:
    nodes, sets = read_inp(deck)
    frd = deck.with_suffix(".frd")
    contact = fem._parse_nodal_blocks(frd, " -4  CONTACT", 6)[-1]
    force = fem.parse_final_nodal_force(frd)
    f = -sum(force[n] for n in sets["DRIVERROOT"] if n in force)[:2]
    n, t = np.asarray(meta["contact_normal"]), np.asarray(meta["contact_tangent"])
    press = {k: v[3] for k, v in contact.items() if v[3] > 0}
    pmax = max(press.values())
    loaded = [k for k, p in press.items() if p > 0.01 * pmax]
    cen = sum(press[k] * nodes[k] for k in loaded) / sum(press[k] for k in loaded)
    fn, ft = float(-(f @ n)), float(f @ t)
    rb = case.base_radius_mm
    rho1 = involute_radius_of_curvature(cen, np.zeros(2), rb)
    rho2 = involute_radius_of_curvature(cen, np.asarray(meta["mating_center_mm"]), rb)
    r_eff = rho1 * rho2 / (rho1 + rho2)
    e_star = case.youngs_modulus_mpa / (2.0 * (1.0 - case.poisson_ratio ** 2))
    b = math.sqrt(4.0 * fn * r_eff / (math.pi * e_star))
    ratio = [math.hypot(contact[k][4], contact[k][5]) / (case.friction * press[k]) for k in loaded if press[k] > 0.2 * pmax]
    return {"normal_force_N_per_mm": fn, "tangential_force_toward_tip_N_per_mm": ft, "tangential_over_normal": ft / fn,
            "contact_centroid_radius_mm": float(np.linalg.norm(cen)), "nominal_contact_radius_mm": meta["contact_radius_mm"],
            "r_eff_mm": r_eff, "hertz_half_width_mm": b, "hertz_peak_pressure_mpa": 2.0 * fn / (math.pi * b),
            "fe_peak_pressure_mpa": float(pmax), "active_nodes": len(loaded),
            "shear_over_mu_pressure_min": float(min(ratio)), "shear_over_mu_pressure_median": float(np.median(ratio)),
            "imposed_slip_toward_tip_mm": meta["imposed_slip_toward_tip_mm"]}


def solve(job, rows, ccx):
    kind, idx, mu = job
    r = rows[idx]
    case = fem.GearMeshCase(line_of_action_mm=r["line_of_action_mm"], approach_mm=r["approach_mm"],
                            friction=r["friction"] if mu is None else mu, youngs_modulus_mpa=r["youngs_modulus_mpa"],
                            mesh_size_mm=0.12 if kind == "base" else 0.08,
                            flank_samples=120 if kind == "fineflank" else 80)
    with tempfile.TemporaryDirectory(prefix="v4_") as work:
        d = Path(work) / "case"
        deck = fem.build_gear_mesh_case(case, d)
        res = fem.run_calculix(deck, ccx, timeout_seconds=3600)
        if res.returncode != 0:
            raise RuntimeError(f"solve failed: {job}")
        meta = json.loads((d / "case.json").read_text(encoding="utf-8"))
        g = fem.project_case_to_grid(d, 64)
        log = (d / "solver.log").read_text(encoding="utf-8", errors="ignore") if (d / "solver.log").exists() else ""
        secs = float(log.rsplit("Total CalculiX Time:", 1)[1].split()[0]) if "Total CalculiX Time:" in log else -1.0
        row = {"job": list(job), **r, "friction": case.friction, "mesh_mm": case.mesh_size_mm,
               "flank_vertices": case.flank_samples, "solve_seconds": secs, "contact_point_mm": meta["contact_point_mm"],
               **contact_metrics(deck, meta, case)}
    return row, g


def assemble(rows, traj, jobs):
    v2 = h5py.File(V2, "r")
    attrs = dict(v2.attrs)
    v2.close()
    variant_of = {None: 0, MUS[0]: 1, MUS[1]: 2}
    xs, ys, cs, ts, vs, meta_rows = [], [], [], [], [], []
    for mu in (None, *MUS):
        for i in range(len(rows)):
            name = name_of("base", i, mu)
            g = np.load(SOLVED / f"{name}.npz")
            m = json.loads((SOLVED / f"{name}.json").read_text(encoding="utf-8"))
            mask = g["mask"]
            x_axis, y_axis = g["x"][0], g["y"][:, 0]
            xx, yy = np.meshgrid(x_axis, y_axis)
            x_norm = (xx - xx.mean()) / max(float(np.ptp(xx)), 1e-6)
            y_norm = (yy - yy.mean()) / max(float(np.ptp(yy)), 1e-6)
            cx, cy = m["contact_point_mm"]
            cmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 0.30 ** 2)).astype(np.float32) * mask
            scal = np.asarray([m["approach_mm"] / APPROACH_SCALE, m["friction"] / 0.10, m["youngs_modulus_mpa"] / 210_000.0,
                               m["line_of_action_mm"] / S_SCALE], dtype=np.float32)
            feat = np.concatenate((mask[None], x_norm[None], y_norm[None],
                                   np.broadcast_to(scal[:, None, None], (4, 64, 64)), cmap[None])).astype(np.float32)
            xs.append(feat)
            ys.append(g["stress"].astype(np.float32))
            cs.append([m["approach_mm"], m["friction"], m["youngs_modulus_mpa"], m["line_of_action_mm"],
                       m["directed_phase"], m["normal_force_N_per_mm"]])
            ts.append(int(traj[i]))
            vs.append(variant_of[mu])
    xs, ys, cs, ts, vs = map(np.asarray, (xs, ys, cs, ts, vs))
    common = {"condition_columns": "approach_mm, friction, youngs_modulus_mpa, line_of_action_mm, directed_phase, "
                                   "normal_force_N_per_mm (output, not an input)",
              "grid_dx_mm": attrs["grid_dx_mm"], "grid_dy_mm": attrs["grid_dy_mm"], "mesh_size_mm": 0.12, "seed": 7,
              "fem_model": "two deformable involute tooth sectors, plane strain, frictional contact; conjugate placement; "
                           "load = approach along the line of action + tangential slip with the kinematic sign",
              "input_channels": "mask, x, y, approach/0.010 mm, mu/0.10, E/210 GPa, s/single-contact half length, contact map"}
    for path, sel in ((OUT4, vs == 0), (OUT4F, np.ones(len(vs), bool))):
        with h5py.File(path, "w") as h:
            h.create_dataset("inputs", data=xs[sel], compression="gzip", shuffle=True)
            h.create_dataset("targets", data=ys[sel], compression="gzip", shuffle=True)
            h.create_dataset("conditions", data=cs[sel].astype(np.float64))
            h.create_dataset("trajectory_ids", data=ts[sel].astype(np.int32))
            h.create_dataset("friction_variant", data=vs[sel].astype(np.int8))
            for k, v in common.items():
                h.attrs[k] = v
            h.attrs["evidence_status"] = "quasi-static frictional contact with kinematically consistent loading (2026-09-27)"
        print(f"wrote {path.relative_to(ROOT)}: {int(sel.sum())} cases")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ccx", type=Path, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--assemble", action="store_true")
    a = p.parse_args()
    rows, traj, test, jobs = job_list()
    SOLVED.mkdir(parents=True, exist_ok=True)
    if a.assemble:
        assemble(rows, traj, jobs)
        metrics = [json.loads((SOLVED / f"{name_of(*j)}.json").read_text(encoding="utf-8")) for j in jobs]
        (ROOT / "outputs" / "surrogate" / "gear_mesh_v4_contact.json").write_text(
            json.dumps({"test_indices": test, "rows": metrics}, indent=2), encoding="utf-8")
        return
    ccx = find_ccx(a.ccx)
    mine = [j for n, j in enumerate(jobs) if n % a.workers == a.worker_id]
    for n, j in enumerate(mine):
        name = name_of(*j)
        if (SOLVED / f"{name}.json").exists():
            continue
        row, g = solve(j, rows, ccx)
        np.savez_compressed(SOLVED / f"{name}.npz", stress=g["stress"], mask=g["mask"], x=g["x"], y=g["y"])
        (SOLVED / f"{name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"[worker {a.worker_id}] {n + 1}/{len(mine)} {name}: F'={row['normal_force_N_per_mm']:.1f} N/mm "
              f"Ft/Fn={row['tangential_over_normal']:+.3f} s={row['line_of_action_mm']:+.2f} ({row['solve_seconds']:.1f} s)",
              flush=True)


if __name__ == "__main__":
    main()

"""Finite-element checks of the gear-mesh model: contact status, convergence, grid
sensitivity of the root peak, and additional test trajectories.

Job families (CalculiX runs single-threaded; jobs run in parallel):

* ``L1``: every v4 case at its own friction coefficient and at mu = 0.04 and 0.12
  (432 solves, production discretization: 0.12 mm elements, 80-vertex flank).
  Records the contact status of every slave node (relative tangential slip
  against the elastic stick limit mu p / k_t, frictional shear against mu p) and
  checks that the solve reproduces the stored grid solution.
* ``L2``: the 27 test cases with 0.08 mm elements and a 120-vertex flank.
* ``L3``: the 27 test cases with 0.06 mm elements and a 160-vertex flank.
* ``X``: six additional trajectories with positions off the training grid and a
  random approach at every position (54 solves, production discretization).

For L1 (own friction), L2 and L3 the nodal stresses are also projected onto
128 x 128 and 256 x 256 grids over the same window to measure how the peak root
tensile stress depends on the grid. For each solve the nodal contact pressures
along the flank are kept, with the contact width measured from them.

Usage:
    python scripts/run_fe_checks.py --workers 8            # run (resumable)
    python scripts/run_fe_checks.py --assemble-extra       # HDF5 of the X cases
Outputs: outputs/extended_study/fe/*.json|npz, data/processed/gear_mesh_quasistatic_v4_extra.h5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gearstress.fem as fem  # noqa: E402
from generate_gear_mesh_dataset import (APPROACH_SCALE, S_SCALE, SOLVED, contact_metrics, job_list,  # noqa: E402
                                        read_inp)
from measure_label_uncertainty import find_ccx  # noqa: E402

OUT = ROOT / "outputs" / "extended_study" / "fe"
EXTRA_H5 = ROOT / "data" / "processed" / "gear_mesh_quasistatic_v4_extra.h5"
LEVELS = {"L1": (0.12, 80), "L2": (0.08, 120), "L3": (0.06, 160), "X": (0.12, 80)}
MUS = (0.04, 0.12)
ROOT_BAND_TOP_MM = 21.5 + 2.0           # root circle to one module above it (as in the 64 x 64 analysis)
THRESHOLDS = (0.01, 0.05, 0.1, 0.2, 0.5)
EXTRA_SEED, N_EXTRA_TRAJ, N_POS = 20261011, 6, 9


def extra_rows(rows):
    """Six trajectories with mu, E and the approach amplitude drawn inside the v4 ranges, nine contact
    positions drawn uniformly over the single-tooth contact zone, and an approach drawn independently
    at every position (instead of the sine law of v4)."""
    e_lo = min(r["youngs_modulus_mpa"] for r in rows)
    e_hi = max(r["youngs_modulus_mpa"] for r in rows)
    rng = np.random.default_rng(EXTRA_SEED)
    out = []
    for t in range(N_EXTRA_TRAJ):
        mu = float(rng.uniform(0.04, 0.12))
        e = float(rng.uniform(e_lo, e_hi))
        d0 = float(rng.uniform(0.004, 0.012))
        s = np.sort(rng.uniform(-S_SCALE, S_SCALE, N_POS))
        f = rng.uniform(0.85, 1.15, N_POS)
        for k in range(N_POS):
            out.append({"index": len(out), "trajectory": 100 + t, "frame": k, "friction": mu, "youngs_modulus_mpa": e,
                        "approach_mm": d0 * float(f[k]), "line_of_action_mm": float(s[k]),
                        "directed_phase": float((s[k] + S_SCALE) / (2 * S_SCALE))})
    return out


def all_jobs():
    rows, traj, test, _ = job_list()
    jobs = [("L1", i, mu) for mu in (None, *MUS) for i in range(len(rows))]
    jobs += [(lev, i, None) for lev in ("L2", "L3") for i in test]
    jobs += [("X", k, None) for k in range(len(extra_rows(rows)))]
    return jobs


def name_of(job):
    lev, i, mu = job
    return f"{lev}_{i:03d}_" + ("own" if mu is None else f"mu{mu:.2f}")


def root_peak(stress, mask, yy):
    s1 = 0.5 * (stress[0] + stress[1]) + np.sqrt(0.25 * (stress[0] - stress[1]) ** 2 + stress[2] ** 2)
    band = mask & (yy <= ROOT_BAND_TOP_MM)
    k = np.unravel_index(np.where(band, s1, -np.inf).argmax(), s1.shape)
    return float(s1[k]), [float(v) for v in k]


def slave_nodes(deck: Path) -> set[int]:
    """Nodes on the faces of the slave surface (MATINGSURF) of the contact pair."""
    elems, faces, mode = {}, [], None
    for line in deck.read_text(encoding="ascii", errors="replace").splitlines():
        u = line.upper()
        if line.startswith("*"):
            mode = "elem" if u.startswith("*ELEMENT") else "surf" if (u.startswith("*SURFACE") and "MATINGSURF" in u) else None
            continue
        if mode == "elem":
            v = [int(s) for s in line.split(",") if s.strip()]
            elems[v[0]] = v[1:]
        elif mode == "surf":
            e, f = (s.strip() for s in line.split(","))
            faces.append((int(e), int(f.upper().lstrip("S"))))
    out = set()
    for e, f in faces:
        n = elems[e]
        out |= {n[f - 1], n[f % 3]}
    return out


def contact_status(deck: Path, meta: dict, case) -> dict:
    """Contact pressure, frictional shear and relative tangential slip at every slave node."""
    nodes, _ = read_inp(deck)
    frd = deck.with_suffix(".frd")
    block = fem._parse_nodal_blocks(frd, " -4  CONTACT", 6)[-1]      # COPEN, CSLIP1, CSLIP2, CPRESS, CSHEAR1, CSHEAR2
    cp, t, nrm = (np.asarray(meta[k]) for k in ("contact_point_mm", "contact_tangent", "contact_normal"))
    k_t = 0.5 * case.youngs_modulus_mpa                                # stick slope, MPa per mm
    rows = []
    on_surface = slave_nodes(deck)
    for nid, v in block.items():
        if nid not in on_surface:
            continue
        xi, dn = float((nodes[nid] - cp) @ t), float((nodes[nid] - cp) @ nrm)
        if abs(xi) > 1.0 or abs(dn) > 0.15:        # keep the loaded flank near the contact point
            continue
        rows.append((xi, float(v[3]), float(math.hypot(v[4], v[5])), float(math.hypot(v[1], v[2]))))
    rows.sort()
    xi, p, tau, slip = (np.array(c) for c in zip(*rows))
    pmax = float(p.max())
    loaded = p > 0
    out = {"n_slave_nodes": int(len(xi)), "n_loaded": int(loaded.sum()), "p_max_mpa": pmax}
    by = {}
    for f in THRESHOLDS:
        sel = p > f * pmax
        limit = case.friction * p[sel]
        by[str(f)] = {"n_nodes": int(sel.sum()),
                      "shear_over_mu_p_min": float((tau[sel] / limit).min()),
                      "slip_over_stick_limit_min": float((k_t * slip[sel] / limit).min()),
                      "n_sliding": int((k_t * slip[sel] > limit * (1 + 1e-6)).sum())}
    out["by_threshold"] = by
    # contact width from the nodal pressures: extent of the loaded nodes, the same extended linearly to
    # the zero-pressure crossing toward the first unloaded neighbor, and a least-squares Hertz fit
    idx = np.flatnonzero(loaded)
    lo, hi = idx.min(), idx.max()
    out["width_loaded_nodes_mm"] = float(xi[hi] - xi[lo])
    left = xi[lo] if lo == 0 else xi[lo - 1]
    right = xi[hi] if hi == len(xi) - 1 else xi[hi + 1]
    out["width_to_unloaded_neighbors_mm"] = float(right - left)
    best = None
    for b in np.linspace(0.3, 3.0, 271) * max(out["width_loaded_nodes_mm"] / 2, 1e-3):
        for c in np.linspace(xi[lo], xi[hi], 41):
            shape = np.sqrt(np.clip(1 - ((xi - c) / b) ** 2, 0, None))
            if shape.sum() == 0:
                continue
            p0 = float((shape @ p) / (shape @ shape))
            err = float(((p - p0 * shape) ** 2).sum())
            if best is None or err < best[0]:
                best = (err, b, c, p0)
    out["hertz_fit"] = {"half_width_mm": float(best[1]), "center_mm": float(best[2]), "p0_mpa": float(best[3]),
                        "rms_residual_mpa": float(math.sqrt(best[0] / len(xi)))}
    out["profile"] = {"xi_mm": [float(v) for v in xi[max(lo - 3, 0): hi + 4]],
                      "p_mpa": [float(v) for v in p[max(lo - 3, 0): hi + 4]],
                      "shear_mpa": [float(v) for v in tau[max(lo - 3, 0): hi + 4]],
                      "slip_mm": [float(v) for v in slip[max(lo - 3, 0): hi + 4]]}
    return out


def delaunay_vs_elements(case_dir: Path, grid: dict) -> dict:
    """Interpolation over the Delaunay triangulation of the nodes (used for the datasets) against
    interpolation over the finite elements themselves, at the material grid points."""
    from matplotlib.tri import LinearTriInterpolator, Triangulation

    with np.load(case_dir / "mesh_data.npz") as mesh:
        ids, xy, conn, tooth = mesh["node_ids"], mesh["coordinates"][:, :2], mesh["driver_connectivity"], mesh["tooth_node_ids"]
    ix = {int(t): k for k, t in enumerate(ids)}
    stress_map = fem.parse_final_nodal_stress(case_dir / "case.frd")
    tri = Triangulation(xy[:, 0], xy[:, 1], np.array([[ix[int(t)] for t in e] for e in conn]))
    m = grid["mask"] > 0.5
    diffs, norms = [], []
    for comp, col in enumerate((0, 1, 3)):
        z = np.zeros(len(ids))
        for t in tooth:
            z[ix[int(t)]] = stress_map[int(t)][col]
        f = LinearTriInterpolator(tri, z)(grid["x"], grid["y"]).filled(0.0)
        diffs.append(grid["stress"][comp][m] - f[m])
        norms.append(f[m])
    d, n = np.concatenate(diffs), np.concatenate(norms)
    return {"max_abs_mpa": float(np.abs(d).max()), "relative_l2": float(np.linalg.norm(d) / np.linalg.norm(n))}


def run(job):
    lev, i, mu = job
    name = name_of(job)
    if (OUT / f"{name}.json").exists():
        return name, "exists"
    rows, traj, test, _ = job_list()
    r = extra_rows(rows)[i] if lev == "X" else rows[i]
    h, flank = LEVELS[lev]
    case = fem.GearMeshCase(line_of_action_mm=r["line_of_action_mm"], approach_mm=r["approach_mm"],
                            friction=r["friction"] if mu is None else mu, youngs_modulus_mpa=r["youngs_modulus_mpa"],
                            mesh_size_mm=h, flank_samples=flank)
    with tempfile.TemporaryDirectory(prefix=f"r1_{lev}_") as work:
        d = Path(work) / "case"
        deck = fem.build_gear_mesh_case(case, d)
        res = fem.run_calculix(deck, find_ccx(None), timeout_seconds=7200)
        if res.returncode != 0:
            return name, "FAILED"
        meta = json.loads((d / "case.json").read_text(encoding="utf-8"))
        log = (d / "solver.log").read_text(encoding="utf-8", errors="ignore") if (d / "solver.log").exists() else ""
        secs = float(log.rsplit("Total CalculiX Time:", 1)[1].split()[0]) if "Total CalculiX Time:" in log else -1.0
        out = {"job": [lev, i, mu], **r, "friction": case.friction, "mesh_mm": h, "flank_vertices": flank,
               "solve_seconds": secs, "contact_point_mm": meta["contact_point_mm"],
               **contact_metrics(deck, meta, case), "contact_status": contact_status(deck, meta, case)}
        g64 = fem.project_case_to_grid(d, 64, material_mask="elements")
        out["root_peak_mpa"] = {"64": root_peak(g64["stress"], g64["mask"] > 0.5, g64["y"])[0]}
        if mu is None and lev in ("L1", "L2", "L3"):
            for n in (128, 256):
                g = fem.project_case_to_grid(d, n, material_mask="elements")
                out["root_peak_mpa"][str(n)], out[f"root_peak_pixel_{n}"] = root_peak(g["stress"], g["mask"] > 0.5, g["y"])
                if n == 128:
                    np.savez_compressed(OUT / f"{name}_g128.npz", stress=g["stress"].astype(np.float32), mask=g["mask"])
        if lev == "L1":
            stored = np.load(SOLVED / (f"base_{i:03d}_" + ("own" if mu is None else f"mu{mu:.2f}") + ".npz"))["stress"]
            m = g64["mask"] > 0.5
            out["reproduces_stored_solution_max_abs_mpa"] = float(np.abs(g64["stress"][:, m] - stored[:, m]).max())
            if mu is None and i in test:
                out["delaunay_vs_elements"] = delaunay_vs_elements(d, g64)
        if lev in ("L2", "L3", "X") or (lev == "L1" and mu is None and i in test):
            np.savez_compressed(OUT / f"{name}.npz", stress=g64["stress"].astype(np.float32), mask=g64["mask"],
                                x=g64["x"], y=g64["y"])
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return name, f"{secs:.0f} s"


def assemble_extra():
    rows, _, _, _ = job_list()
    ex = extra_rows(rows)
    material = fem.gear_mesh_material_mask(64).astype(np.float32)
    xs, ys, cs, ts = [], [], [], []
    for k, r in enumerate(ex):
        g = np.load(OUT / f"X_{k:03d}_own.npz")
        m = json.loads((OUT / f"X_{k:03d}_own.json").read_text(encoding="utf-8"))
        assert np.array_equal(g["mask"] > 0.5, material > 0.5)
        xx, yy = g["x"], g["y"]
        x_norm = (xx - xx.mean()) / float(np.ptp(xx))
        y_norm = (yy - yy.mean()) / float(np.ptp(yy))
        cx, cy = m["contact_point_mm"]
        cmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 0.30 ** 2)).astype(np.float32) * material
        scal = np.asarray([r["approach_mm"] / APPROACH_SCALE, r["friction"] / 0.10, r["youngs_modulus_mpa"] / 210_000.0,
                           r["line_of_action_mm"] / S_SCALE], dtype=np.float32)
        xs.append(np.concatenate((material[None], x_norm[None], y_norm[None],
                                  np.broadcast_to(scal[:, None, None], (4, 64, 64)), cmap[None])).astype(np.float32))
        ys.append(g["stress"].astype(np.float32))
        cs.append([r["approach_mm"], r["friction"], r["youngs_modulus_mpa"], r["line_of_action_mm"],
                   r["directed_phase"], m["normal_force_N_per_mm"]])
        ts.append(r["trajectory"])
    with h5py.File(EXTRA_H5, "w") as h:
        h.create_dataset("inputs", data=np.asarray(xs), compression="gzip", shuffle=True)
        h.create_dataset("targets", data=np.asarray(ys), compression="gzip", shuffle=True)
        h.create_dataset("conditions", data=np.asarray(cs, dtype=np.float64))
        h.create_dataset("trajectory_ids", data=np.asarray(ts, dtype=np.int32))
        h.attrs["condition_columns"] = ("approach_mm, friction, youngs_modulus_mpa, line_of_action_mm, directed_phase, "
                                        "normal_force_N_per_mm (output, not an input)")
        h.attrs["description"] = ("six additional test trajectories: contact positions drawn uniformly over the "
                                  "single-tooth contact zone, approach drawn independently at every position")
    print(f"wrote {EXTRA_H5.relative_to(ROOT)}: {len(xs)} cases")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="comma-separated job families to run, e.g. L2,L3")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many pending jobs (testing)")
    ap.add_argument("--assemble-extra", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.assemble_extra:
        assemble_extra()
        return
    jobs = all_jobs()
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    # slow levels first so the batch finishes evenly
    order = {"L3": 0, "L2": 1, "X": 2, "L1": 3}
    jobs = sorted((j for j in jobs if not (OUT / f"{name_of(j)}.json").exists()), key=lambda j: order[j[0]])
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} pending jobs, {a.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            name, status = fut.result()
            print(f"[{n}/{len(jobs)} {(time.time() - t0) / 60:.1f} min] {name}: {status}", flush=True)


if __name__ == "__main__":
    main()

"""Gear-tooth datasets with a filleted root and a deformable rim (v5, v5f).

The datasets v4 and v4f fix every driver node on and below the root circle, where
the radial flank meets the root circle at a sharp corner, so their root stress is
singular and does not converge under refinement. This script solves the same 432
cases (the 16 trajectories x 9 positions of v4, each at its own friction
coefficient and at mu = 0.04 and 0.12, with unchanged approach, modulus and
contact position) with ``fem.build_fillet_gear_mesh_case``: a circular root
fillet of 0.38 module, a three-tooth sector whose rim deforms and is fixed only at
the bore, and a mesh refined along the fillets and around the contact point. The
27 test cases of the reference partition are also solved with every length scale
refined 1.5-fold (L2) and 2-fold (L3).

Usage:
    python scripts/generate_fillet_rim_dataset.py --workers 8      (solve every job; reruns skip solved ones)
    python scripts/generate_fillet_rim_dataset.py --assemble
Outputs: outputs/fillet_rim_dataset/*.npz|json,
         data/processed/gear_mesh_quasistatic_v5.h5 (144 cases) and _v5f.h5 (432 cases)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gearstress.fem as fem  # noqa: E402
import generate_gear_mesh_dataset as v4gen  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402
from measure_label_uncertainty import find_ccx  # noqa: E402

SOLVED = ROOT / "outputs" / "fillet_rim_dataset"
OUT5 = ROOT / "data" / "processed" / "gear_mesh_quasistatic_v5.h5"
OUT5F = ROOT / "data" / "processed" / "gear_mesh_quasistatic_v5f.h5"
MUS = v4gen.MUS
LEVELS = {"L1": 1.0, "L2": 1.5, "L3": 2.0}


def case_of(row: dict, mu: float | None, level: str) -> fem.FilletGearMeshCase:
    k, base = LEVELS[level], fem.FilletGearMeshCase()
    return fem.FilletGearMeshCase(
        line_of_action_mm=row["line_of_action_mm"], approach_mm=row["approach_mm"],
        friction=row["friction"] if mu is None else mu, youngs_modulus_mpa=row["youngs_modulus_mpa"],
        mesh_size_mm=base.mesh_size_mm / k, outer_mesh_size_mm=base.outer_mesh_size_mm / k,
        contact_mesh_size_mm=base.contact_mesh_size_mm / k, fillet_mesh_size_mm=base.fillet_mesh_size_mm / k,
        flank_samples=int(round(base.flank_samples * k)), fillet_samples=int(round(base.fillet_samples * k)))


def job_list():
    rows, traj = v4gen.v2_parameters()
    split = grouped_trajectory_split(traj, train_fraction=0.625, val_fraction=0.1875, seed=42)
    test = sorted(int(i) for i in split.test)
    jobs = [("L1", i, None) for i in range(len(rows))] + [("L1", i, mu) for mu in MUS for i in range(len(rows))]
    jobs += [(level, i, None) for level in ("L2", "L3") for i in test]
    return rows, traj, test, jobs


def name_of(level: str, idx: int, mu: float | None) -> str:
    return f"{level}_{idx:03d}_" + ("own" if mu is None else f"mu{mu:.2f}")


def nodal_root_peak(case_dir: Path, case: fem.FilletGearMeshCase) -> tuple[float, list[float]]:
    """Largest in-plane maximum principal nodal stress of the centre tooth between 1 mm below the
    root circle and 1.2 modules above it (the root fillets), with its position."""
    mesh = np.load(case_dir / "mesh_data.npz")
    xy = {int(t): c[:2] for t, c in zip(mesh["node_ids"], mesh["coordinates"])}
    stress = fem.parse_final_nodal_stress(case_dir / "case.frd")
    rr, best, where = case.root_radius_mm, -math.inf, None
    for t in mesh["driver_node_ids"]:
        p = xy[int(t)]
        if rr - 1.0 <= float(np.hypot(*p)) <= rr + 1.2 * case.module_mm and abs(p[0]) < 1.75 * case.module_mm:
            v = stress[int(t)]
            s1 = 0.5 * (v[0] + v[1]) + math.sqrt((0.5 * (v[0] - v[1])) ** 2 + v[3] ** 2)
            if s1 > best:
                best, where = float(s1), [float(p[0]), float(p[1])]
    return best, where


def band_peak(g: dict, case: fem.FilletGearMeshCase) -> tuple[float, list[float]]:
    """Largest in-plane maximum principal stress on a projected grid in the band up to one module above the root
    circle (the band of scripts/run_fe_checks.py), with its position (x, y in mm)."""
    s = g["stress"]
    band = (g["mask"] > 0.5) & (g["y"] <= case.root_radius_mm + case.module_mm)
    s1 = 0.5 * (s[0] + s[1]) + np.sqrt(0.25 * (s[0] - s[1]) ** 2 + s[2] ** 2)
    k = np.unravel_index(np.where(band, s1, -np.inf).argmax(), s1.shape)
    return float(s1[k]), [float(g["x"][k]), float(g["y"][k])]


def is_test_case(job, test) -> bool:
    level, idx, mu = job
    return level != "L1" or (idx in test and mu is None)


def solve(job, rows, ccx, test=()):
    level, idx, mu = job
    r = rows[idx]
    case = case_of(r, mu, level)
    with tempfile.TemporaryDirectory(prefix="v5_") as work:
        d = Path(work) / "case"
        deck = fem.build_fillet_gear_mesh_case(case, d)
        res = fem.run_calculix(deck, ccx, timeout_seconds=7200)
        if res.returncode != 0:
            raise RuntimeError(f"solve failed: {job}")
        meta = json.loads((d / "case.json").read_text(encoding="utf-8"))
        g = fem.project_case_to_grid(d, 64, material_mask="elements")
        log = (d / "solver.log").read_text(encoding="utf-8", errors="ignore") if (d / "solver.log").exists() else ""
        secs = float(log.rsplit("Total CalculiX Time:", 1)[1].split()[0]) if "Total CalculiX Time:" in log else -1.0
        peak, where = nodal_root_peak(d, case)
        row = {"job": list(job), **r, "friction": case.friction, "level": level, "mesh_mm": case.mesh_size_mm,
               "flank_vertices": case.flank_samples, "fillet_vertices": case.fillet_samples,
               "elements": meta["elements"], "solve_seconds": secs, "contact_point_mm": meta["contact_point_mm"],
               "nodal_root_peak_mpa": peak, "nodal_root_peak_at_mm": where, **v4gen.contact_metrics(deck, meta, case)}
        if is_test_case(job, test):
            # root peak after projection onto finer output grids, as for the main model (run_fe_checks.py)
            row["grid_root_peak_mpa"], row["grid_root_peak_at_mm"] = {}, {}
            for n in (64, 128, 256):
                gn = g if n == 64 else fem.project_case_to_grid(d, n, material_mask="elements")
                row["grid_root_peak_mpa"][str(n)], row["grid_root_peak_at_mm"][str(n)] = band_peak(gn, case)
    return row, g


def run_one(job):
    rows, _, test, _ = job_list()
    name = name_of(*job)
    done = SOLVED / f"{name}.json"
    if done.exists() and not (is_test_case(job, test) and
                              "grid_root_peak_mpa" not in json.loads(done.read_text(encoding="utf-8"))):
        return name, None
    t0 = time.time()
    row, g = solve(job, rows, find_ccx(None), test)
    np.savez_compressed(SOLVED / f"{name}.npz", stress=g["stress"], mask=g["mask"], x=g["x"], y=g["y"])
    (SOLVED / f"{name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return name, (f"F'={row['normal_force_N_per_mm']:.1f} N/mm, root {row['nodal_root_peak_mpa']:.1f} MPa, "
                  f"p/pH={row['fe_peak_pressure_mpa'] / row['hertz_peak_pressure_mpa']:.3f} ({time.time() - t0:.0f} s)")


def assemble(rows, traj):
    with h5py.File(v4gen.V2, "r") as v2:
        attrs = dict(v2.attrs)
    variant_of = {None: 0, MUS[0]: 1, MUS[1]: 2}
    material = fem.fillet_gear_mesh_material_mask(64).astype(np.float32)
    xs, ys, cs, ts, vs = [], [], [], [], []
    for mu in (None, *MUS):
        for i in range(len(rows)):
            name = name_of("L1", i, mu)
            g = np.load(SOLVED / f"{name}.npz")
            m = json.loads((SOLVED / f"{name}.json").read_text(encoding="utf-8"))
            if not np.array_equal(g["mask"] > 0.5, material > 0.5):
                raise RuntimeError(f"{name}: material mask differs from the default deck's")
            x_axis, y_axis = g["x"][0], g["y"][:, 0]
            xx, yy = np.meshgrid(x_axis, y_axis)
            x_norm = (xx - xx.mean()) / max(float(np.ptp(xx)), 1e-6)
            y_norm = (yy - yy.mean()) / max(float(np.ptp(yy)), 1e-6)
            cx, cy = m["contact_point_mm"]
            cmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 0.30 ** 2)).astype(np.float32) * material
            scal = np.asarray([m["approach_mm"] / v4gen.APPROACH_SCALE, m["friction"] / 0.10,
                               m["youngs_modulus_mpa"] / 210_000.0, m["line_of_action_mm"] / v4gen.S_SCALE],
                              dtype=np.float32)
            feat = np.concatenate((material[None], x_norm[None], y_norm[None],
                                   np.broadcast_to(scal[:, None, None], (4, 64, 64)), cmap[None])).astype(np.float32)
            xs.append(feat)
            ys.append((g["stress"] * material[None]).astype(np.float32))
            cs.append([m["approach_mm"], m["friction"], m["youngs_modulus_mpa"], m["line_of_action_mm"],
                       m["directed_phase"], m["normal_force_N_per_mm"]])
            ts.append(int(traj[i]))
            vs.append(variant_of[mu])
    xs, ys, cs, ts, vs = map(np.asarray, (xs, ys, cs, ts, vs))
    base = fem.FilletGearMeshCase()
    common = {"condition_columns": "approach_mm, friction, youngs_modulus_mpa, line_of_action_mm, directed_phase, "
                                   "normal_force_N_per_mm (output, not an input)",
              "grid_dx_mm": attrs["grid_dx_mm"], "grid_dy_mm": attrs["grid_dy_mm"], "mesh_size_mm": base.mesh_size_mm,
              "seed": 7,
              "fem_model": "involute tooth pair, plane strain, frictional contact; driver: three teeth with circular root "
                           f"fillets of {base.fillet_radius_module} module on a rim fixed at the bore "
                           f"({base.rim_depth_module} modules below the root circle); conjugate placement; load = approach "
                           "along the line of action + tangential slip with the kinematic sign",
              "input_channels": "mask, x, y, approach/0.010 mm, mu/0.10, E/210 GPa, s/single-contact half length, contact map",
              "material_mask": "grid points inside the finite elements of the driver; targets are zero elsewhere"}
    for path, sel in ((OUT5, vs == 0), (OUT5F, np.ones(len(vs), bool))):
        with h5py.File(path, "w") as h:
            h.create_dataset("inputs", data=xs[sel], compression="gzip", shuffle=True)
            h.create_dataset("targets", data=ys[sel], compression="gzip", shuffle=True)
            h.create_dataset("conditions", data=cs[sel].astype(np.float64))
            h.create_dataset("trajectory_ids", data=ts[sel].astype(np.int32))
            h.create_dataset("friction_variant", data=vs[sel].astype(np.int8))
            for k, v in common.items():
                h.attrs[k] = v
            h.attrs["evidence_status"] = "quasi-static frictional contact, filleted root and deformable rim (2026-10-13)"
        print(f"wrote {path.relative_to(ROOT)}: {int(sel.sum())} cases")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--refined-workers", type=int, default=3, help="parallel solves for the L2 and L3 jobs")
    p.add_argument("--assemble", action="store_true")
    a = p.parse_args()
    rows, traj, test, jobs = job_list()
    SOLVED.mkdir(parents=True, exist_ok=True)
    if a.assemble:
        assemble(rows, traj)                      # needs only the production (L1) solutions
        done = [j for j in jobs if (SOLVED / f"{name_of(*j)}.json").exists()]
        metrics = [json.loads((SOLVED / f"{name_of(*j)}.json").read_text(encoding="utf-8")) for j in done]
        (ROOT / "outputs" / "surrogate" / "gear_mesh_v5_contact.json").write_text(
            json.dumps({"test_indices": test, "solved": len(done), "of": len(jobs), "rows": metrics}, indent=2),
            encoding="utf-8")
        print(f"contact summary: {len(done)} of {len(jobs)} jobs solved")
        return
    for group, n in (([j for j in jobs if j[0] == "L1"], a.workers), ([j for j in jobs if j[0] != "L1"], a.refined_workers)):
        with ProcessPoolExecutor(max_workers=n) as ex:
            for k, (name, msg) in enumerate(ex.map(run_one, group), 1):
                if msg:
                    print(f"{k}/{len(group)} {name}: {msg}", flush=True)


if __name__ == "__main__":
    main()

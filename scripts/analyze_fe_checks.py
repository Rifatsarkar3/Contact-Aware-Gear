"""Summary of the finite-element checks of scripts/run_fe_checks.py.

* contact status of all 432 production cases of v4f (sliding against the elastic
  stick limit, frictional shear against mu p, at several pressure thresholds);
* reproducibility of the stored grid solutions, and Delaunay against element
  interpolation;
* convergence of the test-case fields over three discretizations (L1: 0.12 mm,
  80-vertex flank; L2: 0.08 mm, 120 vertices; L3: 0.06 mm, 160 vertices), with an
  observed order and a Richardson estimate of the labels' distance from the limit;
* dependence of the peak root tensile stress on the output grid (64, 128, 256);
* contact width from the nodal pressures against the Hertz half-width.

Usage: python scripts/analyze_fe_checks.py
Output: outputs/extended_study/fe_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gearstress.fem as fem  # noqa: E402
from generate_gear_mesh_dataset import job_list  # noqa: E402

FE = ROOT / "outputs" / "extended_study" / "fe"
OUT = ROOT / "outputs" / "extended_study" / "fe_summary.json"
H = {"L1": 0.12, "L2": 0.08, "L3": 0.06}
NEAR_MM = 1.0


def summ(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def load(name):
    p = FE / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def contact_all(rows):
    out = {}
    recs = [r for i in range(len(rows)) for mu in (None, 0.04, 0.12)
            if (r := load(f"L1_{i:03d}_" + ("own" if mu is None else f"mu{mu:.2f}"))) is not None]
    out["n_cases"] = len(recs)
    if not recs:
        return out
    off = [r for r in recs if abs(r["line_of_action_mm"]) > 0.05]
    pitch = [r for r in recs if abs(r["line_of_action_mm"]) <= 0.05]
    out["normal_force_N_per_mm"] = summ([r["normal_force_N_per_mm"] for r in recs])
    out["hertz_p0_mpa"] = summ([r["hertz_peak_pressure_mpa"] for r in recs])
    out["fe_over_hertz"] = summ([r["fe_peak_pressure_mpa"] / r["hertz_peak_pressure_mpa"] for r in recs])
    out["abs_Ft_over_Fn_over_mu_off_pitch"] = summ([abs(r["tangential_over_normal"]) / r["friction"] for r in off])
    out["friction_sign_matches_s_off_pitch"] = int(sum(np.sign(r["tangential_force_toward_tip_N_per_mm"]) ==
                                                       np.sign(r["line_of_action_mm"]) for r in off))
    out["n_off_pitch"], out["n_pitch"] = len(off), len(pitch)
    out["pitch_friction_toward_root"] = int(sum(r["tangential_force_toward_tip_N_per_mm"] < 0 for r in pitch))
    def shear_ratio_max(r, f):
        """Largest frictional shear over mu times pressure among the nodes above the pressure threshold."""
        pr, cs = r["contact_status"]["profile"], r["contact_status"]
        p, s = np.asarray(pr["p_mpa"]), np.abs(np.asarray(pr["shear_mpa"]))
        sel = p > float(f) * cs["p_max_mpa"]
        return float((s[sel] / (r["friction"] * p[sel])).max())

    by = {}
    for f in ("0.01", "0.05", "0.1", "0.2", "0.5"):
        b = [r["contact_status"]["by_threshold"][f] for r in recs]
        bo = [r["contact_status"]["by_threshold"][f] for r in off]
        bp = [r["contact_status"]["by_threshold"][f] for r in pitch]
        by[f] = {"cases_all_nodes_sliding": int(sum(x["n_sliding"] == x["n_nodes"] for x in b)),
                 "cases_all_nodes_sliding_off_pitch": int(sum(x["n_sliding"] == x["n_nodes"] for x in bo)),
                 "cases_all_nodes_sliding_pitch": int(sum(x["n_sliding"] == x["n_nodes"] for x in bp)),
                 "nodes": int(sum(x["n_nodes"] for x in b)), "nodes_sliding": int(sum(x["n_sliding"] for x in b)),
                 "slip_over_stick_limit_min": float(min(x["slip_over_stick_limit_min"] for x in b)),
                 "shear_over_mu_p_min": float(min(x["shear_over_mu_p_min"] for x in b)),
                 "off_pitch_slip_over_stick_limit_min": float(min(x["slip_over_stick_limit_min"] for x in bo)),
                 "off_pitch_shear_over_mu_p_min": float(min(x["shear_over_mu_p_min"] for x in bo)),
                 "pitch_slip_over_stick_limit_min": float(min(x["slip_over_stick_limit_min"] for x in bp)) if bp else None,
                 "pitch_slip_over_stick_limit_max_of_case_minima": float(max(x["slip_over_stick_limit_min"] for x in bp)) if bp else None,
                 "pitch_shear_over_mu_p_min": float(min(x["shear_over_mu_p_min"] for x in bp)) if bp else None,
                 "pitch_shear_over_mu_p_max": max(shear_ratio_max(r, f) for r in pitch) if pitch else None,
                 "off_pitch_shear_over_mu_p_max": max(shear_ratio_max(r, f) for r in off)}
    out["by_threshold"] = by
    out["loaded_nodes"] = summ([r["contact_status"]["n_loaded"] for r in recs])
    out["slip_mm"] = summ([max(r["contact_status"]["profile"]["slip_mm"]) for r in recs])
    out["reproduces_stored_max_abs_mpa"] = float(max(r["reproduces_stored_solution_max_abs_mpa"] for r in recs))
    dvse = [r["delaunay_vs_elements"] for r in recs if "delaunay_vs_elements" in r]
    if dvse:
        out["delaunay_vs_elements"] = {"relative_l2": summ([d["relative_l2"] for d in dvse]),
                                       "max_abs_mpa": float(max(d["max_abs_mpa"] for d in dvse))}
    fit = [r["contact_status"]["hertz_fit"]["half_width_mm"] / r["hertz_half_width_mm"] for r in recs if r["job"][2] is None]
    out["fit_half_width_over_hertz_own"] = summ(fit)
    out["solve_seconds_L1"] = summ([r["solve_seconds"] for r in recs])
    return out


def convergence(rows, test):
    mask = fem.gear_mesh_material_mask(64)
    lev = {k: [] for k in H}
    cps, ids = [], []
    for i in test:
        recs = {k: load(f"{k}_{i:03d}_own") for k in H}
        if any(v is None for v in recs.values()) or not all((FE / f"{k}_{i:03d}_own.npz").exists() for k in H):
            continue
        for k in H:
            lev[k].append(np.load(FE / f"{k}_{i:03d}_own.npz")["stress"] * mask[None])
        cps.append(recs["L1"]["contact_point_mm"])
        ids.append(i)
    if not ids:
        return {"n_cases": 0}
    with np.load(FE / f"L1_{ids[0]:03d}_own.npz") as g:
        gx, gy = g["x"], g["y"]
    cps = np.asarray(cps)
    dist = np.hypot(gx[None] - cps[:, 0, None, None], gy[None] - cps[:, 1, None, None])
    body = mask[None] & (dist > NEAR_MM)
    near = mask[None] & (dist <= NEAR_MM)
    u = {k: np.stack(v) for k, v in lev.items()}

    def d(a, b, m):
        return np.array([np.sqrt(((a[j] - b[j]) ** 2)[:, m[j]].sum() / (b[j] ** 2)[:, m[j]].sum()) for j in range(len(a))])

    full = np.broadcast_to(mask[None], body.shape)
    out = {"n_cases": len(ids)}
    for region, m in (("whole", full), ("body", body), ("near", near)):
        r = {"L1_L2": d(u["L1"], u["L2"], m), "L2_L3": d(u["L2"], u["L3"], m), "L1_L3": d(u["L1"], u["L3"], m)}
        blk = {k: summ(v) for k, v in r.items()}
        # observed order from the case-mean differences, assuming ||u_h - u*|| = C h^p
        h1, h2, h3 = H["L1"], H["L2"], H["L3"]
        a, b = r["L1_L2"].mean(), r["L2_L3"].mean()
        f = lambda p: (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - a / b
        try:
            p = brentq(f, 0.05, 6.0)
            e1 = a * h1 ** p / (h1 ** p - h2 ** p)
            blk["observed_order"] = float(p)
            blk["richardson_labels_to_limit"] = float(e1)
            blk["richardson_L3_to_limit"] = float(e1 * (h3 / h1) ** p)
        except ValueError:
            blk["observed_order"] = None
        out[region] = blk
    loads = {k: [load(f"{k}_{i:03d}_own")["normal_force_N_per_mm"] for i in ids] for k in H}
    out["normal_force_change"] = {"L2_vs_L1": summ(np.array(loads["L2"]) / np.array(loads["L1"]) - 1),
                                  "L3_vs_L1": summ(np.array(loads["L3"]) / np.array(loads["L1"]) - 1),
                                  "L3_vs_L2": summ(np.array(loads["L3"]) / np.array(loads["L2"]) - 1)}
    out["solve_seconds"] = {k: summ([load(f"{k}_{i:03d}_own")["solve_seconds"] for i in ids]) for k in H}
    return out


def grid_sensitivity(test):
    out = {}
    for k in H:
        recs = [load(f"{k}_{i:03d}_own") for i in test]
        recs = [r for r in recs if r is not None and "256" in r.get("root_peak_mpa", {})]
        if not recs:
            continue
        rp = {n: np.array([r["root_peak_mpa"][n] for r in recs]) for n in ("64", "128", "256")}
        out[k] = {"n": len(recs), **{f"peak_{n}_mpa": summ(v) for n, v in rp.items()},
                  "ratio_128_to_64": summ(rp["128"] / rp["64"]), "ratio_256_to_64": summ(rp["256"] / rp["64"])}
    common = [i for i in test if all((r := load(f"{k}_{i:03d}_own")) is not None and "256" in r.get("root_peak_mpa", {})
                                     for k in H)]
    if common:
        for n in ("64", "128", "256"):
            pk = {k: np.array([load(f"{k}_{i:03d}_own")["root_peak_mpa"][n] for i in common]) for k in H}
            out[f"labels_vs_refined_grid{n}"] = {"L1_vs_L2": summ(pk["L1"] / pk["L2"] - 1), "L1_vs_L3": summ(pk["L1"] / pk["L3"] - 1),
                                                 "L2_vs_L3": summ(pk["L2"] / pk["L3"] - 1)}
    return out


def contact_width(rows, test):
    split = test[:9]
    out = {}
    for fr in (1, 4, 7):
        i = split[fr]
        blk = {}
        for k in H:
            r = load(f"{k}_{i:03d}_own")
            if r is None:
                continue
            cs = r["contact_status"]
            blk[k] = {"s_mm": r["line_of_action_mm"], "normal_force": r["normal_force_N_per_mm"],
                      "hertz_b_mm": r["hertz_half_width_mm"], "hertz_p0_mpa": r["hertz_peak_pressure_mpa"],
                      "fe_peak_mpa": r["fe_peak_pressure_mpa"], "fit_b_mm": cs["hertz_fit"]["half_width_mm"],
                      "fit_p0_mpa": cs["hertz_fit"]["p0_mpa"], "fit_center_mm": cs["hertz_fit"]["center_mm"],
                      "loaded_nodes": cs["n_loaded"],
                      "width_loaded_nodes_mm": cs["width_loaded_nodes_mm"],
                      "profile": cs["profile"]}
        out[str(fr)] = blk
    return out


def main() -> None:
    rows, traj, test, _ = job_list()
    out = {"contact_all": contact_all(rows), "convergence": convergence(rows, test),
           "grid_sensitivity": grid_sensitivity(test), "contact_width": contact_width(rows, test)}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    ca, cv = out["contact_all"], out["convergence"]
    print(f"contact: {ca.get('n_cases', 0)} cases; convergence: {cv.get('n_cases', 0)} cases")
    if cv.get("n_cases"):
        for reg in ("whole", "body", "near"):
            b = cv[reg]
            print(f"  {reg}: L1-L2 {b['L1_L2']['mean']:.4f} L2-L3 {b['L2_L3']['mean']:.4f} L1-L3 {b['L1_L3']['mean']:.4f} "
                  f"order {b.get('observed_order')} labels-to-limit {b.get('richardson_labels_to_limit')}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

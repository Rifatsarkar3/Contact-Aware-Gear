"""Evaluate the filleted-root, deformable-rim model (datasets v5, v5f) and its surrogate.

Mirrors scripts/analyze_gear_mesh_surrogate.py for the reference ensemble on v4f:
the same 27 test cases (three held-out trajectories, own friction), the same
relative L2 errors over the material points, the same tooth-body metric (material
points more than 1 mm from the contact point), the same root peak (maximum in-plane
principal stress in the grid rows from the root circle to one module above it) and
the same friction-response test (mu = 0.04 against 0.12). In addition:

* convergence of the labels: the 27 test cases solved with every length scale
  refined 1.5-fold (L2) and 2-fold (L3); field differences between the levels,
  their observed order, and the nodal root peak (largest maximum principal nodal
  stress on the centre tooth's fillets, from the solver) at the three levels;
* the same quantities for the sharp-root model v4 where they exist, for comparison;
* contact and load summary of the 144 production cases at their own friction.

Grid differences between levels are taken over the grid points that are material
at every level (the fillet and flank polygons are sampled more finely at L2 and L3).

Usage: python scripts/analyze_fillet_rim.py
Output: outputs/fillet_rim/fillet_rim_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gearstress.fem as fem  # noqa: E402
from analyze_gear_mesh_surrogate import NEAR_MM, rel, rel_masked, root_peak, summ  # noqa: E402
from evaluate_surrogate_accuracy import CFG, predict, r2  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

V5F = ROOT / "data" / "processed" / "gear_mesh_quasistatic_v5f.h5"
SOLVED = ROOT / "outputs" / "fillet_rim_dataset"
CONTACT = ROOT / "outputs" / "surrogate" / "gear_mesh_v5_contact.json"
V4_RESULTS = ROOT / "outputs" / "surrogate" / "gear_mesh_v4_results.json"
OUT = ROOT / "outputs" / "fillet_rim" / "fillet_rim_results.json"
DS5F = "gear_mesh_quasistatic_v5f"
RATIO = (1.5, 2.0)          # refinement of L2 and L3 against L1


def observed_order(d12: float, d23: float) -> float:
    """p from |f1 - f2| / |f2 - f3| = (1 - 1.5^-p) / (1.5^-p - 2^-p); nan if no positive order fits."""
    f = lambda p: (1 - RATIO[0] ** -p) / (RATIO[0] ** -p - RATIO[1] ** -p) - d12 / d23
    try:
        return float(brentq(f, 0.01, 10.0))
    except ValueError:
        return float("nan")


def main() -> None:
    with h5py.File(V5F, "r") as f:
        x = np.asarray(f["inputs"], dtype=np.float32)
        y = np.asarray(f["targets"], dtype=np.float32)
        traj = np.asarray(f["trajectory_ids"])
        var = np.asarray(f["friction_variant"])
        cond = np.asarray(f["conditions"])
        dx, dy = float(f.attrs["grid_dx_mm"]), float(f.attrs["grid_dy_mm"])
    n0 = int((var == 0).sum())
    kw = dict(train_fraction=float(CFG["train_fraction"]), val_fraction=float(CFG["val_fraction"]), seed=42)
    s0, sf = grouped_trajectory_split(traj[:n0], **kw), grouped_trajectory_split(traj, **kw)
    test0 = sorted(int(i) for i in s0.test)
    assert sorted(set(traj[test0])) == sorted(set(traj[sf.test]))
    mats = np.stack([x[i, 0] for i in test0])
    g0 = np.load(SOLVED / f"L1_{test0[0]:03d}_own.npz")
    gx, gy = g0["x"], g0["y"]
    meta = {lv: [json.loads((SOLVED / f"{lv}_{i:03d}_own.json").read_text(encoding="utf-8")) for i in test0]
            for lv in ("L1", "L2", "L3")}
    grids = {lv: [np.load(SOLVED / f"{lv}_{i:03d}_own.npz") for i in test0] for lv in ("L2", "L3")}
    common = np.stack([(mats[j] > 0.5) & (grids["L2"][j]["mask"] > 0.5) & (grids["L3"][j]["mask"] > 0.5)
                       for j in range(len(test0))])
    far = np.stack([common[j] & (np.hypot(gx - meta["L1"][j]["contact_point_mm"][0],
                                          gy - meta["L1"][j]["contact_point_mm"][1]) > NEAR_MM)
                    for j in range(len(test0))])
    cm = common[:, None]
    refs = {"labels": np.stack([y[i] for i in test0]) * cm,
            "L2": np.stack([g["stress"] for g in grids["L2"]]) * cm,
            "L3": np.stack([g["stress"] for g in grids["L3"]]) * cm}
    out = {"test_indices": test0, "n0": n0, "near_mm": NEAR_MM,
           "material_points": {"labels": int(mats[0].sum()), "common_to_all_levels": summ(common.sum((1, 2)))}}

    # ---- convergence of the labels -------------------------------------------------------------
    conv = {}
    for name, keep in (("material", common), ("body", far), ("near", common & ~far)):
        d12 = [rel_masked(refs["labels"][j], refs["L2"][j], keep[j]) for j in range(len(test0))]
        d23 = [rel_masked(refs["L2"][j], refs["L3"][j], keep[j]) for j in range(len(test0))]
        d13 = [rel_masked(refs["labels"][j], refs["L3"][j], keep[j]) for j in range(len(test0))]
        orders = [observed_order(a, b) for a, b in zip(d12, d23)]
        agg = observed_order(float(np.mean(d12)), float(np.mean(d23)))
        conv[name] = {"labels_vs_L2": summ(d12), "L2_vs_L3": summ(d23), "labels_vs_L3": summ(d13),
                      "observed_order_per_case": summ([o for o in orders if np.isfinite(o)]),
                      "n_cases_with_order": int(np.isfinite(orders).sum()), "observed_order_of_means": agg}
    e2 = ((refs["labels"] - refs["L3"]) ** 2).sum(1)
    conv["near_share_labels_vs_L3"] = summ([e2[j][common[j] & ~far[j]].sum() / e2[j][common[j]].sum()
                                            for j in range(len(test0))])
    peaks = {lv: np.asarray([m["nodal_root_peak_mpa"] for m in meta[lv]]) for lv in meta}
    conv["nodal_root_peak"] = {
        "L1_vs_L2": summ((peaks["L1"] - peaks["L2"]) / peaks["L2"]),
        "L2_vs_L3": summ((peaks["L2"] - peaks["L3"]) / peaks["L3"]),
        "L1_vs_L3": summ((peaks["L1"] - peaks["L3"]) / peaks["L3"]),
        "abs_L1_vs_L3": summ(np.abs(peaks["L1"] - peaks["L3"]) / peaks["L3"]),
        "abs_L2_vs_L3": summ(np.abs(peaks["L2"] - peaks["L3"]) / peaks["L3"]),
        "range_L3_mpa": [float(peaks["L3"].min()), float(peaks["L3"].max())],
        "height_above_root_circle_mm": summ([np.hypot(*m["nodal_root_peak_at_mm"]) - fem.FilletGearMeshCase().root_radius_mm
                                             for m in meta["L3"]])}
    gpk = {k: root_peak(v, common) for k, v in refs.items()}
    conv["grid_root_peak"] = {"labels_vs_L2": summ((gpk["labels"] - gpk["L2"]) / gpk["L2"]),
                              "L2_vs_L3": summ((gpk["L2"] - gpk["L3"]) / gpk["L3"]),
                              "labels_vs_L3": summ((gpk["labels"] - gpk["L3"]) / gpk["L3"]),
                              "abs_labels_vs_L3": summ(np.abs(gpk["labels"] - gpk["L3"]) / gpk["L3"])}
    # root peak after projection onto 64, 128 and 256 grids (same band and layout as fe_summary.json of the main model)
    gpeak = {lv: {n: np.asarray([m["grid_root_peak_mpa"][n] for m in meta[lv]]) for n in ("64", "128", "256")}
             for lv in meta}
    gs = {lv: {**{f"peak_{n}_mpa": summ(gpeak[lv][n]) for n in ("64", "128", "256")},
               "ratio_128_to_64": summ(gpeak[lv]["128"] / gpeak[lv]["64"]),
               "ratio_256_to_64": summ(gpeak[lv]["256"] / gpeak[lv]["64"]), "n": len(test0)} for lv in meta}
    for n in ("64", "128", "256"):
        gs[f"labels_vs_refined_grid{n}"] = {
            "L1_vs_L2": summ(gpeak["L1"][n] / gpeak["L2"][n] - 1), "L1_vs_L3": summ(gpeak["L1"][n] / gpeak["L3"][n] - 1),
            "L2_vs_L3": summ(gpeak["L2"][n] / gpeak["L3"][n] - 1)}
    at = [m["grid_root_peak_at_mm"]["256"] for lv in meta for m in meta[lv]]
    gs["peak_position_256_mm"] = {"x": summ([a[0] for a in at]), "y": summ([a[1] for a in at])}
    # grid peak against the solver's largest nodal value on the fillets (the grid samples the surface at its spacing)
    gs["grid_vs_nodal_peak"] = {f"{lv}_{n}": summ(gpeak[lv][n] / peaks[lv] - 1) for lv in meta for n in ("64", "128", "256")}
    conv["grid_sensitivity"] = gs
    conv["elements"] = {lv: summ([m["elements"] for m in meta[lv]]) for lv in meta}
    conv["solve_seconds"] = {lv: summ([m["solve_seconds"] for m in meta[lv]]) for lv in meta}
    conv["fe_over_hertz"] = {lv: summ([m["fe_peak_pressure_mpa"] / m["hertz_peak_pressure_mpa"] for m in meta[lv]])
                             for lv in meta}
    conv["active_contact_nodes"] = {lv: summ([m["active_nodes"] for m in meta[lv]]) for lv in meta}
    out["convergence"] = conv
    per_case = {"index": test0, "trajectory": [int(traj[i]) for i in test0],
                "s_mm": [float(cond[i, 3]) for i in test0],
                "normal_force_N_per_mm": [m["normal_force_N_per_mm"] for m in meta["L1"]],
                "nodal_root_peak_mpa": {lv: peaks[lv].tolist() for lv in peaks},
                "field_L1_L2": [rel_masked(refs["labels"][j], refs["L2"][j], common[j]) for j in range(len(test0))],
                "field_L2_L3": [rel_masked(refs["L2"][j], refs["L3"][j], common[j]) for j in range(len(test0))],
                "body_L1_L2": [rel_masked(refs["labels"][j], refs["L2"][j], far[j]) for j in range(len(test0))],
                "body_L2_L3": [rel_masked(refs["L2"][j], refs["L3"][j], far[j]) for j in range(len(test0))]}

    # ---- the reference ensemble on v5f -----------------------------------------------------------
    scale = max(float(np.max(np.abs(y[sf.train]))), 1e-8)

    def pred(idx):
        xs = x[idx]
        p = predict(DS5F, "fno", "fno", {"x": torch.from_numpy(xs), "dx": dx, "dy": dy}).numpy() * scale
        return p * (xs[:, 0:1] > 0.5)

    p = pred(test0)
    ens = p.mean(0)
    res = {"n_seeds": int(len(p)), "target_scale_mpa": scale}
    labels_full = np.stack([y[i] for i in test0])
    res["ensemble_vs_labels"] = summ([rel(ens[j], labels_full[j]) for j in range(len(test0))])
    res["single_vs_labels"] = summ([np.mean([rel(p[s, j], labels_full[j]) for s in range(len(p))]) for j in range(len(test0))])
    res["reduction_single_to_ensemble"] = 1 - res["ensemble_vs_labels"]["mean"] / res["single_vs_labels"]["mean"]
    res["per_seed_mean_vs_labels"] = [float(np.mean([rel(p[s, j], labels_full[j]) for j in range(len(test0))]))
                                      for s in range(len(p))]
    per_case["ensemble_vs_labels"] = [rel(ens[j], labels_full[j]) for j in range(len(test0))]
    ensc = ens * cm
    for k in ("labels", "L2", "L3"):
        res[f"ensemble_vs_{k}_common"] = summ([rel_masked(ensc[j], refs[k][j], common[j]) for j in range(len(test0))])
        res[f"ensemble_vs_{k}_body"] = summ([rel_masked(ensc[j], refs[k][j], far[j]) for j in range(len(test0))])
    res["per_trajectory_vs_labels"] = {str(int(t)): float(np.mean([rel(ens[j], labels_full[j])
                                                                   for j, i in enumerate(test0) if traj[i] == t]))
                                       for t in sorted(set(traj[test0].tolist()))}
    material = mats > 0.5
    yt = torch.from_numpy(labels_full)
    res["r2"] = r2(torch.from_numpy(ens), yt, material)
    val = sorted(int(i) for i in sf.val)
    pv = pred(val).mean(0)
    rv_t, rv_p = root_peak(y[val], x[val][:, 0]), root_peak(pv, x[val][:, 0])
    bias = float(np.mean((rv_p - rv_t) / rv_t))
    tr, pe = root_peak(labels_full, mats), root_peak(ens, mats)
    raw = (pe - tr) / tr
    per_case["root_grid_label_mpa"], per_case["root_grid_ensemble_mpa"] = tr.tolist(), pe.tolist()
    res["root"] = {"validation_bias": bias,
                   "raw": {"mean_abs": float(np.mean(np.abs(raw))), "mean_signed": float(raw.mean()),
                           "worst_under": float(raw.min()), "n_under": int((raw < 0).sum())},
                   "truth_range_mpa": [float(tr.min()), float(tr.max())]}
    pec = root_peak(ensc, common)
    for k in ("L2", "L3"):
        e = (pec - gpk[k]) / gpk[k]
        res["root"][f"raw_vs_{k}"] = {"mean_abs": float(np.mean(np.abs(e))), "mean_signed": float(e.mean()),
                                     "worst_under": float(e.min()), "n_under": int((e < 0).sum())}
    # surrogate root peak against the solver's nodal root peak of the 2-fold refinement
    e = (pe - peaks["L3"]) / peaks["L3"]
    res["root"]["raw_vs_nodal_L3"] = {"mean_abs": float(np.mean(np.abs(e))), "mean_signed": float(e.mean()),
                                     "worst_under": float(e.min()), "n_under": int((e < 0).sum())}
    lo, hi = [i + n0 for i in test0], [i + 2 * n0 for i in test0]
    fe_field = [rel(y[h], y[l]) for l, h in zip(lo, hi)]
    fe_root = root_peak(y[hi], mats) / root_peak(y[lo], mats) - 1
    out["friction_fe"] = {"field_change": summ(fe_field), "root_change": summ(fe_root),
                          "root_change_abs": summ(np.abs(fe_root)), "n_root_rises": int((fe_root > 0).sum())}
    # as in the main model: away from the slip ramp at the pitch point, does the root change have the sign of s?
    off = [j for j, i in enumerate(test0) if abs(cond[i, 3]) > 0.05]
    out["friction_fe"]["root_sign_matches_s"] = int(sum(np.sign(fe_root[j]) == np.sign(cond[test0[j], 3]) for j in off))
    out["friction_fe"]["n_off_pitch"] = len(off)
    out["friction_fe"]["root_change_at_pitch"] = summ([fe_root[j] for j, i in enumerate(test0) if abs(cond[i, 3]) <= 0.05])
    pl, ph = pred(lo).mean(0), pred(hi).mean(0)
    sr = root_peak(ph, mats) / root_peak(pl, mats) - 1
    res["friction"] = {"field_change": summ([rel(ph[j], pl[j]) for j in range(len(test0))]),
                       "root_change_abs": summ(np.abs(sr)), "sign_agreement": int((np.sign(sr) == np.sign(fe_root)).sum()),
                       "root_change_mae_pp": float(np.mean(np.abs(sr - fe_root))),
                       "root_change_corr": float(np.corrcoef(sr, fe_root)[0, 1]),
                       "err_variants_vs_labels": summ([rel(pl[j], y[lo[j]]) for j in range(len(test0))] +
                                                      [rel(ph[j], y[hi[j]]) for j in range(len(test0))])}
    out["v5f"] = res
    per_case["ensemble_vs_L3_common"] = [rel_masked(ensc[j], refs["L3"][j], common[j]) for j in range(len(test0))]
    out["per_case"] = per_case
    # fields of the test case with the median label root peak, for the figure
    j = int(np.argsort(tr)[len(tr) // 2])
    out["figure_case"] = {"index": test0[j], "position": j, "s_mm": float(cond[test0[j], 3]),
                          "normal_force_N_per_mm": meta["L1"][j]["normal_force_N_per_mm"]}
    np.savez_compressed(OUT.parent / "fillet_case_fields.npz", label=labels_full[j], ensemble=ens[j], L3=refs["L3"][j],
                        mask=mats[j], x=gx, y=gy)

    # ---- contact and load summary of the production cases ---------------------------------------
    c = json.loads(CONTACT.read_text(encoding="utf-8"))
    by = {(r["job"][0], r["job"][1], r["job"][2]): r for r in c["rows"]}
    own = [by[("L1", i, None)] for i in range(n0)]
    every = [r for r in c["rows"] if r["job"][0] == "L1"]
    ramp = 0.05
    slid = [r for r in own if abs(r["line_of_action_mm"]) > ramp]
    out["contact"] = {
        "normal_force_N_per_mm": summ([r["normal_force_N_per_mm"] for r in own]),
        "normal_force_all_432": summ([r["normal_force_N_per_mm"] for r in every]),
        "hertz_p0_mpa": summ([r["hertz_peak_pressure_mpa"] for r in own]),
        "hertz_half_width_mm": summ([r["hertz_half_width_mm"] for r in own]),
        "fe_over_hertz": summ([r["fe_peak_pressure_mpa"] / r["hertz_peak_pressure_mpa"] for r in own]),
        "active_contact_nodes": summ([r["active_nodes"] for r in own]),
        "shear_over_mu_pressure_min_sliding_cases": summ([r["shear_over_mu_pressure_min"] for r in slid]),
        "n_sliding_cases": len(slid),
        "nodal_root_peak_mpa": summ([r["nodal_root_peak_mpa"] for r in own]),
        "root_peak_per_load_mpa_per_N_per_mm": summ([r["nodal_root_peak_mpa"] / r["normal_force_N_per_mm"] for r in own]),
        "solve_seconds": summ([r["solve_seconds"] for r in every]),
        "elements": summ([r["elements"] for r in every])}

    # ---- the sharp-root model for comparison ---------------------------------------------------
    if V4_RESULTS.exists():
        v4 = json.loads(V4_RESULTS.read_text(encoding="utf-8"))
        out["v4_for_comparison"] = {k: v4[k] for k in ("labels_vs_fineflank", "labels_vs_fineflank_body", "labels_vs_L3",
                                                        "labels_vs_L3_body", "root_labels_vs_L3") if k in v4}
        out["v4_for_comparison"]["v4f_ensemble_vs_labels"] = v4["v4f"]["ensemble_vs_labels"]
        out["v4_for_comparison"]["v4f_root_raw"] = v4["v4f"]["root"]["raw"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    cv, r = out["convergence"], out["v5f"]
    print(f"labels: L1-L2 {cv['material']['labels_vs_L2']['mean']:.4f}, L2-L3 {cv['material']['L2_vs_L3']['mean']:.4f}, "
          f"L1-L3 {cv['material']['labels_vs_L3']['mean']:.4f} (order {cv['material']['observed_order_of_means']:.2f}); "
          f"body L1-L2 {cv['body']['labels_vs_L2']['mean']:.4f}, L2-L3 {cv['body']['L2_vs_L3']['mean']:.4f} "
          f"(order {cv['body']['observed_order_of_means']:.2f})")
    print(f"nodal root peak: L1-L3 {100 * cv['nodal_root_peak']['L1_vs_L3']['mean']:+.2f}% "
          f"(|.| max {100 * cv['nodal_root_peak']['abs_L1_vs_L3']['max']:.2f}%), L2-L3 "
          f"{100 * cv['nodal_root_peak']['L2_vs_L3']['mean']:+.2f}%; grid root peak L1-L3 "
          f"{100 * cv['grid_root_peak']['labels_vs_L3']['mean']:+.2f}%")
    print(f"ensemble vs labels {r['ensemble_vs_labels']['mean']:.4f} (single {r['single_vs_labels']['mean']:.4f}), "
          f"R2 {r['r2']:.5f}; vs L3 {r['ensemble_vs_L3_common']['mean']:.4f} (body {r['ensemble_vs_L3_body']['mean']:.4f}); "
          f"root raw {100 * r['root']['raw']['mean_abs']:.2f}% (signed {100 * r['root']['raw']['mean_signed']:+.2f}%, "
          f"under {r['root']['raw']['n_under']}/27); vs L3 {100 * r['root']['raw_vs_L3']['mean_abs']:.2f}%; friction sign "
          f"{r['friction']['sign_agreement']}/27")


if __name__ == "__main__":
    main()

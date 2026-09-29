"""Evaluate the surrogates trained on the regenerated gear-mesh data (v4, v4f).

* Field accuracy of the ten-member FNO ensembles trained on v4 (90 training cases,
  friction fixed per trajectory) and on v4f (270 training cases, every case also
  at mu = 0.04 and 0.12) on the 27 original test cases, against the 0.12 mm
  labels, a 0.08 mm solution with the same 80-vertex flank, and a 0.08 mm solution
  with a 120-vertex flank.
* Peak root tensile stress (maximum in-plane principal stress in the root band),
  raw and with a bias estimated on each model's own validation cases.
* Response to friction: finite-element change between mu = 0.04 and 0.12 on each
  test case against the change each surrogate predicts when only its friction
  input changes.
* Contact and load summary from the solver output (normal load, friction force,
  sliding, Hertz comparison, refinement), and solve times.

Usage: python scripts/analyze_gear_mesh_surrogate.py
Output: outputs/surrogate/gear_mesh_v4_results.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_root_region_stress_error import ROOT_ROWS  # noqa: E402
from evaluate_surrogate_accuracy import CFG, predict, r2  # noqa: E402
from gearstress.losses import relative_l2  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

V4F = ROOT / "data" / "processed" / "gear_mesh_quasistatic_v4f.h5"
SOLVED = ROOT / "outputs" / "gear_mesh_v4_dataset"
CONTACT = ROOT / "outputs" / "surrogate" / "gear_mesh_v4_contact.json"
OUT = ROOT / "outputs" / "surrogate" / "gear_mesh_v4_results.json"
DS4, DS4F = "gear_mesh_quasistatic_v4", "gear_mesh_quasistatic_v4f"
FLANK_SEG = {80: None, 120: None}


def rel(a, b):
    return float(relative_l2(torch.from_numpy(np.asarray(a, np.float32))[None],
                             torch.from_numpy(np.asarray(b, np.float32))[None]))


def sigma1(s):
    return 0.5 * (s[:, 0] + s[:, 1]) + np.sqrt(0.25 * (s[:, 0] - s[:, 1]) ** 2 + s[:, 2] ** 2)


def root_peak(fields, mask):
    band = (mask > 0.5) & ROOT_ROWS[None, :, None]
    return np.where(band, sigma1(fields), -np.inf).reshape(len(fields), -1).max(1)


NEAR_MM = 1.0      # tooth-body metric: material pixels farther than this from the contact point


def rel_masked(a, b, keep):
    return float(np.linalg.norm((a - b)[:, keep]) / np.linalg.norm(b[:, keep]))


def summ(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "n": int(len(v))}


def main() -> None:
    with h5py.File(V4F, "r") as f:
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
    g0 = np.load(SOLVED / f"base_{test0[0]:03d}_own.npz")
    gx, gy = g0["x"], g0["y"]
    far = []
    for i in test0:
        cx, cy = json.loads((SOLVED / f"base_{i:03d}_own.json").read_text(encoding="utf-8"))["contact_point_mm"]
        far.append((x[i, 0] > 0.5) & (np.hypot(gx - cx, gy - cy) > NEAR_MM))
    refs = {"labels": np.stack([y[i] for i in test0]),
            "fine": np.stack([np.load(SOLVED / f"fine_{i:03d}_own.npz")["stress"] for i in test0]),
            "fineflank": np.stack([np.load(SOLVED / f"fineflank_{i:03d}_own.npz")["stress"] for i in test0])}
    out = {"test_indices": test0, "n0": n0,
           "masks_identical": {k: int(sum(np.array_equal(np.load(SOLVED / f"{k}_{i:03d}_own.npz")["mask"] > 0.5,
                                                          x[i, 0] > 0.5) for i in test0)) for k in ("fine", "fineflank")}}
    for k in ("fine", "fineflank"):
        out[f"labels_vs_{k}"] = summ([rel(refs["labels"][j], refs[k][j]) for j in range(len(test0))])
        out[f"labels_vs_{k}_body"] = summ([rel_masked(refs["labels"][j], refs[k][j], far[j]) for j in range(len(test0))])
        # share of the squared label-reference difference within NEAR_MM of the contact point
        e2 = ((refs["labels"] - refs[k]) ** 2).sum(1)
        out[f"labels_vs_{k}_near_share"] = summ([e2[j][(mats[j] > 0.5) & ~far[j]].sum() / e2[j][mats[j] > 0.5].sum()
                                                 for j in range(len(test0))])
        # share of the tooth-body difference inside the root band (root circle to one module above it)
        out[f"labels_vs_{k}_body_root_share"] = summ([e2[j][far[j] & ROOT_ROWS[:, None]].sum() / e2[j][far[j]].sum()
                                                      for j in range(len(test0))])
        rl, rr = root_peak(refs["labels"], mats), root_peak(refs[k], mats)
        out[f"root_labels_vs_{k}"] = {"abs": summ(np.abs((rl - rr) / rr)), "signed": summ((rl - rr) / rr),
                                      "n_under": int(((rl - rr) < 0).sum())}
    lo = [i + n0 for i in test0]
    hi = [i + 2 * n0 for i in test0]
    fe_field = [rel(y[h], y[l]) for l, h in zip(lo, hi)]
    fe_root = root_peak(y[hi], mats) / root_peak(y[lo], mats) - 1
    out["friction_fe"] = {"field_change": summ(fe_field), "root_change": summ(fe_root),
                          "root_change_abs": summ(np.abs(fe_root)), "n_root_rises": int((fe_root > 0).sum()),
                          "s_of_case": [float(cond[i, 3]) for i in test0],
                          "root_change_by_case": fe_root.tolist(), "field_change_by_case": fe_field}
    models = {"v4": (DS4, s0, y[:n0]), "v4f": (DS4F, sf, y)}
    for name, (ds, split, ypool) in models.items():
        scale = max(float(np.max(np.abs(ypool[split.train]))), 1e-8)

        def pred(idx):
            xs = x[idx]
            p = predict(ds, "fno", "fno", {"x": torch.from_numpy(xs), "dx": dx, "dy": dy}).numpy() * scale
            return p * (xs[:, 0:1] > 0.5)

        p = pred(test0)
        ens = p.mean(0)
        res = {"n_seeds": int(len(p)), "target_scale_mpa": scale}
        for k, ref in refs.items():
            res[f"ensemble_vs_{k}"] = summ([rel(ens[j], ref[j]) for j in range(len(test0))])
            res[f"ensemble_vs_{k}_body"] = summ([rel_masked(ens[j], ref[j], far[j]) for j in range(len(test0))])
        res["single_vs_labels"] = summ([np.mean([rel(p[s, j], refs["labels"][j]) for s in range(len(p))])
                                        for j in range(len(test0))])
        res["reduction_single_to_ensemble"] = 1 - res["ensemble_vs_labels"]["mean"] / res["single_vs_labels"]["mean"]
        res["per_trajectory_vs_labels"] = {str(int(t)): float(np.mean([rel(ens[j], refs["labels"][j])
                                                                       for j, i in enumerate(test0) if traj[i] == t]))
                                           for t in sorted(set(traj[test0].tolist()))}
        material = mats > 0.5
        yt = torch.from_numpy(refs["labels"])
        res["r2"] = r2(torch.from_numpy(ens), yt, material)
        res["r2_per_channel"] = [r2(torch.from_numpy(ens), yt, material, c) for c in range(3)]
        # root, with a bias estimated on the model's own validation cases
        val = sorted(int(i) for i in split.val)
        pv = pred(val).mean(0)
        rv_t, rv_p = root_peak(y[val], x[val][:, 0]), root_peak(pv, x[val][:, 0])
        bias = float(np.mean((rv_p - rv_t) / rv_t))
        tr, pe = root_peak(refs["labels"], mats), root_peak(ens, mats)
        raw, cal = (pe - tr) / tr, (pe / (1 + bias) - tr) / tr
        res["root"] = {"validation_bias": bias, "n_validation": len(val),
                       "raw": {"mean_abs": float(np.mean(np.abs(raw))), "mean_signed": float(raw.mean()),
                               "worst_under": float(raw.min()), "n_under": int((raw < 0).sum())},
                       "calibrated": {"mean_abs": float(np.mean(np.abs(cal))), "mean_signed": float(cal.mean()),
                                      "worst_under": float(cal.min()), "n_under": int((cal < 0).sum())},
                       "truth_range_mpa": [float(tr.min()), float(tr.max())]}
        for k in ("fine", "fineflank"):
            rr = root_peak(refs[k], mats)
            res["root"][f"raw_vs_{k}_mean_abs"] = float(np.mean(np.abs((pe - rr) / rr)))
        # friction response
        pl, ph = pred(lo).mean(0), pred(hi).mean(0)
        sf_field = [rel(ph[j], pl[j]) for j in range(len(test0))]
        sr = root_peak(ph, mats) / root_peak(pl, mats) - 1
        res["friction"] = {"field_change": summ(sf_field), "root_change_abs": summ(np.abs(sr)),
                           "sign_agreement": int((np.sign(sr) == np.sign(fe_root)).sum()),
                           "root_change_mae_pp": float(np.mean(np.abs(sr - fe_root))),
                           "root_change_corr": float(np.corrcoef(sr, fe_root)[0, 1]),
                           "err_variants_vs_labels": summ([rel(pl[j], y[lo[j]]) for j in range(len(test0))] +
                                                          [rel(ph[j], y[hi[j]]) for j in range(len(test0))]),
                           "root_change_by_case": sr.tolist(), "field_change_by_case": sf_field}
        out[name] = res
        print(f"{name}: ens vs labels {res['ensemble_vs_labels']['mean']:.4f} (single {res['single_vs_labels']['mean']:.4f},"
              f" -{100 * res['reduction_single_to_ensemble']:.1f}%), R2 {res['r2']:.4f}; vs fine "
              f"{res['ensemble_vs_fine']['mean']:.4f}; vs fineflank {res['ensemble_vs_fineflank']['mean']:.4f}; root raw "
              f"{100 * res['root']['raw']['mean_abs']:.2f}% (signed {100 * res['root']['raw']['mean_signed']:+.2f}%, "
              f"under {res['root']['raw']['n_under']}/27, worst {100 * res['root']['raw']['worst_under']:.1f}%), val-corrected "
              f"{100 * res['root']['calibrated']['mean_abs']:.2f}%; friction: field {100 * res['friction']['field_change']['mean']:.1f}% "
              f"sign {res['friction']['sign_agreement']}/27 corr {res['friction']['root_change_corr']:.2f} MAE "
              f"{100 * res['friction']['root_change_mae_pp']:.1f}pp; variants err {res['friction']['err_variants_vs_labels']['mean']:.4f}")
    for k in ("fine", "fineflank"):
        print(f"labels vs {k}: {out[f'labels_vs_{k}']['mean']:.4f} ({out[f'labels_vs_{k}']['min']:.4f}-"
              f"{out[f'labels_vs_{k}']['max']:.4f}); body {out[f'labels_vs_{k}_body']['mean']:.4f}; root "
              f"{100 * out[f'root_labels_vs_{k}']['abs']['mean']:.2f}%")
    for name in models:
        r = out[name]
        print(f"{name} tooth body (>{NEAR_MM} mm from contact): vs labels {r['ensemble_vs_labels_body']['mean']:.4f}, vs fine "
              f"{r['ensemble_vs_fine_body']['mean']:.4f}, vs fineflank {r['ensemble_vs_fineflank_body']['mean']:.4f}")
    out["near_mm"] = NEAR_MM
    out["near_pixel_share"] = summ([((mats[j] > 0.5) & ~far[j]).sum() / (mats[j] > 0.5).sum() for j in range(len(test0))])
    print(f"FE friction: field {100 * out['friction_fe']['field_change']['mean']:.1f}% "
          f"({100 * out['friction_fe']['field_change']['min']:.1f}-{100 * out['friction_fe']['field_change']['max']:.1f}), "
          f"|root| {100 * out['friction_fe']['root_change_abs']['mean']:.1f}% "
          f"({100 * out['friction_fe']['root_change']['min']:+.1f} to {100 * out['friction_fe']['root_change']['max']:+.1f}), "
          f"rises in {out['friction_fe']['n_root_rises']}/27")

    # contact and load summary from the solver output
    c = json.loads(CONTACT.read_text(encoding="utf-8"))
    rows = c["rows"]
    key = lambda r: (r["job"][0], r["job"][1], r["job"][2])
    by = {key(r): r for r in rows}
    own = [by[("base", i, None)] for i in range(n0)]
    ramp = 0.05
    slid = [r for r in own if abs(r["line_of_action_mm"]) > ramp]
    out["contact"] = {
        "normal_force_N_per_mm": summ([r["normal_force_N_per_mm"] for r in own]),
        "hertz_p0_mpa": summ([r["hertz_peak_pressure_mpa"] for r in own]),
        "hertz_half_width_mm": summ([r["hertz_half_width_mm"] for r in own]),
        "fe_over_hertz": summ([r["fe_peak_pressure_mpa"] / r["hertz_peak_pressure_mpa"] for r in own]),
        "contact_radius_mm": summ([r["contact_centroid_radius_mm"] for r in own]),
        "centroid_offset_mm": summ([abs(r["contact_centroid_radius_mm"] - r["nominal_contact_radius_mm"]) for r in own]),
        "abs_Ft_over_Fn_over_mu": summ([abs(r["tangential_over_normal"]) / r["friction"] for r in slid]),
        "shear_over_mu_pressure_min": summ([r["shear_over_mu_pressure_min"] for r in slid]),
        "sliding_within_1pct": int(sum(r["shear_over_mu_pressure_min"] >= 0.99 for r in slid)),
        "pitch_shear_over_mu_pressure_min": summ([r["shear_over_mu_pressure_min"] for r in own
                                                  if abs(r["line_of_action_mm"]) <= ramp]),
        "friction_sign_matches_s": int(sum(np.sign(r["tangential_force_toward_tip_N_per_mm"]) == np.sign(r["line_of_action_mm"])
                                           for r in slid)), "n_sliding_cases": len(slid),
        "pitch_cases_friction_sign": [float(np.sign(r["tangential_force_toward_tip_N_per_mm"])) for r in own
                                      if abs(r["line_of_action_mm"]) <= ramp],
        "solve_seconds": summ([r["solve_seconds"] for r in own]),
    }
    dF = [by[("base", i, 0.12)]["normal_force_N_per_mm"] / by[("base", i, 0.04)]["normal_force_N_per_mm"] - 1 for i in test0]
    out["contact"]["normal_force_change_mu004_to_012"] = summ(dF)
    out["contact"]["normal_force_change_refined"] = summ(
        [by[("fineflank", i, None)]["normal_force_N_per_mm"] / by[("base", i, None)]["normal_force_N_per_mm"] - 1 for i in test0])
    fe_root_sign_vs_s = [np.sign(fe_root[j]) == np.sign(cond[i, 3]) for j, i in enumerate(test0) if abs(cond[i, 3]) > ramp]
    out["friction_fe"]["root_sign_matches_s"] = int(sum(fe_root_sign_vs_s))
    out["friction_fe"]["n_off_pitch"] = len(fe_root_sign_vs_s)
    from gearstress.fem import GearMeshCase, conjugate_involute_flank
    for nv in FLANK_SEG:
        fl = conjugate_involute_flank(GearMeshCase(), nv)
        FLANK_SEG[nv] = float(np.mean(np.linalg.norm(np.diff(fl, axis=0), axis=1)))
    out["flank_segment_mm"] = FLANK_SEG
    hz = []
    for fr in (1, 4, 7):
        i = test0[fr]
        a, b = by[("base", i, None)], by[("fineflank", i, None)]
        hz.append({"frame": fr, "prod": a, "refined": b,
                   "segments_across_2b_prod": 2 * a["hertz_half_width_mm"] / FLANK_SEG[80],
                   "segments_across_2b_refined": 2 * b["hertz_half_width_mm"] / FLANK_SEG[120],
                   "elements_across_2b_prod": 2 * a["hertz_half_width_mm"] / 0.12,
                   "pixels_across_2b": 2 * a["hertz_half_width_mm"] / dx})
        d_prod = y[i + 2 * n0] - y[i + n0]
        rl = np.load(SOLVED / f"fineflank_{i:03d}_mu0.04.npz")["stress"]
        rh = np.load(SOLVED / f"fineflank_{i:03d}_mu0.12.npz")["stress"]
        hz[-1]["friction_effect_prod"] = rel(y[i + 2 * n0], y[i + n0])
        hz[-1]["friction_effect_refined"] = rel(rh, rl)
        hz[-1]["difference_field_mismatch"] = rel(d_prod, rh - rl)
    out["hertz_positions"] = hz
    cc = out["contact"]
    print(f"F' {cc['normal_force_N_per_mm']['min']:.0f}-{cc['normal_force_N_per_mm']['max']:.0f} N/mm (median "
          f"{cc['normal_force_N_per_mm']['median']:.0f}); p0 {cc['hertz_p0_mpa']['min']:.0f}-{cc['hertz_p0_mpa']['max']:.0f} MPa; "
          f"FE/Hertz {cc['fe_over_hertz']['min']:.2f}-{cc['fe_over_hertz']['max']:.2f} (median {cc['fe_over_hertz']['median']:.2f}); "
          f"radius {cc['contact_radius_mm']['min']:.2f}-{cc['contact_radius_mm']['max']:.2f}; |Ft/Fn|/mu "
          f"{cc['abs_Ft_over_Fn_over_mu']['min']:.2f}-{cc['abs_Ft_over_Fn_over_mu']['max']:.2f}; friction sign = sign(s) in "
          f"{cc['friction_sign_matches_s']}/{cc['n_sliding_cases']}; min shear/mu p {cc['shear_over_mu_pressure_min']['min']:.2f}; "
          f"dF'(mu) {100 * min(dF):+.1f}% to {100 * max(dF):+.1f}%; solve median {cc['solve_seconds']['median']:.1f} s")
    print(f"FE root friction sign = sign(s) in {out['friction_fe']['root_sign_matches_s']}/{out['friction_fe']['n_off_pitch']}")
    for h in hz:
        print(f"frame {h['frame']}: s={h['prod']['line_of_action_mm']:+.2f} F'={h['prod']['normal_force_N_per_mm']:.0f}/"
              f"{h['refined']['normal_force_N_per_mm']:.0f} p0={h['prod']['hertz_peak_pressure_mpa']:.0f} FE="
              f"{h['prod']['fe_peak_pressure_mpa']:.0f}/{h['refined']['fe_peak_pressure_mpa']:.0f} seg "
              f"{h['segments_across_2b_prod']:.1f}/{h['segments_across_2b_refined']:.1f} px {h['pixels_across_2b']:.1f}; "
              f"friction effect {100 * h['friction_effect_prod']:.1f}%/{100 * h['friction_effect_refined']:.1f}% mismatch "
              f"{100 * h['difference_field_mismatch']:.0f}%")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

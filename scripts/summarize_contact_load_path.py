"""Summary statistics of the contact load path (analyze_contact_load_path.py).

Reads outputs/surrogate/contact_load_path.json, the friction response of the
test cases (friction_response_test.json), the boundary-refined solutions and
the friction-resolved dataset, and writes contact_load_path_summary.json:

* over all 144 v2 cases: normal load, angle between the imposed displacement and
  the flank normal, imposed normal approach and slip, Hertz and FE peak pressure,
  contact radius, full-sliding check (shear / mu p and |Ft/Fn| / mu);
* over the 27 test cases: whether the sign of the friction effect on the peak root
  tensile stress follows the sign of the imposed slip, and how much the normal
  load changes between mu = 0.04 and 0.12 at the same imposed displacement;
* the Hertz-table positions at both discretizations (0.12 mm with an 80-vertex
  flank, 0.08 mm with a 120-vertex flank), with flank segments across 2b;
* whether the friction difference field (mu 0.12 minus mu 0.04) of the
  production discretization matches the boundary-refined one.

Usage: python scripts/summarize_contact_load_path.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gearstress.fem as fem  # noqa: E402

S = ROOT / "outputs" / "surrogate"
V2F = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v2f.h5"


def rng(v):
    v = np.asarray(v, dtype=float)
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()), "mean": float(v.mean())}


def rel(a, b):
    return float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel()))


def main() -> None:
    lp = json.loads((S / "contact_load_path.json").read_text(encoding="utf-8"))
    rows, test = lp["rows"], lp["test_indices"]
    own = [r for r in rows if r["kind"] == "load"][:144]            # job order: 144 own-friction solves first
    assert sorted(r["index"] for r in own) == list(range(144))
    ratio_mu = [abs(r["tangential_over_normal"]) / r["friction"] for r in own]
    out = {"all_cases": {
        "normal_force_N_per_mm": rng([r["normal_force_N_per_mm"] for r in own]),
        "displacement_angle_deg": rng([r["displacement_angle_to_flank_normal_deg"] for r in own]),
        "imposed_normal_approach_mm": rng([r["imposed_normal_approach_mm"] for r in own]),
        "imposed_slip_abs_mm": rng([abs(r["imposed_tangential_slip_toward_tip_mm"]) for r in own]),
        "slip_toward_tip_fraction": float(np.mean([r["imposed_tangential_slip_toward_tip_mm"] > 0 for r in own])),
        "hertz_p0_mpa": rng([r["hertz_peak_pressure_mpa"] for r in own]),
        "fe_peak_pressure_mpa": rng([r["fe_peak_pressure_mpa"] for r in own]),
        "fe_over_hertz": rng([r["fe_peak_pressure_mpa"] / r["hertz_peak_pressure_mpa"] for r in own]),
        "hertz_half_width_mm": rng([r["hertz_half_width_mm"] for r in own]),
        "contact_radius_mm": rng([r["contact_radius_mm"] for r in own]),
        "shear_over_mu_pressure_min": rng([r["shear_over_mu_pressure_min"] for r in own]),
        "abs_Ft_over_Fn_over_mu": rng(ratio_mu),
        "friction_force_sign_matches_slip": float(np.mean([np.sign(r["tangential_force_toward_tip_N_per_mm"]) ==
                                                           np.sign(r["imposed_tangential_slip_toward_tip_mm"])
                                                           for r in own])),
    }}
    # friction effect on the test cases
    fr = {r["index"]: r for r in json.loads((S / "friction_response_test.json").read_text(encoding="utf-8"))["rows"]}
    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze_contact_load_path import job_list
    jobs = job_list()[2]
    assert len(jobs) == len(rows) and all(j[0] == r["kind"] and j[1] == r["index"] for j, r in zip(jobs, rows))
    by = {j: r for j, r in zip(jobs, rows)}                    # key: (kind, index, mu or None for own friction)
    agree, dF, slip_sign = [], [], []
    for i in test:
        lo, hi, base = by[("load", i, 0.04)], by[("load", i, 0.12)], [r for r in own if r["index"] == i][0]
        s = np.sign(base["imposed_tangential_slip_toward_tip_mm"])
        slip_sign.append(float(s))
        agree.append(bool(np.sign(fr[i]["fe_root_sigma1_change"]) == s))
        dF.append(hi["normal_force_N_per_mm"] / lo["normal_force_N_per_mm"] - 1)
    agree_opp = [not a for a in agree]
    out["test_cases"] = {"n": len(test), "root_effect_sign_equals_slip_sign": int(sum(agree)),
                         "root_effect_sign_opposite_slip_sign": int(sum(agree_opp)),
                         "n_slip_toward_tip": int(sum(1 for s in slip_sign if s > 0)),
                         "normal_force_change_mu004_to_012": rng(dF),
                         "normal_force_change_abs": rng(np.abs(dF))}
    # Hertz-table positions at both discretizations
    case0 = fem.GearPairCase()
    flank80 = fem.conjugate_involute_flank(case0, 80)
    flank120 = fem.conjugate_involute_flank(case0, 120)
    seg80 = float(np.mean(np.linalg.norm(np.diff(flank80, axis=0), axis=1)))
    seg120 = float(np.mean(np.linalg.norm(np.diff(flank120, axis=0), axis=1)))
    out["flank_segment_mm"] = {"80_vertices": seg80, "120_vertices": seg120}
    hz = []
    for fr_i in (1, 4, 7):
        i = test[fr_i]
        a, b = [r for r in own if r["index"] == i][0], by[("refined", i, None)]
        hz.append({"frame": fr_i, "index": i, "prod": a, "refined": b,
                   "segments_across_2b_prod": 2 * a["hertz_half_width_mm"] / seg80,
                   "segments_across_2b_refined": 2 * b["hertz_half_width_mm"] / seg120,
                   "pixels_across_2b": 2 * a["hertz_half_width_mm"] / 0.1111})
    out["hertz_positions"] = hz
    # friction difference field: production vs boundary-refined discretization
    with h5py.File(V2F, "r") as f:
        y = np.asarray(f["targets"], dtype=np.float32)
        var = np.asarray(f["friction_variant"])
    n0 = int((var == 0).sum())
    diffs = []
    for fr_i in (1, 4, 7):
        i = test[fr_i]
        d_prod = y[i + 2 * n0] - y[i + n0]
        rl = np.load(S / "refined_flank_test" / f"refined_{i:03d}_mu0.04.npz")["stress"]
        rh = np.load(S / "refined_flank_test" / f"refined_{i:03d}_mu0.12.npz")["stress"]
        d_ref = rh - rl
        diffs.append({"frame": fr_i, "diff_field_rel_l2_prod_vs_refined": rel(d_prod, d_ref),
                      "friction_effect_prod": rel(y[i + 2 * n0], y[i + n0]), "friction_effect_refined": rel(rh, rl)})
    out["friction_difference_field"] = diffs
    (S / "contact_load_path_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    a = out["all_cases"]
    print(f"F' {a['normal_force_N_per_mm']['min']:.1f}-{a['normal_force_N_per_mm']['max']:.1f} N/mm (median "
          f"{a['normal_force_N_per_mm']['median']:.1f}); angle {a['displacement_angle_deg']['min']:.1f}-"
          f"{a['displacement_angle_deg']['max']:.1f} deg (median {a['displacement_angle_deg']['median']:.1f}); "
          f"p0 {a['hertz_p0_mpa']['min']:.0f}-{a['hertz_p0_mpa']['max']:.0f} MPa; FE/Hertz "
          f"{a['fe_over_hertz']['min']:.2f}-{a['fe_over_hertz']['max']:.2f}; radius {a['contact_radius_mm']['min']:.2f}-"
          f"{a['contact_radius_mm']['max']:.2f}; min shear/mu p {a['shear_over_mu_pressure_min']['min']:.3f}; "
          f"|Ft/Fn|/mu {a['abs_Ft_over_Fn_over_mu']['min']:.3f}-{a['abs_Ft_over_Fn_over_mu']['max']:.3f}; "
          f"friction sign = slip sign in {100 * a['friction_force_sign_matches_slip']:.0f}%")
    t = out["test_cases"]
    print(f"test: root friction effect sign = slip sign in {t['root_effect_sign_equals_slip_sign']}/{t['n']}; "
          f"dF'(mu .04->.12) {100 * t['normal_force_change_mu004_to_012']['min']:+.1f}% to "
          f"{100 * t['normal_force_change_mu004_to_012']['max']:+.1f}%")
    for h in hz:
        print(f"frame {h['frame']}: r={h['prod']['contact_radius_mm']:.2f} theta={h['prod']['displacement_angle_to_flank_normal_deg']:.1f} "
              f"F'={h['prod']['normal_force_N_per_mm']:.1f}/{h['refined']['normal_force_N_per_mm']:.1f} "
              f"p0={h['prod']['hertz_peak_pressure_mpa']:.0f} FE={h['prod']['fe_peak_pressure_mpa']:.0f}/"
              f"{h['refined']['fe_peak_pressure_mpa']:.0f} ratio {h['prod']['fe_peak_pressure_mpa'] / h['prod']['hertz_peak_pressure_mpa']:.2f}/"
              f"{h['refined']['fe_peak_pressure_mpa'] / h['refined']['hertz_peak_pressure_mpa']:.2f} "
              f"segments {h['segments_across_2b_prod']:.1f}/{h['segments_across_2b_refined']:.1f} px {h['pixels_across_2b']:.1f}")
    for d in diffs:
        print(f"frame {d['frame']}: friction effect {100 * d['friction_effect_prod']:.1f}% prod vs "
              f"{100 * d['friction_effect_refined']:.1f}% refined; difference-field mismatch "
              f"{100 * d['diff_field_rel_l2_prod_vs_refined']:.1f}%")


if __name__ == "__main__":
    main()

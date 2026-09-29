"""Supplementary statistics for the model comparisons and the root-stress calibration.

* Ensembling regime table evaluated over material pixels (the domain the
  headline accuracy uses), for v1 (30 seeds), v3, v2 and the v2 alternate
  partition.
* 95% confidence interval for Physics-FNO's field-accuracy cost on v2.
* Confidence interval for the data-scaling exponent.
* Peak root-fillet tensile stress (maximum in-plane principal stress): the
  headline ensemble's signed bias is estimated on the validation trajectories
  only and then applied, unchanged, to the test trajectories.

Usage:
    python scripts/analyze_comparison_statistics.py
Output: outputs/surrogate/comparison_statistics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_root_region_stress_error import ROOT_ROWS  # noqa: E402
from evaluate_surrogate_accuracy import CFG, predict  # noqa: E402
from gearstress.losses import relative_l2  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

OUT = ROOT / "outputs" / "surrogate" / "comparison_statistics.json"
SMOKE = ROOT / "outputs" / "smoke"


def split_data(dataset: str, split_seed: int, part: str):
    with h5py.File(ROOT / "data" / "processed" / f"{dataset}.h5", "r") as f:
        x = np.asarray(f["inputs"], dtype=np.float32)
        y = np.asarray(f["targets"], dtype=np.float32)
        traj = np.asarray(f["trajectory_ids"])
        dx, dy = float(f.attrs["grid_dx_mm"]), float(f.attrs["grid_dy_mm"])
    s = grouped_trajectory_split(traj, train_fraction=float(CFG["train_fraction"]),
                                 val_fraction=float(CFG["val_fraction"]), seed=split_seed)
    idx = getattr(s, part)
    scale = max(float(np.max(np.abs(y[s.train]))), 1e-8)
    return {"x": torch.from_numpy(x[idx]), "y": y[idx], "dx": dx, "dy": dy, "scale": scale, "traj": traj[idx]}


def rel(p, t):
    return float(relative_l2(torch.from_numpy(np.asarray(p, np.float32)), torch.from_numpy(np.asarray(t, np.float32))))


def sigma1(s):
    return 0.5 * (s[:, 0] + s[:, 1]) + np.sqrt(0.25 * (s[:, 0] - s[:, 1]) ** 2 + s[:, 2] ** 2)


def root_peak(fields, mask):
    band = (mask > 0.5) & ROOT_ROWS[None, :, None]
    return np.where(band, sigma1(fields), -np.inf).reshape(len(fields), -1).max(1)


def main() -> None:
    out = {}
    # 1. ensembling regime, material pixels
    regime = {}
    for label, ds, prefix, split_seed in (("v1_36_train_k30", "gear_pair_transient_quasistatic_v1", "fno", 42),
                                          ("v3_36_train", "gear_pair_transient_quasistatic_v3", "fno", 42),
                                          ("v2_90_train", "gear_pair_transient_quasistatic_v2", "fno", 42),
                                          ("v2_90_train_alternate_partition", "gear_pair_transient_quasistatic_v2",
                                           "fno_splitseed123", 123)):
        d = split_data(ds, split_seed, "test")
        p = predict(ds, prefix, "fno", d).numpy() * d["scale"]
        m = d["x"].numpy()[:, 0:1] > 0.5
        single = float(np.mean([rel(p[k] * m, d["y"]) for k in range(len(p))]))
        ens = rel(p.mean(0) * m, d["y"])
        regime[label] = {"k": int(len(p)), "single_material": single, "ensemble_material": ens,
                         "reduction": 1 - ens / single}
        print(f"regime {label}: k={len(p)} single {single:.4f} ensemble {ens:.4f} reduction {100 * (1 - ens / single):.1f}%")
    out["regime_material_pixels"] = regime

    # 2. Physics-FNO cost CI on v2 (paired over ten seeds, training-time test metric)
    run = lambda c, s: json.loads((SMOKE / "gear_pair_transient_quasistatic_v2" / f"{c}_seed{s}_ep300" / "results.json")
                                  .read_text(encoding="utf-8"))["test"]["relative_l2"]
    diffs = np.array([run("fno_physics", s) - run("fno", s) for s in range(1, 11)])
    base = np.mean([run("fno", s) for s in range(1, 11)])
    half = stats.t.ppf(0.975, 9) * diffs.std(ddof=1) / np.sqrt(10)
    out["physics_fno_cost_ci95_relative"] = [float((diffs.mean() - half) / base), float((diffs.mean() + half) / base)]
    print(f"Physics-FNO cost 95% CI: {100 * out['physics_fno_cost_ci95_relative'][0]:+.1f}% to "
          f"{100 * out['physics_fno_cost_ci95_relative'][1]:+.1f}%")

    # 3. scaling exponent CI (five nested sizes, ten-seed means)
    pts = json.loads((ROOT / "outputs" / "surrogate" / "surrogate_accuracy.json").read_text(encoding="utf-8"))["scaling_law_v2"]["points"]
    n = np.array([float(k) for k in pts]); e = np.array([pts[k]["mean_relative_l2"] for k in pts])
    fit = stats.linregress(np.log(n), np.log(e))
    ci = stats.t.ppf(0.975, len(n) - 2) * fit.stderr
    out["scaling_exponent"] = {"slope": fit.slope, "ci95": [fit.slope - ci, fit.slope + ci], "r2": fit.rvalue ** 2}
    print(f"scaling exponent {fit.slope:.2f} (95% CI {fit.slope - ci:.2f} to {fit.slope + ci:.2f}), R2 {fit.rvalue ** 2:.2f}")

    # 4. root tensile stress: bias fitted on validation, applied to test
    res = {}
    for part in ("val", "test"):
        d = split_data("gear_pair_transient_quasistatic_v2", 42, part)
        p = predict("gear_pair_transient_quasistatic_v2", "fno", "fno", d).numpy() * d["scale"]
        m = d["x"].numpy()[:, 0:1] > 0.5
        mask = d["x"].numpy()[:, 0]
        truth = root_peak(d["y"], mask)
        ens = root_peak(p.mean(0) * m, mask)
        seeds = np.stack([root_peak(p[k] * m, mask) for k in range(len(p))])
        res[part] = {"truth": truth, "ens": ens, "seed_sd": seeds.std(0, ddof=1)}
    bias_val = float(np.mean((res["val"]["ens"] - res["val"]["truth"]) / res["val"]["truth"]))
    t = res["test"]
    raw = (t["ens"] - t["truth"]) / t["truth"]
    corrected = (t["ens"] / (1 + bias_val) - t["truth"]) / t["truth"]
    out["root_sigma1"] = {
        "validation_bias": bias_val,
        "test_raw": {"mean_abs": float(np.mean(np.abs(raw))), "mean_signed": float(raw.mean()),
                     "worst_under": float(raw.min()), "share_under": float(np.mean(raw < 0))},
        "test_calibrated": {"mean_abs": float(np.mean(np.abs(corrected))), "mean_signed": float(corrected.mean()),
                            "worst_under": float(corrected.min()), "share_under": float(np.mean(corrected < 0))},
        "test_truth_range_mpa": [float(t["truth"].min()), float(t["truth"].max())]}
    r = out["root_sigma1"]
    print(f"root sigma1: validation bias {100 * bias_val:+.2f}% | test raw |err| {100 * r['test_raw']['mean_abs']:.2f}% "
          f"(signed {100 * r['test_raw']['mean_signed']:+.2f}%, worst {100 * r['test_raw']['worst_under']:+.1f}%, under in "
          f"{100 * r['test_raw']['share_under']:.0f}% of cases) | calibrated |err| {100 * r['test_calibrated']['mean_abs']:.2f}% "
          f"(signed {100 * r['test_calibrated']['mean_signed']:+.2f}%, worst {100 * r['test_calibrated']['worst_under']:+.1f}%)")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

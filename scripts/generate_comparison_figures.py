"""Comparison figures on the earlier datasets v1 to v3, from the committed result files.

physics_fno                 equilibrium residual and field error, vanilla FNO against Physics-FNO
fno_variants                ten FNO variants against the vanilla FNO on v1 and v2
robustness_checks           training-set size curve and the second v2 partition
active_learning             acquisition strategies against random selection
alternative_architectures   U-Net and DeepONet against the FNO
geometry_holdout            random split against a held-out pressure-angle band

Each figure asserts that the statistic it plots matches the reference value
reported alongside it, so a stale result file cannot produce a silent mismatch.
Figures are written as 600 dpi PNGs to outputs/figures/.

Usage:
    python scripts/generate_comparison_figures.py [name ...]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SMOKE = ROOT / "outputs" / "smoke"
V1 = SMOKE / "gear_pair_transient_quasistatic_v1"
V2 = SMOKE / "gear_pair_transient_quasistatic_v2"
V3 = SMOKE / "gear_pair_transient_quasistatic_v3"

NAVY = "#1f4e79"
TEAL = "#2e9e9e"
GRAY = "#8a8f94"
RED = "#c0392b"
AMBER = "#c58a1f"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10.5,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def check(actual: float, expected: float, tol: float, label: str) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"{label}: computed {actual!r} from the result files does not match "
            f"the reference value {expected!r} (tolerance {tol})"
        )


def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def physics_fno() -> None:
    v1 = load(V1 / "extended_multiseed_summary.json")
    v2 = load(V2 / "v2_multiseed_summary.json")

    check(v1["fno"]["relative_l2_mean"], 0.2181, 0.001, "v1 fno relative_l2")
    check(v1["fno_physics"]["equilibrium_residual_mean"], 343.60, 0.5, "v1 physics-fno residual")
    check(v2["fno_physics"]["equilibrium_residual_mean"], 391.50, 0.5, "v2 physics-fno residual")

    datasets = [("v1 (36 train)", v1), ("v2 (90 train)", v2)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))

    def sd(d: dict, model: str, metric: str) -> float:
        # sample SD over the ten seeds (the v1 summary stores population SD)
        return statistics.stdev(r[metric] for r in d[model]["runs"])

    ax = axes[0]
    x = range(2)
    w = 0.32
    fno_res = [d["fno"]["equilibrium_residual_mean"] for _, d in datasets]
    fno_res_e = [sd(d, "fno", "equilibrium_residual") for _, d in datasets]
    phys_res = [d["fno_physics"]["equilibrium_residual_mean"] for _, d in datasets]
    phys_res_e = [sd(d, "fno_physics", "equilibrium_residual") for _, d in datasets]
    ax.bar([i - w / 2 for i in x], fno_res, w, yerr=fno_res_e, capsize=3, color=NAVY, label="Vanilla FNO")
    ax.bar([i + w / 2 for i in x], phys_res, w, yerr=phys_res_e, capsize=3, color=TEAL, label="Physics-FNO")
    ax.set_xticks(list(x))
    ax.set_xticklabels([n for n, _ in datasets])
    ax.set_ylabel("Equilibrium residual")
    ax.set_title("(a) Equilibrium residual")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    fno_l2 = [d["fno"]["relative_l2_mean"] for _, d in datasets]
    fno_l2_e = [sd(d, "fno", "relative_l2") for _, d in datasets]
    phys_l2 = [d["fno_physics"]["relative_l2_mean"] for _, d in datasets]
    phys_l2_e = [sd(d, "fno_physics", "relative_l2") for _, d in datasets]
    ax.bar([i - w / 2 for i in x], fno_l2, w, yerr=fno_l2_e, capsize=3, color=NAVY, label="Vanilla FNO")
    ax.bar([i + w / 2 for i in x], phys_l2, w, yerr=phys_l2_e, capsize=3, color=TEAL, label="Physics-FNO")
    ax.set_xticks(list(x))
    ax.set_xticklabels([n for n, _ in datasets])
    ax.set_ylabel("Relative L2 field error")
    ax.set_title("(b) Relative L2 field error")
    ax.legend(frameon=False, fontsize=9)

    savefig(fig, "physics_fno.png")


def fno_variants() -> None:
    v1_ext = load(V1 / "extended_multiseed_summary.json")
    v2_multi = load(V2 / "v2_multiseed_summary.json")
    v2_rem = load(V2 / "v2_remaining_mechanisms_summary.json")

    fno_v1 = v1_ext["fno"]["relative_l2_mean"]
    fno_v2 = v2_multi["fno"]["relative_l2_mean"]

    def pct(mean, base):
        return (mean - base) / base * 100.0

    pct_v1, pct_v2 = {}, {}
    for m in ["aee_fno", "ccp_fno", "rcr_fno", "lpm_fno", "lpm_rcr_fno", "mice_lpm"]:
        pct_v1[m] = pct(v1_ext[m]["relative_l2_mean"], fno_v1)
    for m in ["ccm_fno", "hgm_fno", "dwm_fno"]:
        d = load(V1 / f"{m}_validated_summary.json")
        pct_v1[m] = pct(d["relative_l2_mean"], fno_v1)

    # DCT-FNO is capacity-matched: baseline is vanilla FNO retrained at DCT's
    # own width/modes (width=32, modes=16), not the default-capacity FNO.
    w32_v1 = [
        load(V1 / f"fno_w32_mx16_my16_l4_seed{s}_ep300" / "results.json")["test"]["relative_l2"]
        for s in range(1, 11)
    ]
    fno_w32_v1 = statistics.mean(w32_v1)
    dct_v1 = load(V1 / "dct_fno_validated_summary.json")["relative_l2_mean"]
    pct_v1["dct_fno"] = pct(dct_v1, fno_w32_v1)

    pct_v2["aee_fno"] = pct(v2_multi["aee_fno"]["relative_l2_mean"], fno_v2)
    for m in ["rcr_fno", "ccp_fno", "lpm_rcr_fno", "mice_lpm", "ccm_fno", "hgm_fno", "dwm_fno"]:
        pct_v2[m] = pct(v2_rem[m]["relative_l2_mean"], fno_v2)
    pct_v2["lpm_fno"] = pct(v2_multi["lpm_fno"]["relative_l2_mean"], fno_v2)
    fno_w32_v2 = v2_multi["fno_w32"]["relative_l2_mean"]
    pct_v2["dct_fno"] = pct(v2_rem["dct_fno"]["relative_l2_mean"], fno_w32_v2)

    check(pct_v1["aee_fno"], 1.5, 0.3, "v1 AEE-FNO pct")
    check(pct_v2["aee_fno"], 11.9, 0.3, "v2 AEE-FNO pct")
    check(pct_v1["ccp_fno"], 211.3, 1.0, "v1 CCP-FNO pct")
    check(pct_v2["ccp_fno"], 641.2, 2.0, "v2 CCP-FNO pct")
    check(pct_v1["dct_fno"], 30.2, 0.3, "v1 DCT-FNO pct (capacity-matched)")
    check(pct_v2["dct_fno"], 17.9, 0.3, "v2 DCT-FNO pct (capacity-matched)")
    check(pct_v1["dwm_fno"], 2.8, 0.3, "v1 DWM-FNO pct")
    check(pct_v2["dwm_fno"], 9.4, 0.3, "v2 DWM-FNO pct")

    labels_map = {
        "aee_fno": "AEE-FNO",
        "dct_fno": "DCT-FNO*",
        "ccm_fno": "CCM-FNO",
        "hgm_fno": "HGM-FNO",
        "dwm_fno": "DWM-FNO",
        "lpm_fno": "LPM-FNO",
        "rcr_fno": "RCR-FNO",
        "ccp_fno": "CCP-FNO",
        "lpm_rcr_fno": "LPM-RCR-FNO",
        "mice_lpm": "MICE-LPM",
    }

    small_family = ["ccm_fno", "hgm_fno", "aee_fno", "dwm_fno", "dct_fno"]
    large_family = ["lpm_fno", "rcr_fno", "mice_lpm", "lpm_rcr_fno", "ccp_fno"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))

    bars_for_legend = None
    for ax, family, title in [
        (axes[0], small_family, "(a) Variants with smaller differences"),
        (axes[1], large_family, "(b) Variants with larger differences"),
    ]:
        y = range(len(family))
        h = 0.35
        v1_vals = [pct_v1[m] for m in family]
        v2_vals = [pct_v2[m] for m in family]
        b1 = ax.barh([i + h / 2 for i in y], v1_vals, h, color=NAVY, label="v1 (36 train)")
        b2 = ax.barh([i - h / 2 for i in y], v2_vals, h, color=TEAL, label="v2 (90 train)")
        bars_for_legend = (b1, b2)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels([labels_map[m] for m in family])
        ax.set_xlabel("Relative change in error vs. vanilla FNO (%)")
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        ax.margins(x=0.12)

    fig.legend(
        handles=list(bars_for_legend),
        labels=["v1 (36 train)", "v2 (90 train)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        "Ten FNO variants against vanilla FNO, v1 and v2 (*capacity-matched baseline)",
        fontsize=10.5,
        y=1.14,
    )
    savefig(fig, "fno_variants.png")


def robustness_checks() -> None:
    curve = load(V2 / "train_size_curve" / "training_size_curve_summary.json")
    ns = sorted(int(k) for k in curve.keys())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    ax = axes[0]
    for model, color, label in [("fno", NAVY, "Vanilla FNO"), ("aee_fno", TEAL, "AEE-FNO"), ("dwm_fno", AMBER, "DWM-FNO")]:
        means = [curve[str(n)][model]["relative_l2_mean"] for n in ns]
        stds = [curve[str(n)][model]["relative_l2_std"] for n in ns]
        ax.errorbar(ns, means, yerr=stds, marker="o", color=color, label=label, capsize=3, linewidth=1.4)
    ax.set_xlabel("Training samples (N)")
    ax.set_ylabel("Relative L2 field error")
    ax.set_title("(a) Error against training-set size (v2, nested subsets)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    # Split-seed panel: original (v2, split_seed=42) vs alternate (split_seed=123) vs pooled.
    v2_multi = load(V2 / "v2_multiseed_summary.json")
    v2_rem = load(V2 / "v2_remaining_mechanisms_summary.json")
    alt = load(V2 / "split_seed_robustness_summary.json")

    def per_seed(d, model):
        return {r["seed"]: r["relative_l2"] for r in d[model]["runs"]}

    fno_orig = per_seed(v2_multi, "fno")
    aee_orig = per_seed(v2_multi, "aee_fno")
    dwm_orig = per_seed(v2_rem, "dwm_fno")
    fno_alt = per_seed(alt, "fno")
    aee_alt = per_seed(alt, "aee_fno")
    dwm_alt = per_seed(alt, "dwm_fno")

    def diffs(mech, base):
        return [mech[s] - base[s] for s in sorted(mech)]

    aee_orig_d = diffs(aee_orig, fno_orig)
    dwm_orig_d = diffs(dwm_orig, fno_orig)
    aee_alt_d = diffs(aee_alt, fno_alt)
    dwm_alt_d = diffs(dwm_alt, fno_alt)
    aee_pooled_d = aee_orig_d + aee_alt_d
    dwm_pooled_d = dwm_orig_d + dwm_alt_d

    check(statistics.mean(aee_orig_d), 0.01075, 0.0003, "AEE-FNO original-split mean diff")
    check(statistics.mean(dwm_orig_d), 0.00854, 0.0003, "DWM-FNO original-split mean diff")
    check(statistics.mean(aee_alt_d), 0.00434, 0.0003, "AEE-FNO alt-split mean diff")
    check(statistics.mean(dwm_alt_d), -0.00437, 0.0003, "DWM-FNO alt-split mean diff")
    check(statistics.mean(aee_pooled_d), 0.00754, 0.0003, "AEE-FNO pooled mean diff")
    check(statistics.mean(dwm_pooled_d), 0.00209, 0.0003, "DWM-FNO pooled mean diff")

    ax = axes[1]
    # Seeds from two partitions are not exchangeable (the partition is the unit of
    # replication), so the two partitions are shown side by side and not pooled.
    groups = ["Original partition\n(n=10 seeds)", "Second partition\n(n=10 seeds)"]
    aee_means = [statistics.mean(aee_orig_d), statistics.mean(aee_alt_d)]
    aee_sems = [statistics.stdev(v) / len(v) ** 0.5 for v in (aee_orig_d, aee_alt_d)]
    dwm_means = [statistics.mean(dwm_orig_d), statistics.mean(dwm_alt_d)]
    dwm_sems = [statistics.stdev(v) / len(v) ** 0.5 for v in (dwm_orig_d, dwm_alt_d)]

    x = range(2)
    w = 0.32
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.bar([i - w / 2 for i in x], aee_means, w, yerr=aee_sems, capsize=3, color=TEAL, label="AEE-FNO")
    ax.bar([i + w / 2 for i in x], dwm_means, w, yerr=dwm_sems, capsize=3, color=AMBER, label="DWM-FNO")
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("Paired mean diff. in relative L2\n(mechanism - vanilla FNO)")
    ax.set_title("(b) Difference from vanilla FNO on two partitions", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    savefig(fig, "robustness_checks.png")


def active_learning() -> None:
    d = load(ROOT / "outputs" / "active_learning" / "full_study_summary.json")
    rounds = ["round_2", "round_4", "round_6"]
    ns = [d["comparison"]["pool_based"][r]["n_train"] for r in rounds]

    random_means = []
    for r in rounds:
        vals = [x["relative_l2"] for x in d["random_baseline"][r]["per_seed_results"]]
        random_means.append(statistics.mean(vals))

    check(d["comparison"]["pool_based"]["round_6"]["relative_l2_mean"], 0.0697, 0.0002, "pool_based N=144")
    check(d["comparison"]["mc_dropout"]["round_6"]["relative_l2_mean"], 0.2265, 0.0002, "mc_dropout N=144")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for method, color, label in [
        ("pool_based", TEAL, "pool_based"),
        ("ensemble_surrogate", NAVY, "ensemble_surrogate"),
        ("mc_dropout", RED, "mc_dropout"),
    ]:
        vals = [d["comparison"][method][r]["relative_l2_mean"] for r in rounds]
        ax.plot(ns, vals, marker="o", color=color, label=method, linewidth=1.6)
    ax.plot(ns, random_means, marker="s", color=GRAY, linestyle="--", label="random_baseline", linewidth=1.4)
    ax.set_xlabel("Training samples (N)")
    ax.set_ylabel("Relative L2 field error")
    ax.set_title("Acquisition strategies and random selection")
    ax.set_xticks(ns)
    ax.legend(frameon=False, fontsize=9)
    savefig(fig, "active_learning.png")


def alternative_architectures() -> None:
    rows = []
    for tag, base in [("v1 (36 train)", V1), ("v2 (90 train)", V2)]:
        unet = load(base / "unet_baseline_summary.json")
        deep = load(base / "deeponet_baseline_summary.json")
        rows.append(
            (
                tag,
                {
                    "Vanilla FNO": (unet["fno"]["relative_l2_mean"], unet["fno"]["relative_l2_std"]),
                    "U-Net (default)": (unet["unet"]["relative_l2_mean"], unet["unet"]["relative_l2_std"]),
                    "U-Net (matched)": (unet["unet_w38"]["relative_l2_mean"], unet["unet_w38"]["relative_l2_std"]),
                    "DeepONet (searched)": (
                        deep["deeponet_searched"]["relative_l2_mean"],
                        deep["deeponet_searched"]["relative_l2_std"],
                    ),
                    "DeepONet (matched)": (
                        deep["deeponet_capacity_matched"]["relative_l2_mean"],
                        deep["deeponet_capacity_matched"]["relative_l2_std"],
                    ),
                },
            )
        )

    check(rows[0][1]["U-Net (default)"][0], 0.5280, 0.001, "v1 U-Net default")
    check(rows[1][1]["DeepONet (searched)"][0], 0.8020, 0.001, "v2 DeepONet searched")

    models = ["Vanilla FNO", "U-Net (default)", "U-Net (matched)", "DeepONet (searched)", "DeepONet (matched)"]
    colors = [NAVY, TEAL, "#7fc4c4", AMBER, "#e0bf80"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, (tag, vals) in zip(axes, rows):
        means = [vals[m][0] for m in models]
        stds = [vals[m][1] for m in models]
        ax.bar(range(len(models)), means, yerr=stds, capsize=3, color=colors)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8.5)
        ax.set_title(tag, fontsize=10)
        ax.set_yscale("log")
    axes[0].set_ylabel("Relative L2 field error (log scale)")
    fig.suptitle("U-Net and DeepONet against vanilla FNO", fontsize=10.5, y=1.03)
    savefig(fig, "alternative_architectures.png")


def geometry_holdout() -> None:
    d = load(SMOKE / "gear_pair_geometry_v1" / "geometry_generalization_summary.json")
    conds = ["random_split", "geometry_holdout"]
    labels = ["random_split\n(in-distribution)", "geometry_holdout\n(out-of-distribution)"]
    medians = [d[c]["relative_l2_median_mean"] for c in conds]
    med_std = [d[c]["relative_l2_median_std"] for c in conds]

    check(medians[0], 0.2963, 0.001, "geometry random_split median")
    check(medians[1], 0.3535, 0.001, "geometry holdout median")

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.bar(range(2), medians, yerr=med_std, capsize=4, color=[NAVY, RED], width=0.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median relative L2 field error")
    ax.set_title("Random split and geometry holdout")
    savefig(fig, "geometry_holdout.png")


FIGURES = {f.__name__: f for f in (physics_fno, fno_variants, robustness_checks, active_learning,
                                   alternative_architectures, geometry_holdout)}


def main() -> None:
    for name in sys.argv[1:] or list(FIGURES):
        FIGURES[name]()


if __name__ == "__main__":
    main()

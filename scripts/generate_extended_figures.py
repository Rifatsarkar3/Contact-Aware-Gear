"""Figures of the robustness and comparison analyses (release 3.1.0), drawn from the result files only.

fig_robustness  (a) ensemble error of each test trajectory in every partition in
                which it is tested; (b) error of each case of the six additional
                trajectories against its contact position, with the original test
                cases of the reference partition at the nine fixed positions
fig_comparison  (a) single-network and ensemble errors of the five network
                families and the POD baseline on v4f; (b) equilibrium residual of
                each family's ensemble prediction relative to that of the label
figS_variants   relative change in error of the ten FNO variants against the
                vanilla FNO on the earlier datasets v1 and v2, material-point metric
figS_contact    nodal contact pressure along the flank at three positions and
                three discretizations, with the fitted and the Hertz profiles

Every plotted statistic that is also reported in a table is asserted against the
reported value before the figure is saved.

Usage: python scripts/generate_extended_figures.py [robustness] [comparison] [variants] [contact]
Output: outputs/figures/ (600 dpi PNG)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "outputs" / "extended_study"
OUT = Path(os.environ.get("FIG_OUT", ROOT / "outputs" / "figures"))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
                     "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white"})
BLUE, GREY, RED, GOLD, TEAL = "#1f4e79", "#8a8f94", "#c0392b", "#c58a1f", "#2e9e9e"
FAM = {"fno": "FNO", "fno_physics": "Physics-FNO", "unet": "U-Net", "deeponet": "DeepONet", "pod_deeponet": "POD-DeepONet"}
TICK = {"fno_physics": "Physics-\nFNO", "pod_deeponet": "POD-\nDeepONet"}     # two-line tick labels


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png")


def close(a: float, b: float, label: str, tol: float = 1e-6) -> None:      # float32 averages differ at ~1e-7
    assert abs(a - b) <= tol * max(1.0, abs(b)), f"{label}: plotted {a} against reported {b}"


def fig_robustness() -> None:
    part = json.loads((EXT_DIR / "extended_results.json").read_text(encoding="utf-8"))["partitions"]
    folds = part["folds"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for k in sorted(folds, key=int):
        f = folds[k]
        bc = f["by_case"]
        e, t = np.asarray(bc["ensemble_vs_labels"]), np.asarray(bc["trajectory"])
        close(float(e.mean()), f["ensemble_vs_labels"]["mean"], f"partition {k} mean")
        for tr, v in f["per_trajectory"].items():
            close(float(e[t == int(tr)].mean()), v, f"partition {k} trajectory {tr}")
        tr = np.array([int(t) for t in f["per_trajectory"]])
        v = np.array(list(f["per_trajectory"].values()))
        if k == "0":
            ax.plot(tr, v, "o", color=BLUE, ms=6, zorder=4, label="reference partition")
        else:
            ax.plot(tr + 0.12 * (int(k) - 3), v, "o", mfc="none", mec=GREY, ms=5, zorder=3,
                    label="partitions 1 to 5" if k == "1" else None)
    ax.set_xticks(range(16))
    ax.set_xlabel("Trajectory")
    ax.set_ylabel("Ensemble relative L2 error")
    ax.set_ylim(bottom=0)
    ax.set_title("(a) Test trajectories over six partitions", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", color="#e3e3e3", lw=0.6)
    ax = axes[1]
    f0 = folds["0"]
    own, ex = f0["by_case"], f0["extra"]["by_case"]
    close(float(np.mean(ex["ensemble_vs_labels"])), f0["extra"]["ensemble_vs_labels"]["mean"], "extra mean")
    ax.axvspan(-0.05, 0.05, color="#efe6d4", lw=0, zorder=0)
    ax.axvline(0, color="#999999", lw=0.6, ls=":", zorder=1)
    ax.plot(own["s_mm"], own["ensemble_vs_labels"], "o", mfc="none", mec=GREY, ms=5, zorder=3,
            label="original test cases (nine fixed positions)")
    ax.plot(ex["s_mm"], ex["ensemble_vs_labels"], "x", color=BLUE, ms=5, mew=1.2, zorder=4,
            label="additional trajectories")
    allv = np.concatenate([own["ensemble_vs_labels"], ex["ensemble_vs_labels"]])
    if allv.max() / allv.min() > 8:
        ax.set_yscale("log")
        ax.set_ylim(allv.min() / 1.3, allv.max() * 2.6)      # free band above the data for the legend
    else:
        ax.set_ylim(0, allv.max() * 1.35)
    ax.set_xlabel("Contact position s on the line of action (mm)")
    ax.set_ylabel("Relative L2 error per case")
    ax.set_title("(b) Unseen positions and approaches", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", color="#e3e3e3", lw=0.6)
    fig.tight_layout()
    save(fig, "fig_robustness")


def fig_comparison() -> None:
    res = json.loads((EXT_DIR / "extended_results.json").read_text(encoding="utf-8"))
    cmp_, pod = res["comparison"], res["pod"]["v4f"]
    fams = [f for f in FAM if f in cmp_]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, f in enumerate(fams):
        c = cmp_[f]
        s = np.asarray(c["single_27_per_seed"])
        close(float(s.mean()), c["single_27"]["mean"], f"{f} single mean")
        ax.plot(i + rng.uniform(-0.12, 0.12, len(s)), s, "o", mfc="none", mec=BLUE if f == "fno" else GREY, ms=4.5,
                zorder=3, label="single networks (ten seeds)" if i == 0 else None)
        ax.plot(i, c["ensemble_27"]["mean"], "D", color=BLUE if f == "fno" else "#4d5359", ms=6.5, zorder=4,
                label="ten-member ensemble" if i == 0 else None)
    ax.plot(len(fams), pod["test_relative_l2"]["mean"], "s", color=GOLD, ms=6.5, zorder=4, label="POD (deterministic)")
    ax.set_xticks(range(len(fams) + 1))
    ax.set_xticklabels([TICK.get(f, FAM[f]) for f in fams] + ["POD"], fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(top=max(1.0, ax.get_ylim()[1]))       # show the 10^0 decade above the largest errors
    ax.set_ylabel("Relative L2 error, 27 test cases")
    ax.set_title("(a) Field error", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", color="#e3e3e3", lw=0.6, which="both")
    ax = axes[1]
    data = []
    for f in fams:
        e = np.asarray(cmp_[f]["equilibrium_ratio_27_by_case"])
        close(float(e.mean()), cmp_[f]["equilibrium_ratio_27"]["mean"], f"{f} equilibrium ratio")
        data.append(e)
    bp = ax.boxplot(data, positions=range(len(fams)), widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "lw": 1})
    for patch, f in zip(bp["boxes"], fams):
        patch.set_facecolor("#d6e2ee" if f == "fno" else "#e6e7e8")
        patch.set_edgecolor("#4d5359")
    for i, e in enumerate(data):
        ax.plot(i + rng.uniform(-0.15, 0.15, len(e)), e, ".", color="#4d5359", ms=3, zorder=3)
    ax.axhline(1.0, color=RED, lw=0.8, ls="--")
    ax.text(len(fams) - 1.5, 1.0, "label", color=RED, fontsize=8, va="bottom", ha="center")   # in the gap between the last two boxes
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([TICK.get(f, FAM[f]) for f in fams], fontsize=8.5)
    ax.set_ylabel("Residual of prediction / residual of label")
    ax.set_title("(b) Equilibrium residual of the ensemble", fontsize=9.5)
    ax.grid(axis="y", color="#e3e3e3", lw=0.6)
    fig.tight_layout()
    save(fig, "fig_comparison")


def figS_variants() -> None:
    ev = json.loads((EXT_DIR / "earlier_variants.json").read_text(encoding="utf-8"))
    rows = {d: {r["name"]: r for r in ev["datasets"][d]["comparison"] if r["family"] == "variant"} for d in ("v1", "v2")}
    names = list(rows["v2"])
    chg = {d: {n: 100 * rows[d][n]["material"]["relative_change"] for n in names} for d in ("v1", "v2")}
    small = sorted([n for n in names if max(abs(chg["v1"][n]), abs(chg["v2"][n])) < 50], key=lambda n: chg["v2"][n])
    large = sorted([n for n in names if n not in small], key=lambda n: chg["v2"][n])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, group, title in ((axes[0], small, "(a) Variants with smaller differences"),
                             (axes[1], large, "(b) Variants with larger differences")):
        y = np.arange(len(group))
        ax.barh(y - 0.18, [chg["v1"][n] for n in group], 0.36, color=BLUE, label="v1 (36 training cases)")
        ax.barh(y + 0.18, [chg["v2"][n] for n in group], 0.36, color=TEAL, label="v2 (90 training cases)")
        ax.set_yticks(y)
        ax.set_yticklabels([n + ("*" if rows["v2"][n]["reference"] != "fno" else "") for n in group])
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_xlabel("Change in error against the FNO (%)")
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="x", color="#e3e3e3", lw=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "figS_variants")


def figS_contact() -> None:
    cw = json.loads((EXT_DIR / "fe_summary.json").read_text(encoding="utf-8"))["contact_width"]
    levels = [k for k in ("L1", "L2", "L3") if all(k in cw[fr] for fr in ("1", "4", "7"))]
    names = {"L1": "production (0.12 mm, 80 vertices)", "L2": "1.5-fold (0.08 mm, 120)", "L3": "2-fold (0.06 mm, 160)"}
    fig, axes = plt.subplots(len(levels), 3, figsize=(8.4, 2.3 * len(levels)), sharex=True, squeeze=False)
    for i, k in enumerate(levels):
        for j, fr in enumerate(("1", "4", "7")):
            c = cw[fr][k]
            ax = axes[i, j]
            xi, p = np.asarray(c["profile"]["xi_mm"]), np.asarray(c["profile"]["p_mpa"])
            xx = np.linspace(xi.min(), xi.max(), 400)
            hz = lambda b, p0, x0: p0 * np.sqrt(np.clip(1 - ((xx - x0) / b) ** 2, 0, None))
            # centre of the fitted profile, shared by the Hertz curve so that only the shapes are compared
            x0 = c.get("fit_center_mm", 0.0)
            ax.plot(xx, hz(c["hertz_b_mm"], c["hertz_p0_mpa"], x0), color=RED, lw=1.0, label="Hertz, from the FE load")
            ax.plot(xx, hz(c["fit_b_mm"], c["fit_p0_mpa"], x0), color=BLUE, lw=1.0, ls="--", label="Hertz shape fitted")
            ax.plot(xi, p, "o", ms=3.5, mfc="none", mec="#333333", label="nodal pressure")
            close(float(p.max()), c["fe_peak_mpa"], f"peak pressure frame {fr} {k}", tol=1e-3)
            ax.text(0.02, 0.95, f"b = {c['hertz_b_mm']:.3f} (Hertz)\nb = {c['fit_b_mm']:.3f} (fit)", transform=ax.transAxes,
                    fontsize=7, va="top")
            if i == 0:
                ax.set_title(f"s = {c['s_mm']:+.2f} mm".replace("-", "−"), fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"{names[k]}\np (MPa)", fontsize=8)
            if i == len(levels) - 1:
                ax.set_xlabel("Distance along the flank (mm)")
            ax.grid(color="#eeeeee", lw=0.5)
    h = [Line2D([], [], color=RED, lw=1.0, label="Hertz profile from the finite-element load"),
         Line2D([], [], color=BLUE, lw=1.0, ls="--", label="Hertz profile fitted to the nodal pressures"),
         Line2D([], [], ls="none", marker="o", ms=4, mfc="none", mec="#333333", label="nodal pressure")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "figS_contact")


if __name__ == "__main__":
    which = sys.argv[1:] or ["robustness", "comparison", "variants", "contact"]
    for w in which:
        {"robustness": fig_robustness, "comparison": fig_comparison, "variants": figS_variants, "contact": figS_contact}[w]()

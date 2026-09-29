"""Figures of the corrected gear-mesh study (datasets v4 and v4f), from the result files.

model      the finite-element tooth pair: mesh, fixed and displaced rims, contact
           point, line of action and the 64 x 64 prediction window
fields     finite-element label, ten-member surrogate and their difference
           (plane-strain von Mises, MPa) at three positions of a test trajectory
root       peak root tensile stress: prediction against finite element on the test set
friction   change between mu = 0.04 and 0.12: finite element against both ensembles

Run scripts/analyze_gear_mesh_surrogate.py first. Figures are written as 600 dpi
PNGs to outputs/figures/.

Usage:
    python scripts/generate_figures.py [model|fields|root|friction ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_root_region_stress_error import ROOT_ROWS  # noqa: E402
from evaluate_surrogate_accuracy import CFG, predict  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

OUT = ROOT / "outputs" / "figures"
NU = 0.30
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
                     "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png")


def vm_plane_strain(s):
    x, y, z = s[..., 0, :, :], s[..., 1, :, :], s[..., 2, :, :]
    zz = NU * (x + y)
    return np.sqrt(np.maximum(0.5 * ((x - y) ** 2 + (y - zz) ** 2 + (zz - x) ** 2) + 3 * z * z, 0.0))


V4 = ROOT / "data" / "processed" / "gear_mesh_quasistatic_v4.h5"
V4F = ROOT / "data" / "processed" / "gear_mesh_quasistatic_v4f.h5"
V4_RESULTS = ROOT / "outputs" / "surrogate" / "gear_mesh_v4_results.json"
V4_CASE = ROOT / "outputs" / "surrogate" / "model_figure_case_v4"


def load_v4(path=V4):
    with h5py.File(path, "r") as f:
        x = np.asarray(f["inputs"], dtype=np.float32)
        y = np.asarray(f["targets"], dtype=np.float32)
        traj = np.asarray(f["trajectory_ids"])
        dx, dy = float(f.attrs["grid_dx_mm"]), float(f.attrs["grid_dy_mm"])
    split = grouped_trajectory_split(traj, train_fraction=float(CFG["train_fraction"]),
                                     val_fraction=float(CFG["val_fraction"]), seed=42)
    scale = max(float(np.max(np.abs(y[split.train]))), 1e-8)
    return x, y, traj, dx, dy, split, scale


def fig_model_v4() -> None:
    """Mesh of one v4 case (first test trajectory, middle position) with the loading."""
    import gearstress.fem as fem
    res = json.loads(V4_RESULTS.read_text(encoding="utf-8"))
    job = res["hertz_positions"][0]["prod"]          # a position away from the pitch point
    case = fem.GearMeshCase(line_of_action_mm=job["line_of_action_mm"], approach_mm=job["approach_mm"],
                            friction=job["friction"], youngs_modulus_mpa=job["youngs_modulus_mpa"])
    if not (V4_CASE / "case.inp").exists():
        fem.build_gear_mesh_case(case, V4_CASE)
    text = (V4_CASE / "case.inp").read_text(encoding="ascii", errors="replace").splitlines()
    nodes, elems, sets, mode, cur = {}, [], {}, None, None
    for line in text:
        if line.startswith("*"):
            mode = None
            u = line.upper()
            if u.startswith("*NODE") and "FILE" not in u and "PRINT" not in u:
                mode = "node"
            elif u.startswith("*ELEMENT"):
                mode = "elem"
            m = re.match(r"\*NSET,\s*NSET=(\w+)", line, re.I)
            if m:
                mode, cur = "nset", m.group(1).upper()
                sets[cur] = []
            continue
        if mode == "node":
            p = line.split(",")
            nodes[int(p[0])] = (float(p[1]), float(p[2]))
        elif mode == "elem":
            p = [int(v) for v in line.split(",") if v.strip()]
            elems.append(p[1:4])
        elif mode == "nset":
            sets[cur] += [int(v) for v in line.split(",") if v.strip()]
    meta = json.loads((V4_CASE / "case.json").read_text(encoding="utf-8"))
    segs = []
    for a, b, c in elems:
        pa, pb, pc = nodes[a], nodes[b], nodes[c]
        segs += [(pa, pb), (pb, pc), (pc, pa)]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.add_collection(LineCollection(segs, colors="#9aa3ab", linewidths=0.12))
    for name, color in (("DRIVERROOT", "#1f4e79"), ("MATINGROOT", "#c0392b")):
        pts = np.array([nodes[n] for n in sets.get(name, [])])
        ax.scatter(pts[:, 0], pts[:, 1], s=1.2, color=color, zorder=3)
    cp = np.array(meta["contact_point_mm"]); nrm = np.array(meta["contact_normal"])
    centre = np.array(meta["mating_center_mm"])
    pitch = 0.5 * centre
    ax.plot(*cp, marker="o", ms=5, mfc="none", mec="#c58a1f", mew=1.4, zorder=5)
    ax.plot(*pitch, marker="x", ms=6, color="#333333", mew=1.2, zorder=5)
    ax.plot(*np.column_stack((pitch - 3.0 * nrm, pitch + 3.0 * nrm)), color="#c58a1f", lw=1.0, ls="--", zorder=4)
    ax.annotate("", xy=tuple(cp), xytext=tuple(cp + 1.8 * nrm), arrowprops=dict(arrowstyle="->", color="#c58a1f", lw=1.3))
    g = np.load(sorted((ROOT / "outputs" / "gear_mesh_v4_dataset").glob("base_*_own.npz"))[0])
    x0, x1, y0, y1 = g["x"].min(), g["x"].max(), g["y"].min(), g["y"].max()
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#2e9e9e", lw=1.2, ls="--"))
    ax.set_aspect("equal"); ax.autoscale_view()
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker="o", ms=5, color="#1f4e79", label="driver rim below root circle: fixed"),
               Line2D([], [], ls="none", marker="o", ms=5, color="#c0392b", label="mating rim: displaced (approach and slip)"),
               Line2D([], [], ls="none", marker="o", ms=6, mfc="none", mec="#c58a1f", mew=1.4, label="contact point"),
               Line2D([], [], color="#c58a1f", lw=1.0, ls="--", label="line of action (approach direction)"),
               Line2D([], [], ls="none", marker="x", ms=6, color="#333333", label="pitch point"),
               Line2D([], [], ls="--", color="#2e9e9e", lw=1.2, label=f"64 × 64 prediction window ({x1 - x0:.1f} × {y1 - y0:.1f} mm)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7.5, frameon=False)
    save(fig, "fig_model")


def fig_fields_v4() -> None:
    x, y, traj, dx, dy, split, scale = load_v4()
    scale_f = load_v4(V4F)[6]
    t0 = int(traj[sorted(split.test)[0]])
    first = int(np.flatnonzero(traj == t0).min())
    frames = (1, 4, 7)
    idxs = [first + fr for fr in frames]
    data = {"x": torch.from_numpy(x[idxs]), "dx": dx, "dy": dy}
    p = predict("gear_mesh_quasistatic_v4f", "fno", "fno", data).numpy().mean(0) * scale_f
    p = p * (x[idxs][:, 0:1] > 0.5)
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 7.2), constrained_layout=True)
    g0 = np.load(sorted((ROOT / "outputs" / "gear_mesh_v4_dataset").glob("base_*_own.npz"))[0])
    ext = (g0["x"].min(), g0["x"].max(), g0["y"].min(), g0["y"].max())
    for r, (i, fr) in enumerate(zip(idxs, frames)):
        mat = x[i, 0] > 0.5
        fe, sg = vm_plane_strain(y[i]), vm_plane_strain(p[r])
        vmax = float(np.percentile(fe[mat], 99.5))
        err = np.where(mat, sg - fe, np.nan)
        lim = float(np.nanpercentile(np.abs(err), 99.5))
        for c, (img, cmap, lo, hi, title) in enumerate((
                (np.where(mat, fe, np.nan), "viridis", 0, vmax, "Finite element (0.12 mm)"),
                (np.where(mat, sg, np.nan), "viridis", 0, vmax, "Surrogate (10-member ensemble)"),
                (err, "RdBu_r", -lim, lim, "Surrogate minus finite element"))):
            ax = axes[r, c]
            im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"position {fr + 1} of 9", fontsize=9)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            cb.set_label("MPa", fontsize=7)
    save(fig, "fig_fields")


def fig_root_v4() -> None:
    x, y, traj, dx, dy, split, scale = load_v4()
    scale_f = load_v4(V4F)[6]
    test = sorted(int(i) for i in split.test)
    data = {"x": torch.from_numpy(x[test]), "dx": dx, "dy": dy}
    m = x[test][:, 0:1] > 0.5
    p4 = predict("gear_mesh_quasistatic_v4", "fno", "fno", data).numpy().mean(0) * scale * m
    pf = predict("gear_mesh_quasistatic_v4f", "fno", "fno", data).numpy().mean(0) * scale_f * m

    def s1(s):
        return 0.5 * (s[:, 0] + s[:, 1]) + np.sqrt(0.25 * (s[:, 0] - s[:, 1]) ** 2 + s[:, 2] ** 2)

    band = (x[test][:, 0] > 0.5) & ROOT_ROWS[None, :, None]
    peak = lambda f: np.where(band, s1(f), -np.inf).reshape(len(test), -1).max(1)
    truth, a, b = peak(y[test]), peak(p4), peak(pf)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    lim = [0, float(max(truth.max(), a.max(), b.max()) * 1.05)]
    ax.plot(lim, lim, color="#555555", lw=0.8, ls="--", label="y = x")
    ax.scatter(truth, a, s=22, facecolors="none", edgecolors="#8a8f94", label="trained on v4 (144 cases)", zorder=3)
    ax.scatter(truth, b, s=14, color="#1f4e79", label="trained on v4f (432 cases)", zorder=4)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Finite-element peak root tensile stress (MPa)")
    ax.set_ylabel("Predicted peak root tensile stress (MPa)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig_root")


def fig_friction_v4() -> None:
    res = json.loads(V4_RESULTS.read_text(encoding="utf-8"))
    fe = np.array(res["friction_fe"]["root_change_by_case"]) * 100
    s = np.array(res["friction_fe"]["s_of_case"])
    s4 = np.array(res["v4"]["friction"]["root_change_by_case"]) * 100
    sf = np.array(res["v4f"]["friction"]["root_change_by_case"]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    ax = axes[0]
    ax.axhline(0, color="#999999", lw=0.6); ax.axvline(0, color="#999999", lw=0.6, ls=":")
    order = np.argsort(s)
    ax.plot(s[order], fe[order], "o", color="#c0392b", ms=4, label="finite element")
    ax.plot(s[order], s4[order], "o", mfc="none", mec="#8a8f94", ms=5, label="trained on v4 (144 cases)")
    ax.plot(s[order], sf[order], "x", color="#1f4e79", ms=5, label="trained on v4f (432 cases)")
    ax.set_xlabel("Contact position s on the line of action (mm)")
    ax.set_ylabel("Change of peak root tensile stress (%)")
    ax.set_title("(a) Root stress, μ = 0.04 to 0.12", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    fef = np.array(res["friction_fe"]["field_change_by_case"]) * 100
    s4f = np.array(res["v4"]["friction"]["field_change_by_case"]) * 100
    sff = np.array(res["v4f"]["friction"]["field_change_by_case"]) * 100
    idx = np.arange(len(fef))
    ax.plot(idx, fef, "o-", color="#c0392b", ms=3, lw=1, label="finite element")
    ax.plot(idx, s4f, "o-", color="#8a8f94", ms=3, lw=1, label="trained on v4 (144 cases)")
    ax.plot(idx, sff, "o-", color="#1f4e79", ms=3, lw=1, label="trained on v4f (432 cases)")
    ax.set_ylim(0, 1.15 * max(fef.max(), s4f.max(), sff.max()))
    ax.set_xlabel("Test case (three trajectories × nine positions)"); ax.set_ylabel("Relative L2 change (%)")
    ax.set_title("(b) Whole-field change, μ = 0.04 to 0.12", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=3, handlelength=1.6, handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "fig_friction")


if __name__ == "__main__":
    figures = {"model": fig_model_v4, "fields": fig_fields_v4, "root": fig_root_v4, "friction": fig_friction_v4}
    for w in sys.argv[1:] or list(figures):
        figures[w]()

"""Robustness and comparison analyses of the corrected datasets (release 3.1.0).

* partitions: the vanilla-FNO ensemble on the reference partition and on the five
  further trajectory-grouped partitions of v4f (gearstress.splits), each tested on
  its own three held-out trajectories, and on six additional trajectories;
* comparison: FNO, Physics-FNO, U-Net, DeepONet and POD-DeepONet after the matched screening of
  scripts/run_extended_training.py, and a POD baseline, on the reference partition;
* pod: POD with linear or polynomial ridge regression of the mode coefficients,
  with the projection error of the test fields onto the retained basis reported
  separately from the regression error (v4f, and the earlier dataset v2);
* distance: field error binned by distance from the contact point.

All field errors are relative L2 errors over the material pixels (grid points
inside the finite elements of the driving tooth), averaged over cases.

Usage: python scripts/analyze_extended_study.py [partitions] [comparison] [pod] [distance]
Output: outputs/extended_study/extended_results.json (updated section by section)
"""
from __future__ import annotations

import json
import re
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
from gearstress.losses import equilibrium_residual  # noqa: E402
from gearstress.models import DeepONet, FNO2d, PODDeepONet, UNetSmall  # noqa: E402
from gearstress.splits import grouped_trajectory_fold_split, grouped_trajectory_split  # noqa: E402

DATA = ROOT / "data" / "processed"
SMOKE = ROOT / "outputs" / "smoke"
EXT = ROOT / "outputs" / "extended_study"
OUT = EXT / "extended_results.json"
V4F = "gear_mesh_quasistatic_v4f"
N0 = 144                                   # cases per friction variant in v4f
BINS_MM = (0.0, 0.25, 0.5, 1.0, 2.0, np.inf)


def load(name: str):
    with h5py.File(DATA / f"{name}.h5", "r") as f:
        return {k: np.asarray(f[k]) for k in f.keys()} | {"attrs": dict(f.attrs)}


def run_dirs(dataset: str, prefix: str, epochs: int = 300):
    pat = re.compile(rf"^{re.escape(prefix)}_seed(\d+)_ep{epochs}$")
    runs = [(int(m.group(1)), p) for p in (SMOKE / dataset).iterdir() if (m := pat.match(p.name))]
    return [p for _, p in sorted(runs) if (p / "results.json").exists()]


def build(run: Path) -> tuple[torch.nn.Module, float]:
    res = json.loads((run / "results.json").read_text(encoding="utf-8"))
    a, name = res["model_architecture"], res["model"]
    if name in ("fno", "fno_physics"):
        model = FNO2d(width=a["width"], modes_x=a["modes_x"], modes_y=a["modes_y"], layers=a["layers"])
    elif name == "unet":
        model = UNetSmall(width=a["width"])
    elif name == "deeponet":
        model = DeepONet(width=a["width"], basis_dim=int(re.search(r"_bd(\d+)", run.name).group(1)))
    elif name == "pod_deeponet":            # the fixed POD basis is restored from the state dict
        model = PODDeepONet(rank=int(re.search(r"_bd(\d+)", run.name).group(1)), width=a["width"])
    else:
        raise ValueError(name)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
    return model.eval(), float(res["target_scale_mpa_or_native"])


def n_params(model: torch.nn.Module) -> dict:
    """PyTorch's count (a complex weight counts once), the count of real numbers, and the fixed (untrained) values."""
    return {"torch": int(sum(p.numel() for p in model.parameters())),
            "real": int(sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())),
            "fixed": int(sum(b.numel() for b in model.buffers()))}


@torch.no_grad()
def predict(runs, x: np.ndarray) -> np.ndarray:
    """Predictions in MPa of every run, (n_runs, n_cases, 3, 64, 64), zero outside the material."""
    xt, mask = torch.from_numpy(x.astype(np.float32)), x[:, 0:1] > 0.5
    out = []
    for run in runs:
        model, scale = build(run)
        out.append(np.concatenate([model(xt[i:i + 32]).numpy() for i in range(0, len(xt), 32)]) * scale * mask)
    return np.stack(out)


def rel(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Relative L2 error per case over the pixels in m (cases along the first axis)."""
    sel = np.broadcast_to(m[:, None], a.shape)
    num = np.sqrt(((a - b) ** 2 * sel).reshape(len(a), -1).sum(1))
    den = np.sqrt((b ** 2 * sel).reshape(len(b), -1).sum(1))
    return num / den


def r2(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    sel = np.broadcast_to(m[:, None], a.shape)
    p, t = a[sel], b[sel]
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def sigma1(s):
    return 0.5 * (s[:, 0] + s[:, 1]) + np.sqrt(0.25 * (s[:, 0] - s[:, 1]) ** 2 + s[:, 2] ** 2)


def root_peak(s, m):
    band = m & ROOT_ROWS[None, :, None]
    return np.where(band, sigma1(s), -np.inf).reshape(len(s), -1).max(1)


def grad_rms(s, m):
    """RMS of the central-difference gradient of the three stress components per case, over the material points
    whose four neighbours are also material (MPa per grid spacing)."""
    inner = np.zeros_like(m)
    inner[:, 1:-1, 1:-1] = m[:, 1:-1, 1:-1] & m[:, :-2, 1:-1] & m[:, 2:, 1:-1] & m[:, 1:-1, :-2] & m[:, 1:-1, 2:]
    gy = 0.5 * (s[:, :, 2:, 1:-1] - s[:, :, :-2, 1:-1])
    gx = 0.5 * (s[:, :, 1:-1, 2:] - s[:, :, 1:-1, :-2])
    sel = inner[:, None, 1:-1, 1:-1]
    return np.sqrt(((gx ** 2 + gy ** 2) * sel).sum((1, 2, 3)) / (3 * sel.sum((1, 2, 3))))


def mde_paired(sd_diff: float, n: int, alpha: float, power: float = 0.8) -> float:
    """Smallest mean paired difference that a two-sided paired t-test at level alpha detects with the given power
    (noncentral t distribution, n pairs)."""
    from scipy import optimize
    df, tc = n - 1, stats.t.ppf(1 - alpha / 2, n - 1)

    def gap(d):
        nc = d * np.sqrt(n) / sd_diff
        lower = np.nan_to_num(stats.nct.cdf(-tc, df, nc), nan=0.0)        # negligible tail; NaN at large nc
        return stats.nct.sf(tc, df, nc) + lower - power
    return float(optimize.brentq(gap, 1e-9 * sd_diff, 3 * sd_diff))      # for n >= 10 the power is ~1 at 3 s.d.


def summ(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def root_block(pred, true, m):
    rp, rt = root_peak(pred, m), root_peak(true, m)
    e = (rp - rt) / rt
    return {"mean_abs": float(np.abs(e).mean()), "mean_signed": float(e.mean()), "worst_under": float(e.min()),
            "n_under": int((e < 0).sum()), "n": int(e.size)}


def friction_block(pred_lo, pred_hi, true_lo, true_hi, m, s_of_case):
    fe = root_peak(true_hi, m) / root_peak(true_lo, m) - 1
    pr = root_peak(pred_hi, m) / root_peak(pred_lo, m) - 1
    off = np.abs(s_of_case) > 0.05
    return {"fe_field_change": summ(rel(true_hi, true_lo, m)), "pred_field_change": summ(rel(pred_hi, pred_lo, m)),
            "sign_agreement": int((np.sign(fe) == np.sign(pr)).sum()), "n": int(fe.size),
            "fe_sign_matches_s_off_pitch": int((np.sign(fe[off]) == np.sign(s_of_case[off])).sum()), "n_off_pitch": int(off.sum()),
            "root_change_mae_pp": float(100 * np.abs(fe - pr).mean()), "root_change_corr": float(np.corrcoef(fe, pr)[0, 1])}


def update(section: str, value) -> None:
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data[section] = value
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote section {section!r} of {OUT.relative_to(ROOT)}")


# ------------------------------------------------------------------------------ partitions
def partitions() -> dict:
    d, ex = load(V4F), load("gear_mesh_quasistatic_v4_extra")
    x, y, traj, cond, var = d["inputs"], d["targets"], d["trajectory_ids"], d["conditions"], d["friction_variant"]
    m_all = x[:, 0] > 0.5
    out = {"folds": {}}
    per_traj = {}
    for k in range(6):
        split = grouped_trajectory_fold_split(traj, k)
        prefix = "fno" if k == 0 else f"fno_fold{k}"
        runs = run_dirs(V4F, prefix)
        if len(runs) < 10:
            print(f"fold {k}: {len(runs)} runs, skipped")
            continue
        test = np.asarray(sorted(int(i) for i in split.test))
        own = test[var[test] == 0]
        p = predict(runs, x[own])
        ens = p.mean(0)
        m = m_all[own]
        e = rel(ens, y[own], m)
        f = {"test_trajectories": sorted(int(t) for t in set(traj[test].tolist())),
             "val_trajectories": sorted(int(t) for t in set(traj[split.val].tolist())),
             "ensemble_vs_labels": summ(e), "single_vs_labels": summ(np.mean([rel(p[s], y[own], m) for s in range(len(p))], 0)),
             "r2": r2(ens, y[own], m), "root": root_block(ens, y[own], m),
             "per_trajectory": {str(int(t)): float(e[traj[own] == t].mean()) for t in sorted(set(traj[own].tolist()))},
             "by_case": {"s_mm": cond[own, 3].tolist(), "trajectory": traj[own].tolist(), "ensemble_vs_labels": e.tolist()}}
        for t, v in f["per_trajectory"].items():
            per_traj.setdefault(t, []).append(v)
        # normal load seen in training (all friction variants) against the load of each evaluated test trajectory
        train = np.asarray(sorted(int(i) for i in split.train))
        f["load"] = {"train_max_normal_force": float(cond[train, 5].max()),
                     "test_max_normal_force_by_trajectory": {str(int(t)): float(cond[own][traj[own] == t, 5].max())
                                                             for t in sorted(set(traj[own].tolist()))},
                     "extra_max_normal_force": float(ex["conditions"][:, 5].max())}
        lo, hi = own + N0, own + 2 * N0
        p_lo, p_hi = predict(runs, x[lo]).mean(0), predict(runs, x[hi]).mean(0)
        f["friction_variants_vs_labels"] = summ(np.concatenate([rel(p_lo, y[lo], m), rel(p_hi, y[hi], m)]))
        f["friction"] = friction_block(p_lo, p_hi, y[lo], y[hi], m, cond[own, 3])
        pe = predict(runs, ex["inputs"])
        ee, me = pe.mean(0), ex["inputs"][:, 0] > 0.5
        f["extra"] = {"ensemble_vs_labels": summ(rel(ee, ex["targets"], me)), "r2": r2(ee, ex["targets"], me),
                      "by_case": {"s_mm": ex["conditions"][:, 3].tolist(), "trajectory": ex["trajectory_ids"].tolist(),
                                  "ensemble_vs_labels": rel(ee, ex["targets"], me).tolist()},
                      "root": root_block(ee, ex["targets"], me),
                      "per_trajectory": {str(int(t)): float(rel(ee, ex["targets"], me)[ex["trajectory_ids"] == t].mean())
                                         for t in sorted(set(ex["trajectory_ids"].tolist()))}}
        out["folds"][str(k)] = f
        print(f"fold {k}: test {f['test_trajectories']} ens {e.mean():.4f} single {f['single_vs_labels']['mean']:.4f} "
              f"root {100 * f['root']['mean_abs']:.2f}% friction signs {f['friction']['sign_agreement']}/27 "
              f"extra {f['extra']['ensemble_vs_labels']['mean']:.4f}")
    folds = out["folds"].values()
    out["across_partitions"] = {k: summ([f[k]["mean"] for f in folds]) for k in ("ensemble_vs_labels", "single_vs_labels")}
    out["across_partitions"]["root_mean_abs"] = summ([f["root"]["mean_abs"] for f in folds])
    out["across_partitions"]["friction_sign_agreement"] = [f["friction"]["sign_agreement"] for f in folds]
    out["across_partitions"]["extra_ensemble_vs_labels"] = summ([f["extra"]["ensemble_vs_labels"]["mean"] for f in folds])
    out["per_trajectory_when_tested"] = {t: v for t, v in sorted(per_traj.items(), key=lambda kv: int(kv[0]))}
    out["extra_parameters"] = {"mu": summ(ex["conditions"][:, 1]), "E_mpa": summ(ex["conditions"][:, 2]),
                               "approach_mm": summ(ex["conditions"][:, 0]), "s_mm": summ(ex["conditions"][:, 3]),
                               "normal_force": summ(ex["conditions"][:, 5]),
                               "n_inside_pitch_ramp": int((np.abs(ex["conditions"][:, 3]) < 0.05).sum())}
    return out


# ------------------------------------------------------------------------------ comparison
def comparison() -> dict:
    sel = json.loads((EXT / "model_selection.json").read_text(encoding="utf-8"))
    d, ex = load(V4F), load("gear_mesh_quasistatic_v4_extra")
    x, y, traj, var = d["inputs"], d["targets"], d["trajectory_ids"], d["friction_variant"]
    split = grouped_trajectory_split(traj, 0.625, 0.1875, seed=42)
    test = np.asarray(sorted(int(i) for i in split.test))
    own = test[var[test] == 0]
    m27, m81 = x[own, 0] > 0.5, x[test, 0] > 0.5
    dx, dy = float(d["attrs"]["grid_dx_mm"]), float(d["attrs"]["grid_dy_mm"])
    mask_t = torch.from_numpy(x[own][:, 0:1])

    def eq_ratio(s):
        a = [float(equilibrium_residual(torch.from_numpy(s[i:i + 1]).float(), mask_t[i:i + 1], dx, dy)) for i in range(len(s))]
        b = [float(equilibrium_residual(torch.from_numpy(y[own][i:i + 1]).float(), mask_t[i:i + 1], dx, dy)) for i in range(len(s))]
        return np.asarray(a) / np.asarray(b)

    out, singles = {}, {}
    for fam, s in sel.items():
        cfg = s["selected"]
        if fam == "fno" and cfg.get("width") == 24 and cfg.get("modes") == 12:
            prefix = "fno"
        elif fam == "fno":
            prefix = f"fno_w{cfg['width']}_mx{cfg['modes']}_my{cfg['modes']}_l4"
        elif fam == "fno_physics":
            prefix = f"fno_physics_pw{cfg['physics_weight']:g}"
        elif fam == "unet":
            prefix = f"unet_w{cfg['width']}_mx12_my12_l4"
        elif fam == "pod_deeponet":
            prefix = f"pod_deeponet_w{cfg['width']}_mx12_my12_l4_bd{cfg['basis']}"
        else:
            prefix = f"deeponet_w{cfg['width']}_mx12_my12_l4_bd{cfg['basis']}"
        runs = run_dirs(V4F, prefix)
        if len(runs) < 10:
            print(f"{fam}: {len(runs)} final runs, skipped")
            continue
        p27, p81 = predict(runs, x[own]), predict(runs, x[test])
        ens27 = p27.mean(0)
        single = np.array([rel(p27[k], y[own], m27).mean() for k in range(len(runs))])
        singles[fam] = single
        res = [json.loads((r / "results.json").read_text()) for r in runs]
        out[fam] = {"selected": cfg, "prefix": prefix, "params": n_params(build(runs[0])[0]),
                    "ensemble_27": summ(rel(ens27, y[own], m27)), "ensemble_81": summ(rel(p81.mean(0), y[test], m81)),
                    "single_27_per_seed": single.tolist(), "single_27": summ(single),
                    "full_grid_logged_81_per_seed": [r["test"]["relative_l2"] for r in res],
                    "r2_27": r2(ens27, y[own], m27), "root_27": root_block(ens27, y[own], m27),
                    "equilibrium_ratio_27": summ(eq_ratio(ens27)), "equilibrium_ratio_27_by_case": eq_ratio(ens27).tolist(),
                    "gradient_ratio_27": summ(grad_rms(ens27, m27) / grad_rms(y[own], m27)),
                    "label_equilibrium_note": "residual of the prediction divided by that of the label, per case"}
        # the six additional trajectories: contact positions between the training positions, approach drawn per position
        pe = predict(runs, ex["inputs"])
        me = ex["inputs"][:, 0] > 0.5
        out[fam]["extra_54"] = {"ensemble": summ(rel(pe.mean(0), ex["targets"], me)),
                                "single_per_seed": [float(rel(pe[k], ex["targets"], me).mean()) for k in range(len(runs))]}
        print(f"{fam:12s} {prefix:36s} ens27 {out[fam]['ensemble_27']['mean']:.4f} single {single.mean():.4f} "
              f"root {100 * out[fam]['root_27']['mean_abs']:.2f}% eq ratio {out[fam]['equilibrium_ratio_27']['mean']:.3f}")
    if "fno" in singles:
        fams = [f for f in singles if f != "fno"]
        for f in fams:
            t, p = stats.ttest_rel(singles[f], singles["fno"])
            sd_d = float(np.std(singles[f] - singles["fno"], ddof=1))
            base = float(singles["fno"].mean())
            out[f]["paired_vs_fno"] = {"mean_relative_change": float(singles[f].mean() / singles["fno"].mean() - 1),
                                       "t": float(t), "p": float(p), "p_bonferroni": float(min(1.0, p * len(fams))),
                                       "family_size": len(fams),
                                       # detection limit at 80% power, as a fraction of the FNO's mean error
                                       "mde80_relative": mde_paired(sd_d, len(singles[f]), 0.05) / base,
                                       "mde80_relative_adjusted": mde_paired(sd_d, len(singles[f]), 0.05 / len(fams)) / base}
        # Physics-FNO keeps the default FNO (width 24, 12 modes) and varies only the penalty weight; compare it
        # also with the FNO of the same architecture, so that the penalty is not confounded with the width
        if "fno_physics" in out and out["fno"]["prefix"] != "fno":
            runs24 = run_dirs(V4F, "fno")
            if len(runs24) >= 10:
                p24 = predict(runs24, x[own])
                s24 = np.array([rel(p24[k], y[own], m27).mean() for k in range(len(runs24))])
                t, p = stats.ttest_rel(singles["fno_physics"], s24)
                sd24 = float(np.std(singles["fno_physics"] - s24, ddof=1))
                out["fno_physics"]["paired_vs_fno_same_width"] = {
                    "mde80_relative": mde_paired(sd24, len(s24), 0.05) / float(s24.mean()),
                    "reference_prefix": "fno", "reference_single_27": summ(s24),
                    "reference_ensemble_27": summ(rel(p24.mean(0), y[own], m27)),
                    "reference_equilibrium_ratio_27": summ(eq_ratio(p24.mean(0))),
                    "mean_relative_change": float(singles["fno_physics"].mean() / s24.mean() - 1), "t": float(t), "p": float(p)}
        # whether each selected setting lies on the edge of its screening grid
        grids = {}
        for fam, s in sel.items():
            keys = [k for k in s["selected"] if k != "model"]
            vals = {k: sorted({sc["config"][k] for sc in s["scores"]}) for k in keys}
            grids[fam] = {k: {"selected": s["selected"][k], "grid": v,
                              "at_edge": s["selected"][k] in (v[0], v[-1])} for k, v in vals.items()}
        out["selection_edges"] = grids
        # rank agreement between the logged full-grid metric and the material-pixel metric
        names = list(singles)
        full = [np.mean(out[f]["full_grid_logged_81_per_seed"]) for f in names]
        mat = [out[f]["single_27"]["mean"] for f in names]
        out["ranking"] = {"order_material_pixels": [names[i] for i in np.argsort(mat)],
                          "order_full_grid_logged": [names[i] for i in np.argsort(full)]}
    out["screening"] = sel
    return out


# ------------------------------------------------------------------------------ POD
class Ridge:
    """Ridge regression with an unpenalized intercept (closed form)."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, X, Y):
        self.xm, self.ym = X.mean(0), Y.mean(0)
        Xc = X - self.xm
        self.W = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(X.shape[1]), Xc.T @ (Y - self.ym))
        return self

    def predict(self, X):
        return (X - self.xm) @ self.W + self.ym


class Standardizer:
    def __init__(self, X):
        self.m, self.s = X.mean(0), X.std(0)
        self.s[self.s < 1e-12] = 1.0

    def transform(self, X):
        return (X - self.m) / self.s


def poly_features(X, degree: int):
    """All monomials of the columns of X up to the given degree, without the constant."""
    from itertools import combinations_with_replacement
    cols = [np.prod(X[:, list(c)], axis=1) for d in range(1, degree + 1)
            for c in combinations_with_replacement(range(X.shape[1]), d)]
    return np.stack(cols, axis=1)


def pod_study(name: str, cond_cols, ranks, alphas, degrees, split) -> dict:
    d = load(name)
    x, y, var = d["inputs"], d["targets"], d.get("friction_variant", np.zeros(len(d["inputs"]), int))
    m = x[:, 0] > 0.5
    tr, va, te = (np.asarray(sorted(int(i) for i in getattr(split, k))) for k in ("train", "val", "test"))
    te_own = te[var[te] == 0]
    flat = lambda idx: (y[idx] * m[idx][:, None]).reshape(len(idx), -1).astype(np.float64)
    ytr, yva, yte = flat(tr), flat(va), flat(te_own)
    mean = ytr.mean(0)
    _, sv, vt = np.linalg.svd(ytr - mean, full_matrices=False)
    c = d["conditions"][:, cond_cols].astype(np.float64)
    scaler = Standardizer(c[tr])

    def recon(coeff, r):
        return (mean + coeff @ vt[:r]).reshape(-1, *y.shape[1:])

    def err(pred_flat_fields, idx):
        return rel(pred_flat_fields, y[idx] * m[idx][:, None], m[idx])

    search, best = [], None
    for deg in degrees:
        f = lambda idx, deg=deg: poly_features(scaler.transform(c[idx]), deg)
        for r in ranks:
            coef_tr = (ytr - mean) @ vt[:r].T
            for a in alphas:
                reg = Ridge(alpha=a).fit(f(tr), coef_tr)
                v = float(err(recon(reg.predict(f(va)), r), va).mean())
                search.append({"degree": deg, "rank": r, "alpha": a, "val_relative_l2": v})
                if best is None or v < best["val_relative_l2"]:
                    best = search[-1]
    deg, r, a = best["degree"], best["rank"], best["alpha"]
    f = lambda idx: poly_features(scaler.transform(c[idx]), deg)
    reg = Ridge(alpha=a).fit(f(tr), (ytr - mean) @ vt[:r].T)
    test_err = err(recon(reg.predict(f(te_own)), r), te_own)
    proj_err = err(recon((yte - mean) @ vt[:r].T, r), te_own)                 # best fit within the retained basis
    proj_by_rank = {str(rr): float(err(recon((yte - mean) @ vt[:rr].T, rr), te_own).mean()) for rr in ranks}
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    out = {"conditions_used": cond_cols, "selected": best, "test_relative_l2": summ(test_err),
           "projection_error_selected_rank": summ(proj_err), "projection_error_by_rank": proj_by_rank,
           "energy_retained_by_rank": {str(rr): float(energy[min(rr, len(energy)) - 1]) for rr in ranks},
           "mean_field_error": summ(err(np.broadcast_to(mean.reshape(1, *y.shape[1:]), (len(te_own), *y.shape[1:])), te_own)),
           "search": search}
    # best purely linear map, for comparison with the earlier linear baseline
    lin = [s for s in search if s["degree"] == 1]
    out["best_linear"] = min(lin, key=lambda s: s["val_relative_l2"])
    b = out["best_linear"]
    reg1 = Ridge(alpha=b["alpha"]).fit(scaler.transform(c[tr]), (ytr - mean) @ vt[:b["rank"]].T)
    out["best_linear"]["test_relative_l2"] = summ(err(recon(reg1.predict(scaler.transform(c[te_own])), b["rank"]), te_own))
    return out


def pod() -> dict:
    d = load(V4F)
    s4 = grouped_trajectory_split(d["trajectory_ids"], 0.625, 0.1875, seed=42)
    v4f = pod_study(V4F, [0, 1, 2, 3], (5, 10, 20, 40, 80, 160, 269), (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0), (1, 2, 3), s4)
    print(f"POD v4f: selected {v4f['selected']}, test {v4f['test_relative_l2']['mean']:.4f}, projection "
          f"{v4f['projection_error_selected_rank']['mean']:.4f}, best linear {v4f['best_linear']['test_relative_l2']['mean']:.4f}")
    d2 = load("gear_pair_transient_quasistatic_v2")
    s2 = grouped_trajectory_split(d2["trajectory_ids"], 0.625, 0.1875, seed=42)
    v2 = pod_study("gear_pair_transient_quasistatic_v2", list(range(d2["conditions"].shape[1])), (5, 10, 20, 40, 89),
                   (0.01, 0.1, 1.0, 10.0, 100.0), (1,), s2)
    print(f"POD v2 (linear): selected {v2['selected']}, test {v2['test_relative_l2']['mean']:.4f} (median "
          f"{v2['test_relative_l2']['median']:.3f}), projection {v2['projection_error_selected_rank']['mean']:.4f}")
    return {"v4f": v4f, "v2_linear": v2}


# ------------------------------------------------------------------------------ distance from contact
def distance() -> dict:
    d = load(V4F)
    x, y, traj, var = d["inputs"], d["targets"], d["trajectory_ids"], d["friction_variant"]
    split = grouped_trajectory_split(traj, 0.625, 0.1875, seed=42)
    test = np.asarray(sorted(int(i) for i in split.test))
    own = test[var[test] == 0]
    sol = ROOT / "outputs" / "gear_mesh_v4_dataset"
    fe = EXT / "fe"
    m = x[own, 0] > 0.5
    with np.load(sol / "base_000_own.npz") as g:
        gx, gy = g["x"], g["y"]
    cp = np.array([json.loads((sol / f"base_{i:03d}_own.json").read_text())["contact_point_mm"] for i in own])
    dist = np.hypot(gx[None] - cp[:, 0, None, None], gy[None] - cp[:, 1, None, None])
    refs = {"labels": y[own]}
    for lev in ("L2", "L3"):
        files = [fe / f"{lev}_{i:03d}_own.npz" for i in own]
        if all(f.exists() for f in files):
            refs[lev] = np.stack([np.load(f)["stress"] for f in files]) * m[:, None]
    ens = predict(run_dirs(V4F, "fno"), x[own]).mean(0)
    pairs = {"surrogate_vs_labels": (ens, refs["labels"])}
    for lev in ("L2", "L3"):
        if lev in refs:
            pairs[f"labels_vs_{lev}"] = (refs["labels"], refs[lev])
            pairs[f"surrogate_vs_{lev}"] = (ens, refs[lev])
    if "L2" in refs and "L3" in refs:
        pairs["L2_vs_L3"] = (refs["L2"], refs["L3"])
    out = {"bins_mm": [float(b) for b in BINS_MM]}
    for name, (a, b) in pairs.items():
        e2 = ((a - b) ** 2).sum(1)
        blk = []
        for lo, hi in zip(BINS_MM[:-1], BINS_MM[1:]):
            sel = m & (dist >= lo) & (dist < hi)
            blk.append({"pixels_per_case": float(sel.sum() / len(own)),
                        "relative_l2": float(np.mean([np.sqrt(e2[j][sel[j]].sum() / (b[j] ** 2).sum(0)[sel[j]].sum())
                                                      for j in range(len(own))])),
                        "share_of_squared_error": float(np.mean([e2[j][sel[j]].sum() / e2[j][m[j]].sum() for j in range(len(own))]))})
        out[name] = blk
    return out


def main() -> None:
    EXT.mkdir(parents=True, exist_ok=True)
    todo = sys.argv[1:] or ["partitions", "comparison", "pod", "distance"]
    for section in todo:
        update(section, {"partitions": partitions, "comparison": comparison, "pod": pod, "distance": distance}[section]())


if __name__ == "__main__":
    main()

"""Held-out accuracy of the released surrogate, reproducing every headline
accuracy number.

Reads only checkpoints and per-seed logs already produced by ``train_smoke.py``
(10 seeds x 300 epochs per configuration). No training happens here.

What it computes, all on the leakage-safe held-out trajectories:

* v2, split_seed 42: single-seed (expected over 10 seeds), 10-seed ensemble, and
  ensemble plus void masking, for relative L2, R2 over material pixels (overall
  and per channel), hotspot MAE and the interior equilibrium residual;
* the share of squared error that lies in the void before masking;
* v2, split_seed 123: single seed against ensemble, the partition check;
* v1 (30 seeds) and v3 (10 seeds): the same single-versus-ensemble comparison,
  giving the bias- versus variance-dominated regime dependence;
* Physics-FNO's equilibrium residual, single seed and ensembled;
* peak root-fillet von Mises error for vanilla FNO and AEE-FNO under three
  estimators, with the signed bias;
* the data-scaling law fitted to the v2 training-size curve.

The public repository ships the twenty v2 checkpoints behind the headline
figures (vanilla FNO and AEE-FNO, ten seeds each). Blocks whose checkpoints
are not present, such as the v1/v3 regime rows, are recorded as unavailable
rather than failing; the committed JSON was produced with every checkpoint
present, and the remainder are available on reasonable request.

Usage:
    python scripts/evaluate_surrogate_accuracy.py

Writes outputs/surrogate/surrogate_accuracy.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_root_region_stress_error import ROOT_ROWS, build_model  # noqa: E402
from gearstress.losses import equilibrium_residual, hotspot_mae, relative_l2  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
OUT = ROOT / "outputs" / "surrogate" / "surrogate_accuracy.json"


def load_split(dataset: str, split_seed: int):
    with h5py.File(ROOT / "data" / "processed" / f"{dataset}.h5", "r") as f:
        x = np.asarray(f["inputs"], dtype=np.float32)
        y = np.asarray(f["targets"], dtype=np.float32)
        traj = np.asarray(f["trajectory_ids"])
        dx = float(f.attrs["grid_dx_mm"])
        dy = float(f.attrs["grid_dy_mm"])
    split = grouped_trajectory_split(
        traj, train_fraction=float(CFG["train_fraction"]),
        val_fraction=float(CFG["val_fraction"]), seed=split_seed)
    scale = max(float(np.max(np.abs(y[split.train]))), 1e-8)   # train-only, as in training
    return {
        "x": torch.from_numpy(x[split.test]),
        "y": torch.from_numpy(y[split.test] / scale),
        "dx": dx, "dy": dy,
        "n_test_cases": int(len(split.test)),
        "n_test_trajectories": int(len(set(traj[split.test].tolist()))),
    }


def seed_of(path: Path) -> int:
    return int(re.search(r"_seed(\d+)_ep300$", path.name).group(1))


def predict(dataset: str, run_prefix: str, model_name: str, data) -> torch.Tensor:
    runs = sorted((ROOT / "outputs" / "smoke" / dataset).glob(f"{run_prefix}_seed*_ep300"), key=seed_of)
    if not runs:
        raise FileNotFoundError(f"no {run_prefix} checkpoints under outputs/smoke/{dataset}")
    preds = []
    for run in runs:
        model = build_model(model_name, data["dx"], data["dy"])
        model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            preds.append(model(data["x"]))
    return torch.stack(preds)


def r2(pred: torch.Tensor, target: torch.Tensor, material: np.ndarray, channel=None) -> float:
    p, t = pred.numpy(), target.numpy()
    if channel is None:
        sel = np.broadcast_to(material[:, None], t.shape)
        p, t = p[sel], t[sel]
    else:
        p, t = p[:, channel][material], t[:, channel][material]
    return float(1.0 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def field_block(preds: torch.Tensor, data) -> dict:
    y = data["y"]
    mask4 = (data["x"][:, 0:1] > 0.5).float()
    material = mask4[:, 0].numpy() > 0.5
    ens = preds.mean(0)
    ens_masked = ens * mask4

    def eq(p):
        return float(equilibrium_residual(p, mask4, dx=data["dx"], dy=data["dy"]))

    sq = (ens - y).square().sum(1)
    void_share = float(sq[torch.from_numpy(~material)].sum() / sq.sum())
    return {
        "n_seeds": int(preds.shape[0]),
        "single_seed_mean": {
            "relative_l2": float(np.mean([float(relative_l2(p, y)) for p in preds])),
            "relative_l2_sd": float(np.std([float(relative_l2(p, y)) for p in preds], ddof=1)),
            "r2": float(np.mean([r2(p * mask4, y * mask4, material) for p in preds])),
            "hotspot_mae": float(np.mean([float(hotspot_mae(p, y)) for p in preds])),
            "equilibrium_residual": float(np.mean([eq(p) for p in preds])),
        },
        "ensemble": {
            "relative_l2": float(relative_l2(ens, y)),
            "r2": r2(ens * mask4, y * mask4, material),
            "hotspot_mae": float(hotspot_mae(ens, y)),
            "equilibrium_residual": eq(ens),
        },
        "ensemble_void_masked": {
            "relative_l2": float(relative_l2(ens_masked, y)),
            "r2": r2(ens_masked, y * mask4, material),
            "r2_per_channel": {n: r2(ens_masked, y * mask4, material, i)
                               for i, n in enumerate(["sxx", "syy", "sxy"])},
            "pearson_r": float(np.corrcoef(
                ens_masked.numpy()[np.broadcast_to(material[:, None], y.shape)],
                (y * mask4).numpy()[np.broadcast_to(material[:, None], y.shape)])[0, 1]),
            "hotspot_mae": float(hotspot_mae(ens_masked, y)),
            "equilibrium_residual": eq(ens_masked),
        },
        "void_share_of_squared_error_before_masking": void_share,
    }


def regime_block(preds: torch.Tensor, data) -> dict:
    y = data["y"]
    single = float(np.mean([float(relative_l2(p, y)) for p in preds]))
    ens = float(relative_l2(preds.mean(0), y))
    return {"n_seeds": int(preds.shape[0]), "single_seed_relative_l2": single,
            "ensemble_relative_l2": ens, "error_reduction": (single - ens) / single}


def root_peak_block(preds: torch.Tensor, data) -> dict:
    def vm(a):
        return np.sqrt(np.maximum(a[:, 0] ** 2 - a[:, 0] * a[:, 1] + a[:, 1] ** 2 + 3 * a[:, 2] ** 2, 0.0))

    material = data["x"][:, 0].numpy() > 0.5
    region = material & ROOT_ROWS[None, :, None]
    mask4 = torch.from_numpy(material.astype(np.float32))[:, None]
    true_peak = np.where(region, vm(data["y"].numpy()), -np.inf).max(axis=(1, 2))

    def peaks(a):
        return np.where(region, vm(a), -np.inf).max(axis=(1, 2))

    per_seed = [peaks(p.numpy()) for p in preds]
    field_ens = peaks((preds.mean(0) * mask4).numpy())
    peak_ens = np.mean(per_seed, axis=0)

    def err(pk):
        return float(np.mean(np.abs(pk - true_peak) / true_peak))

    def bias(pk):
        return float(np.mean((pk - true_peak) / true_peak))

    return {
        "single_seed_expected": {"abs_rel_error": float(np.mean([err(q) for q in per_seed])),
                                 "signed_bias": float(np.mean([bias(q) for q in per_seed]))},
        "ensemble_field_then_peak": {"abs_rel_error": err(field_ens), "signed_bias": bias(field_ens)},
        "ensemble_of_peak_values": {"abs_rel_error": err(peak_ens), "signed_bias": bias(peak_ens)},
    }


def scaling_law() -> dict:
    base = ROOT / "outputs" / "smoke" / "gear_pair_transient_quasistatic_v2"
    by_n: dict[int, list[float]] = {}
    for f in (base / "train_size_curve").glob("fno_*samples_seed*_ep300/results.json"):
        n = int(re.search(r"_(\d+)samples_", f.parent.name).group(1))
        by_n.setdefault(n, []).append(json.loads(f.read_text(encoding="utf-8"))["test"]["relative_l2"])
    full = [json.loads(f.read_text(encoding="utf-8"))["test"]["relative_l2"]
            for f in base.glob("fno_seed*_ep300/results.json")]
    by_n.setdefault(90, full)
    ns = np.array(sorted(by_n), dtype=float)
    es = np.array([np.mean(by_n[int(n)]) for n in ns])
    slope, intercept = np.polyfit(np.log(ns), np.log(es), 1)
    r2_fit = float(np.corrcoef(np.log(ns), np.log(es))[0, 1] ** 2)
    return {"points": {str(int(n)): {"mean_relative_l2": float(e), "n_seeds": len(by_n[int(n)])}
                       for n, e in zip(ns, es)},
            "coefficient": float(np.exp(intercept)), "exponent": float(slope),
            "r2_log_log": r2_fit, "fitted_range_n_train": [int(ns.min()), int(ns.max())],
            "note": "Valid over the fitted range only; not to be extrapolated as a result."}


UNAVAILABLE = {"status": "checkpoints not shipped in the public repository; available on reasonable request"}


def optional(fn, *args):
    """Run a block, or record it as unavailable if its checkpoints are absent."""
    try:
        return fn(*args)
    except FileNotFoundError:
        return UNAVAILABLE


def main() -> None:
    v2 = load_split("gear_pair_transient_quasistatic_v2", 42)
    v2_alt = load_split("gear_pair_transient_quasistatic_v2", 123)
    v1 = load_split("gear_pair_transient_quasistatic_v1", 42)
    v3 = load_split("gear_pair_transient_quasistatic_v3", 42)

    fno_v2 = predict("gear_pair_transient_quasistatic_v2", "fno", "fno", v2)
    aee_v2 = predict("gear_pair_transient_quasistatic_v2", "aee_fno", "aee_fno", v2)
    phys_v2 = optional(predict, "gear_pair_transient_quasistatic_v2", "fno_physics", "fno", v2)

    result = {
        "protocol": {
            "held_out": "complete trajectories, grouped_trajectory_split",
            "v2_test_cases": v2["n_test_cases"], "v2_test_trajectories": v2["n_test_trajectories"],
            "target_scale": "max |target| over the training partition only",
        },
        "v2_vanilla_fno": field_block(fno_v2, v2),
        "v2_physics_fno": UNAVAILABLE if isinstance(phys_v2, dict) else field_block(phys_v2, v2),
        "regime_dependence": {
            "v1_36_train": optional(lambda: regime_block(
                predict("gear_pair_transient_quasistatic_v1", "fno", "fno", v1), v1)),
            "v3_36_train_independent": optional(lambda: regime_block(
                predict("gear_pair_transient_quasistatic_v3", "fno", "fno", v3), v3)),
            "v2_90_train": regime_block(fno_v2, v2),
            "v2_90_train_alternate_partition": optional(lambda: regime_block(
                predict("gear_pair_transient_quasistatic_v2", "fno_splitseed123", "fno", v2_alt), v2_alt)),
        },
        "peak_root_fillet_von_mises": {
            "vanilla_fno": root_peak_block(fno_v2, v2),
            "aee_fno": root_peak_block(aee_v2, v2),
        },
        "scaling_law_v2": scaling_law(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    f = result["v2_vanilla_fno"]
    print(f"v2 relative L2   single {f['single_seed_mean']['relative_l2']:.4f}  "
          f"ensemble {f['ensemble']['relative_l2']:.4f}  "
          f"+mask {f['ensemble_void_masked']['relative_l2']:.4f}")
    print(f"v2 R2            single {f['single_seed_mean']['r2']:.4f}  "
          f"ensemble {f['ensemble']['r2']:.4f}  +mask {f['ensemble_void_masked']['r2']:.4f}")
    for k, v in result["regime_dependence"].items():
        if "status" in v:
            print(f"{k:34s} unavailable (checkpoints not shipped)")
            continue
        print(f"{k:34s} {v['single_seed_relative_l2']:.4f} -> {v['ensemble_relative_l2']:.4f}"
              f"  ({100 * v['error_reduction']:.1f}%, k={v['n_seeds']})")
    for m, b in result["peak_root_fillet_von_mises"].items():
        print(f"root peak {m:12s} " + "  ".join(
            f"{k}: {100 * v['abs_rel_error']:.2f}% (bias {100 * v['signed_bias']:+.2f}%)"
            for k, v in b.items()))
    s = result["scaling_law_v2"]
    print(f"scaling law  rel_L2 = {s['coefficient']:.2f} N^{s['exponent']:.2f}  R2={s['r2_log_log']:.2f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

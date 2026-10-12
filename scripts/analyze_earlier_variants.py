"""Exploratory comparison on the earlier datasets v1 and v2: the ten FNO variants, U-Net
and DeepONet against the vanilla FNO, re-evaluated from the stored checkpoints.

For every configuration and seeds 1-10 the test trajectories of the reference
partition (split seed 42) are predicted and scored with two metrics:

* full grid: relative L2 over all grid points and the three components, the
  prediction left unmasked (the metric logged by scripts/train_smoke.py; it is
  recomputed here and checked against the logged value);
* material: the same over the material pixels of the dataset's own mask, with
  the prediction set to zero outside it.

Each configuration is compared with its reference FNO (the default FNO, or the
FNO of matched width and modes for DCT-FNO) by a paired t-test over the ten
seeds. Bonferroni adjustment covers the family of the ten FNO variants; U-Net
and DeepONet form a separate family of four. Parameter counts count a complex
spectral weight once; frozen parts are reported separately.

Usage: python scripts/analyze_earlier_variants.py
Output: outputs/extended_study/earlier_variants.json
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

from gearstress.inventions import (  # noqa: E402
    AdaptiveEquilibriumExchangeFNO, CompatibleContactPotentialFNO, ContactCenteredMultiscaleFNO, DCTFNO,
    DualWindowMultiscaleFNO, HotspotGuidedMultiscaleFNO, LoadPathModulatedFNO, MinimumInterventionContactEquilibrium,
    ResidualCompatibleRepairFNO)
from gearstress.models import DeepONet, FNO2d, UNetSmall  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

torch.set_num_threads(1)
DATA = ROOT / "data" / "processed"
SMOKE = ROOT / "outputs" / "smoke"
OUT = ROOT / "outputs" / "extended_study" / "earlier_variants.json"
SEEDS = range(1, 11)
DATASETS = {"v1": "gear_pair_transient_quasistatic_v1", "v2": "gear_pair_transient_quasistatic_v2"}
BASE = dict(width=24, modes_x=12, modes_y=12, layers=4)

# name, run prefix, reference prefix, family, modified component, added loss or constraint, training of the base
CONFIGS = [
    ("AEE-FNO", "aee_fno", "fno", "variant",
     "Raw output head plus a per-case learned gate that mixes the raw stress with its spectral equilibrium "
     "projection, and an Airy-potential correction around the contact", "none (Equation (1), λp = 0)", "joint"),
    ("RCR-FNO", "rcr_fno", "fno", "variant",
     "Vanilla FNO of the same seed followed by a fixed removal of 50% of its non-equilibrium component and a trained "
     "residual operator (2 layers, width 12, 8 modes) that adds an Airy-potential repair", "none", "base frozen"),
    ("CCP-FNO", "ccp_fno", "fno", "variant",
     "Output replaced by its projection onto discrete equilibrium plus an Airy-potential correction around the contact",
     "hard equilibrium constraint on the output", "joint"),
    ("LPM-RCR-FNO", "lpm_rcr_fno", "fno", "variant",
     "LPM-FNO of the same seed followed by the RCR repair, removing 60% of the non-equilibrium component", "none",
     "base frozen"),
    ("MICE-LPM", "mice_lpm", "fno", "variant",
     "LPM-FNO of the same seed followed at inference by the smallest equilibrium projection, in a metric weighting "
     "contact and hotspot regions, that meets a residual budget of 100", "residual budget at inference",
     "base frozen, no training"),
    ("LPM-FNO", "lpm_fno", "fno", "variant",
     "Feature-wise gain and bias in every layer from the spatial means of the condition channels", "none", "joint"),
    ("DCT-FNO", "dct_fno_w32_mx16_my16_l4", "fno_w32_mx16_my16_l4", "variant",
     "Fourier transform of the spectral layers replaced by a cosine transform (width 32, 16 modes; compared with the "
     "FNO of the same width and modes)", "none", "joint"),
    ("CCM-FNO", "ccm_fno_lw16_lwin0.5_lsz16", "fno", "variant",
     "Untruncated convolutional branch (width 16) on a 16 × 16 window around the contact point, Gaussian-gated, plus "
     "two contact-relative coordinate channels", "none", "joint"),
    ("HGM-FNO", "hgm_fno_lw24_lwin0.25_lsz16", "fno", "variant",
     "As CCM-FNO (branch width 24), with the window centered at the soft-argmax of the predicted stress magnitude",
     "none", "joint"),
    ("DWM-FNO", "dwm_fno_lw8_lwin0.5_lsz16", "fno", "variant",
     "Both windows of CCM-FNO and HGM-FNO (branch width 8)", "none", "joint"),
    ("U-Net", "unet", "fno", "baseline", "Convolutional encoder-decoder, base width 24", "none", "joint"),
    ("U-Net (matched size)", "unet_w38_mx12_my12_l4", "fno", "baseline", "As above, base width 38", "none", "joint"),
    ("DeepONet", "deeponet_w32_mx12_my12_l4_bd32", "fno", "baseline",
     "Convolutional branch and coordinate trunk, width 32, basis 32", "none", "joint"),
    ("DeepONet (matched size)", "deeponet_w98_mx12_my12_l4_bd128", "fno", "baseline",
     "As above, width 98, basis 128", "none", "joint"),
]


def build(prefix: str, dx: float, dy: float, seed: int, dataset: str):
    """Model of one stored run, with its weights."""
    run = SMOKE / dataset / f"{prefix}_seed{seed}_ep300"
    if prefix == "fno":
        m = FNO2d(**BASE)
    elif prefix == "fno_w32_mx16_my16_l4":
        m = FNO2d(width=32, modes_x=16, modes_y=16, layers=4)
    elif prefix == "aee_fno":
        m = AdaptiveEquilibriumExchangeFNO(**BASE, dx=dx, dy=dy)
    elif prefix == "ccp_fno":
        m = CompatibleContactPotentialFNO(**BASE, dx=dx, dy=dy)
    elif prefix == "rcr_fno":
        m = ResidualCompatibleRepairFNO(FNO2d(**BASE), dx=dx, dy=dy, repair_fraction=0.5)
    elif prefix == "lpm_fno":
        m = LoadPathModulatedFNO(**BASE)
    elif prefix == "lpm_rcr_fno":
        m = ResidualCompatibleRepairFNO(LoadPathModulatedFNO(**BASE), dx=dx, dy=dy, repair_fraction=0.6)
    elif prefix == "mice_lpm":
        m = build("lpm_fno", dx, dy, seed, dataset)
        r = json.loads((run / "results.json").read_text(encoding="utf-8"))
        mice = MinimumInterventionContactEquilibrium(dx, dy, equilibrium_budget=r["equilibrium_budget"],
                                                     contact_weight=r["contact_weight"], hotspot_weight=r["hotspot_weight"])
        return ("mice", m, mice)
    elif prefix.startswith("dct_fno"):
        m = DCTFNO(width=32, modes_x=16, modes_y=16, layers=4, grid_h=64, grid_w=64)
    elif prefix.startswith("ccm_fno"):
        m = ContactCenteredMultiscaleFNO(**BASE, local_width=16, local_window=0.5, local_size=16, gate_sigma_fraction=0.25)
    elif prefix.startswith("hgm_fno"):
        m = HotspotGuidedMultiscaleFNO(**BASE, local_width=24, local_window=0.25, local_size=16, gate_sigma_fraction=0.25,
                                       softmax_temperature=0.1)
    elif prefix.startswith("dwm_fno"):
        m = DualWindowMultiscaleFNO(**BASE, local_width=8, local_window=0.5, local_size=16, gate_sigma_fraction=0.25,
                                    softmax_temperature=0.1)
    elif prefix == "unet":
        m = UNetSmall(width=24)
    elif prefix.startswith("unet_w"):
        m = UNetSmall(width=int(prefix.split("_w")[1].split("_")[0]))
    elif prefix.startswith("deeponet"):
        w = int(prefix.split("_w")[1].split("_")[0])
        m = DeepONet(width=w, basis_dim=int(prefix.split("_bd")[1]))
    else:
        raise ValueError(prefix)
    m.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
    m.eval()
    return m


def count(m) -> dict:
    """Parameters with a complex weight counted once, and the trainable part (frozen bases excluded)."""
    if isinstance(m, tuple):
        m = m[1]
    tot = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    real = sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())
    return {"total": int(tot), "trainable": int(train), "real_numbers": int(real)}


def predict(m, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        if isinstance(m, tuple):
            _, lpm, mice = m
            return mice(lpm(x), x[:, :1], x[:, 7:8]).numpy()
        return m(x).numpy()


def rel(p: np.ndarray, t: np.ndarray, m: np.ndarray | None) -> np.ndarray:
    if m is not None:
        p, t = p * m[:, None], t * m[:, None]
    return np.sqrt(((p - t) ** 2).reshape(len(p), -1).sum(1) / (t ** 2).reshape(len(t), -1).sum(1))


def main() -> None:
    out = {"metric_note": "relative L2 per test case, mean over the test cases; full grid unmasked or material pixels "
                          "of the dataset's own mask", "datasets": {}}
    for tag, ds in DATASETS.items():
        with h5py.File(DATA / f"{ds}.h5", "r") as f:
            x = np.asarray(f["inputs"], dtype=np.float32)
            y = np.asarray(f["targets"], dtype=np.float32)
            traj = np.asarray(f["trajectory_ids"])
            dx, dy = float(f.attrs["grid_dx_mm"]), float(f.attrs["grid_dy_mm"])
        split = grouped_trajectory_split(traj, train_fraction=0.625, val_fraction=0.1875, seed=42)
        test = np.asarray(sorted(int(i) for i in split.test))
        scale = max(float(np.max(np.abs(y[split.train]))), 1e-8)
        xt, yt = torch.from_numpy(x[test]), y[test] / scale
        mat = x[test, 0] > 0.5
        per = {}
        prefixes = sorted({c[1] for c in CONFIGS} | {c[2] for c in CONFIGS})
        for prefix in prefixes:
            full, matl, logged = [], [], []
            for s in SEEDS:
                m = build(prefix, dx, dy, s, ds)
                p = predict(m, xt)
                full.append(float(rel(p, yt, None).mean()))
                matl.append(float(rel(p, yt, mat).mean()))
                logged.append(json.loads((SMOKE / ds / f"{prefix}_seed{s}_ep300" / "results.json").read_text())["test"]["relative_l2"])
            per[prefix] = {"full_grid": full, "material": matl, "logged_full_grid": logged, "params": count(m),
                           "max_abs_recomputed_minus_logged": float(np.max(np.abs(np.array(full) - np.array(logged))))}
            print(f"{tag} {prefix:34s} full {np.mean(full):.4f} (logged {np.mean(logged):.4f}) material {np.mean(matl):.4f}")
        rows = []
        for name, prefix, ref, fam, *_ in CONFIGS:
            r = {"name": name, "prefix": prefix, "reference": ref, "family": fam}
            for metric in ("logged_full_grid", "material"):
                a, b = np.array(per[prefix][metric]), np.array(per[ref][metric])
                t, pv = stats.ttest_rel(a, b)
                r[metric] = {"mean": float(a.mean()), "reference_mean": float(b.mean()),
                             "relative_change": float(a.mean() / b.mean() - 1), "t": float(t), "p": float(pv)}
            rows.append(r)
        for fam in ("variant", "baseline"):
            sel = [r for r in rows if r["family"] == fam]
            for metric in ("logged_full_grid", "material"):
                for r in sel:
                    r[metric]["p_bonferroni"] = float(min(1.0, r[metric]["p"] * len(sel)))
                    r[metric]["family_size"] = len(sel)
        rank = {metric: [r["name"] for r in sorted(rows, key=lambda r: r[metric]["mean"])] for metric in ("logged_full_grid", "material")}
        out["datasets"][tag] = {"per_config": per, "comparison": rows, "ranking": rank,
                                "n_test_cases": int(len(test)), "material_pixels_per_case": float(mat.reshape(len(test), -1).sum(1).mean())}
    out["specification"] = [{"name": c[0], "prefix": c[1], "reference": c[2], "family": c[3], "modified_component": c[4],
                             "added_loss_or_constraint": c[5], "training": c[6],
                             "params": out["datasets"]["v2"]["per_config"][c[1]]["params"]} for c in CONFIGS]
    out["fno_params"] = out["datasets"]["v2"]["per_config"]["fno"]["params"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

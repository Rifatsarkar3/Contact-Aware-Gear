"""Training-set-size dose-response curve, within v2, holding val/test fixed.

`docs/DATASET_SCALE_STUDY_2026-07-24.md` compared exactly two points -- v1's
36 training samples and v2's 90 -- and found AEE-FNO's and DWM-FNO's field
accuracy regressions only become statistically detectable at the larger
size. That is a single before/after jump, not a curve: it does not show
*where*, between 36 and 90 samples, each mechanism's true status becomes
visible. This script fills in the gap using only v2's already-solved FEM
cases (no new FEM data): it takes prefixes of v2's own 10 training
trajectories (4, 5, 7, 8, 10 trajectories -> 36, 45, 63, 72, 90 samples),
holding the val/test split exactly fixed throughout, and retrains vanilla
FNO, AEE-FNO, and DWM-FNO at 10 seeds per size.

Subsampling is nested and nonrandom by design: the same trajectory-ordering
permutation `grouped_trajectory_split` (seed=42) uses internally is
reproduced here, and each smaller training set is a strict prefix of every
larger one, so N=45's training set is N=36's plus one more trajectory, not
an independently redrawn subset -- isolating training-set size as the only
thing that changes between points.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from statistics import mean, stdev

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.inventions import AdaptiveEquilibriumExchangeFNO, DualWindowMultiscaleFNO  # noqa: E402
from gearstress.losses import combined_loss, equilibrium_residual, hotspot_mae, relative_l2  # noqa: E402
from gearstress.models import FNO2d  # noqa: E402

DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v2.h5"
CONFIG = ROOT / "configs" / "smoke.yaml"
OUT_ROOT = ROOT / "outputs" / "smoke" / DATA.stem / "train_size_curve"

SPLIT_SEED = 42
TRAIN_FRACTION = 0.625
VAL_FRACTION = 0.1875
TRAJECTORY_COUNTS = [4, 5, 7, 8, 10]  # -> 36, 45, 63, 72, 90 samples (9 phases/trajectory)
SEEDS = list(range(1, 11))
EPOCHS = 300
MODEL_KEYS = ["fno", "aee_fno", "dwm_fno"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ordered_groups(trajectory_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce grouped_trajectory_split's internal permutation exactly."""
    groups = np.unique(trajectory_ids)
    rng = np.random.default_rng(SPLIT_SEED)
    groups = rng.permutation(groups)
    n_train = max(1, int(np.floor(groups.size * TRAIN_FRACTION)))
    n_val = max(1, int(np.floor(groups.size * VAL_FRACTION)))
    return groups[:n_train], groups[n_train : n_train + n_val], groups[n_train + n_val :]


@torch.no_grad()
def evaluate(model, loader, device, dx, dy) -> dict[str, float]:
    model.eval()
    rel, hot, eq, count = 0.0, 0.0, 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        prediction = model(inputs)
        batch = inputs.shape[0]
        rel += float(relative_l2(prediction, targets)) * batch
        hot += float(hotspot_mae(prediction, targets)) * batch
        eq += float(equilibrium_residual(prediction, inputs[:, :1], dx=dx, dy=dy)) * batch
        count += batch
    return {"relative_l2": rel / count, "hotspot_mae": hot / count, "equilibrium_residual": eq / count}


def build_model(key: str, model_cfg: dict, dx: float, dy: float):
    if key == "fno":
        return FNO2d(width=model_cfg["width"], modes_x=model_cfg["modes_x"], modes_y=model_cfg["modes_y"], layers=model_cfg["layers"])
    if key == "aee_fno":
        return AdaptiveEquilibriumExchangeFNO(
            width=model_cfg["width"], modes_x=model_cfg["modes_x"], modes_y=model_cfg["modes_y"],
            layers=model_cfg["layers"], dx=dx, dy=dy,
        )
    if key == "dwm_fno":
        return DualWindowMultiscaleFNO(
            width=model_cfg["width"], modes_x=model_cfg["modes_x"], modes_y=model_cfg["modes_y"],
            layers=model_cfg["layers"], local_width=8, local_window=0.5, local_size=16,
            gate_sigma_fraction=0.25, softmax_temperature=0.1,
        )
    raise ValueError(key)


def run_one(inputs_all, targets_raw, train_idx, val_idx, test_idx, model_key, seed, cfg, grid_dx, grid_dy, n_train) -> Path:
    run_name = f"{model_key}_{n_train}samples_seed{seed}_ep{EPOCHS}"
    out_dir = OUT_ROOT / run_name
    results_path = out_dir / "results.json"
    if results_path.exists():
        print(f"[skip] {run_name}")
        return results_path

    seed_all(seed)
    target_scale = max(float(np.max(np.abs(targets_raw[train_idx]))), 1e-8)
    targets = targets_raw / target_scale

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(inputs_all[indices]), torch.from_numpy(targets[indices]))
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=shuffle, generator=generator)

    train_loader = loader(train_idx, True)
    val_loader = loader(val_idx, False)
    test_loader = loader(test_idx, False)

    model_cfg = dict(cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_key, model_cfg, grid_dx, grid_dy).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    best_state = None
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            prediction = model(batch_inputs)
            loss, _ = combined_loss(
                prediction, batch_targets, batch_inputs[:, :1],
                hotspot_weight=float(cfg["loss"]["hotspot_weight"]), physics_weight=0.0,
                hotspot_quantile=float(cfg["loss"]["hotspot_quantile"]), dx=grid_dx, dy=grid_dy,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val = evaluate(model, val_loader, device, grid_dx, grid_dy)
        if val["relative_l2"] < best_val:
            best_val = val["relative_l2"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, grid_dx, grid_dy)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_key, "seed": seed, "n_train": n_train, "epochs": EPOCHS,
        "best_val_relative_l2": best_val, "test": test,
    }
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] {run_name}: {test}")
    return results_path


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    m = mean(diffs)
    s = stdev(diffs) if n > 1 else 0.0
    se = s / (n**0.5) if n > 1 else 0.0
    t = m / se if se > 0 else float("inf")
    return {"mean_diff": m, "sd_diff": s, "t": t, "n": n}


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    with h5py.File(DATA, "r") as handle:
        inputs_all = np.asarray(handle["inputs"], dtype=np.float32)
        targets_raw = np.asarray(handle["targets"], dtype=np.float32)
        trajectory_ids = np.asarray(handle["trajectory_ids"])
        grid_dx = float(handle.attrs.get("grid_dx_mm", 1.0))
        grid_dy = float(handle.attrs.get("grid_dy_mm", 1.0))

    train_groups_ordered, val_groups, test_groups = ordered_groups(trajectory_ids)
    val_idx = np.flatnonzero(np.isin(trajectory_ids, val_groups))
    test_idx = np.flatnonzero(np.isin(trajectory_ids, test_groups))

    summary: dict = {}
    for k in TRAJECTORY_COUNTS:
        train_groups_k = train_groups_ordered[:k]
        train_idx = np.flatnonzero(np.isin(trajectory_ids, train_groups_k))
        n_train = int(train_idx.size)
        runs_by_model: dict[str, dict[int, dict]] = {}
        for model_key in MODEL_KEYS:
            by_seed = {}
            for seed in SEEDS:
                path = run_one(inputs_all, targets_raw, train_idx, val_idx, test_idx, model_key, seed, cfg, grid_dx, grid_dy, n_train)
                data = json.loads(path.read_text(encoding="utf-8"))
                by_seed[seed] = data["test"]
            runs_by_model[model_key] = by_seed

        baseline = runs_by_model["fno"]
        point = {"n_train": n_train, "n_train_trajectories": int(k)}
        for model_key in MODEL_KEYS:
            by_seed = runs_by_model[model_key]
            entry = {
                "relative_l2_mean": mean(v["relative_l2"] for v in by_seed.values()),
                "relative_l2_std": stdev(v["relative_l2"] for v in by_seed.values()),
                "hotspot_mae_mean": mean(v["hotspot_mae"] for v in by_seed.values()),
                "hotspot_mae_std": stdev(v["hotspot_mae"] for v in by_seed.values()),
            }
            if model_key != "fno":
                diffs_l2 = [by_seed[s]["relative_l2"] - baseline[s]["relative_l2"] for s in SEEDS]
                diffs_hot = [by_seed[s]["hotspot_mae"] - baseline[s]["hotspot_mae"] for s in SEEDS]
                entry["paired_vs_fno"] = {
                    "relative_l2": paired_stats(diffs_l2),
                    "hotspot_mae": paired_stats(diffs_hot),
                }
            point[model_key] = entry
        summary[str(n_train)] = point
        print(f"=== N={n_train} complete ===")

    out_path = OUT_ROOT / "training_size_curve_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary written] {out_path}")
    for n_train, point in summary.items():
        for model_key in ("aee_fno", "dwm_fno"):
            p = point[model_key]["paired_vs_fno"]["relative_l2"]
            print(f"N={n_train} {model_key}: rel_l2 diff={p['mean_diff']:+.4f} t={p['t']:+.2f}")


if __name__ == "__main__":
    main()

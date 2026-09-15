"""FNO/AEE-FNO/DWM-FNO comparison on active-learning-selected vs. randomly-selected data.

**Do NOT run until GPU is free and the user gives an explicit go-ahead**
-- see docs/superpowers/specs/2026-08-04-al-misranking-combined-study-design.md.

Both `data/processed/gear_pair_al_pool_based_extended.h5` and
`data/processed/gear_pair_al_random_baseline_extended.h5` must exist
first (run scripts/reconstruct_al_dataset.py for each arm). Both share
v1's original, fixed val/test split -- only the training set differs
(actively-selected trajectories vs. randomly-selected ones, same count).

Trains FNO, AEE-FNO, DWM-FNO x 10 seeds on each dataset (same
AdamW/300-epoch/best-val-checkpoint protocol as every other model
comparison in this project; AEE-FNO/DWM-FNO reuse their already-selected
v1 leakage-safe-search hyperparameters, no new architecture search).
Reports, per dataset: relative L2 (the accuracy angle) and each
mechanism's paired-vs-FNO |t| (the reliability angle) -- directly
answering whether actively-selected data both lowers error and makes
the misranking-relevant regression easier to detect at the same N.

Usage (once launched):
    python scripts/run_al_misranking_comparison.py
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
from gearstress.splits import grouped_trajectory_split  # noqa: E402

V1_DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v1.h5"
CONFIG = ROOT / "configs" / "smoke.yaml"
DATASETS = {
    "pool_based": ROOT / "data" / "processed" / "gear_pair_al_pool_based_extended.h5",
    "random_baseline": ROOT / "data" / "processed" / "gear_pair_al_random_baseline_extended.h5",
    "ensemble_surrogate": ROOT / "data" / "processed" / "gear_pair_al_ensemble_surrogate_extended.h5",
}
OUT_ROOT = ROOT / "outputs" / "smoke" / "al_misranking_comparison"

SEEDS = list(range(1, 11))
EPOCHS = 300
MODEL_KEYS = ["fno", "aee_fno", "dwm_fno"]

# Hard cap so this can safely coexist with another GPU job -- same rationale
# and value as scripts/run_active_learning_study.py.
GPU_MEMORY_FRACTION = 0.20


def cap_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def v1_fixed_split_trajectory_ids() -> dict[str, set[int]]:
    with h5py.File(V1_DATA, "r") as handle:
        trajectory_ids = np.asarray(handle["trajectory_ids"])
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    split = grouped_trajectory_split(
        trajectory_ids, train_fraction=float(cfg["train_fraction"]), val_fraction=float(cfg["val_fraction"]), seed=int(cfg["seed"]),
    )
    return {
        "train": set(int(t) for t in trajectory_ids[split.train].tolist()),
        "val": set(int(t) for t in trajectory_ids[split.val].tolist()),
        "test": set(int(t) for t in trajectory_ids[split.test].tolist()),
    }


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


def run_one(inputs_all, targets_raw, train_idx, val_idx, test_idx, model_key, seed, cfg, grid_dx, grid_dy, dataset_key: str) -> Path:
    run_name = f"{model_key}_seed{seed}_ep{EPOCHS}"
    out_dir = OUT_ROOT / dataset_key / run_name
    results_path = out_dir / "results.json"
    if results_path.exists():
        print(f"[skip] {dataset_key}/{run_name}")
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
    for _ in range(EPOCHS):
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
    result = {"model": model_key, "seed": seed, "dataset": dataset_key, "epochs": EPOCHS, "best_val_relative_l2": best_val, "test": test}
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] {dataset_key}/{run_name}: {test}")
    return results_path


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    m = mean(diffs)
    s = stdev(diffs) if n > 1 else 0.0
    se = s / (n**0.5) if n > 1 else 0.0
    t = m / se if se > 0 else float("inf")
    return {"mean_diff": m, "sd_diff": s, "t": t, "n": n}


def main() -> None:
    cap_gpu_memory()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    v1_split_ids = v1_fixed_split_trajectory_ids()

    summary: dict = {}
    for dataset_key, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            raise FileNotFoundError(f"{dataset_path} missing -- run scripts/reconstruct_al_dataset.py --arm {dataset_key} first")
        with h5py.File(dataset_path, "r") as handle:
            inputs_all = np.asarray(handle["inputs"], dtype=np.float32)
            targets_raw = np.asarray(handle["targets"], dtype=np.float32)
            trajectory_ids = np.asarray(handle["trajectory_ids"])
            grid_dx = float(handle.attrs["grid_dx_mm"])
            grid_dy = float(handle.attrs["grid_dy_mm"])

        train_id_set = v1_split_ids["train"] | (set(int(t) for t in trajectory_ids.tolist()) - v1_split_ids["train"] - v1_split_ids["val"] - v1_split_ids["test"])
        val_idx = np.flatnonzero(np.isin(trajectory_ids, list(v1_split_ids["val"])))
        test_idx = np.flatnonzero(np.isin(trajectory_ids, list(v1_split_ids["test"])))
        train_idx = np.flatnonzero(np.isin(trajectory_ids, list(train_id_set)))

        runs_by_model: dict[str, dict[int, dict]] = {}
        for model_key in MODEL_KEYS:
            by_seed = {}
            for seed in SEEDS:
                path = run_one(inputs_all, targets_raw, train_idx, val_idx, test_idx, model_key, seed, cfg, grid_dx, grid_dy, dataset_key)
                data = json.loads(path.read_text(encoding="utf-8"))
                by_seed[seed] = data["test"]
            runs_by_model[model_key] = by_seed

        baseline = runs_by_model["fno"]
        point = {"n_train": int(train_idx.size)}
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
                entry["paired_vs_fno"] = {"relative_l2": paired_stats(diffs_l2), "hotspot_mae": paired_stats(diffs_hot)}
            point[model_key] = entry
        summary[dataset_key] = point

    out_path = OUT_ROOT / "al_misranking_comparison_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[summary written] {out_path}")
    for dataset_key, point in summary.items():
        print(f"\n== {dataset_key} (N={point['n_train']}) ==")
        for model_key in ("aee_fno", "dwm_fno"):
            p = point[model_key]["paired_vs_fno"]["relative_l2"]
            print(f"  {model_key}: rel_l2 diff={p['mean_diff']:+.4f} t={p['t']:+.2f}")


if __name__ == "__main__":
    main()

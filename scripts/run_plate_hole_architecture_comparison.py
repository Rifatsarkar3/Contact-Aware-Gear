"""Second-domain replication: does vanilla FNO vs. AEE-FNO misrank at small
sample size on the plate-with-elliptical-hole problem the same way it did on
gear-tooth contact (Table 2, Section 5.4 of the manuscript)?

AEE-FNO's mechanism (a spectral equilibrium projection plus a local
Airy-stress-function correction) is general 2D-elastostatics machinery, not
gear-specific -- see src/gearstress/inventions.py's AdaptiveEquilibriumExchangeFNO
docstring and Eq. (1) of the manuscript, which is exactly the equilibrium
condition it projects toward. It is reused here completely unmodified;
only the input data (built by generate_plate_hole_dataset.py) differs.

Deliberately self-contained rather than routed through scripts/train_smoke.py,
which carries many gear-specific defaults (RCR-FNO checkpoint chaining,
CCM/HGM local-window arguments, a 10-epoch smoke cap, configs/smoke.yaml
coupling) that don't apply here and would be easy to misapply by mistake.

Same protocol logic as the main study, scaled to this problem's size: a
fixed train/val/test split (same partition for every seed and every
architecture, so only model-init/training stochasticity varies across
seeds), 10 seeds per (architecture, dataset-size) cell, paired comparison
on test relative L2.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.inventions import AdaptiveEquilibriumExchangeFNO  # noqa: E402
from gearstress.losses import equilibrium_residual, relative_l2  # noqa: E402
from gearstress.models import FNO2d  # noqa: E402

WIDTH, MODES, LAYERS = 24, 12, 4
LEARNING_RATE, WEIGHT_DECAY = 1e-3, 1e-4
BATCH_SIZE = 4
EPOCHS = 200
SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = max(1, int(round(n * SPLIT_FRACTIONS[0])))
    n_val = max(1, int(round(n * SPLIT_FRACTIONS[1])))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def build_model(name: str, dx: float, dy: float) -> torch.nn.Module:
    if name == "fno":
        return FNO2d(width=WIDTH, modes_x=MODES, modes_y=MODES, layers=LAYERS)
    if name == "aee_fno":
        return AdaptiveEquilibriumExchangeFNO(width=WIDTH, modes_x=MODES, modes_y=MODES, layers=LAYERS, dx=dx, dy=dy)
    raise ValueError(name)


@torch.no_grad()
def eval_relative_l2(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    total, count = 0.0, 0
    for inputs, targets in loader:
        pred = model(inputs)
        total += float(relative_l2(pred, targets)) * inputs.shape[0]
        count += inputs.shape[0]
    return total / count


def run_one(data_path: Path, model_name: str, seed: int) -> dict:
    with h5py.File(data_path, "r") as handle:
        inputs = np.asarray(handle["inputs"], dtype=np.float32)
        targets = np.asarray(handle["targets"], dtype=np.float32)
        window_mm = float(handle.attrs["window_mm"])
    grid_size = inputs.shape[-1]
    dx = dy = 2.0 * window_mm / (grid_size - 1)

    train_idx, val_idx, test_idx = random_split(inputs.shape[0], SPLIT_SEED)
    scale = max(float(np.max(np.abs(targets[train_idx]))), 1e-8)
    targets = targets / scale

    seed_all(seed)

    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(inputs[idx]), torch.from_numpy(targets[idx]))
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=gen)

    train_loader = loader(train_idx, True)
    val_loader = loader(val_idx, False)
    test_loader = loader(test_idx, False)

    model = build_model(model_name, dx, dy)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None
    for _ in range(EPOCHS):
        model.train()
        for inputs_b, targets_b in train_loader:
            optimizer.zero_grad()
            pred = model(inputs_b)
            loss = relative_l2(pred, targets_b)
            loss.backward()
            optimizer.step()
        val = eval_relative_l2(model, val_loader)
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_rel_l2 = eval_relative_l2(model, test_loader)

    eq_total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for inputs_b, _ in test_loader:
            pred = model(inputs_b)
            eq_total += float(equilibrium_residual(pred, inputs_b[:, :1], dx=dx, dy=dy)) * inputs_b.shape[0]
            count += inputs_b.shape[0]

    return {
        "model": model_name,
        "seed": seed,
        "n_cases": int(inputs.shape[0]),
        "best_val_relative_l2": best_val,
        "test_relative_l2": test_rel_l2,
        "test_equilibrium_residual": eq_total / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for model_name in ("fno", "aee_fno"):
        for seed in range(args.seeds):
            result = run_one(args.data, model_name, seed)
            results.append(result)
            print(json.dumps(result))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

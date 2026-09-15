"""Uncertainty-guided active learning for gear-contact-stress FEM data acquisition.

Candidate trajectories are scored by predicted informativeness before any
FEM solve happens, since in this pipeline a case's spatial input and its
label are produced by the same solve -- there is no cheap "unlabeled
input" step. Scoring therefore operates on the scalar, per-trajectory
condition parameters (known before solving), not the spatial fields
(known only after).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gearstress.fem import project_case_to_grid

INDENT_RANGE = (0.018, 0.030)
FRICTION_RANGE = (0.04, 0.12)
YOUNGS_RANGE = (195_000.0, 215_000.0)
SPEED_RANGE = (1500.0, 4500.0)


def sample_candidate_conditions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n candidate (indentation, friction, youngs, speed) trajectory-level tuples."""
    indent = rng.uniform(*INDENT_RANGE, size=n)
    friction = rng.uniform(*FRICTION_RANGE, size=n)
    youngs = rng.uniform(*YOUNGS_RANGE, size=n)
    speed = rng.uniform(*SPEED_RANGE, size=n)
    return np.stack([indent, friction, youngs, speed], axis=1).astype(np.float64)


def normalize_conditions(conditions: np.ndarray) -> np.ndarray:
    """Map (indentation, friction, youngs, speed) to [0, 1]^4 using the fixed sampling ranges."""
    lo = np.array([INDENT_RANGE[0], FRICTION_RANGE[0], YOUNGS_RANGE[0], SPEED_RANGE[0]])
    hi = np.array([INDENT_RANGE[1], FRICTION_RANGE[1], YOUNGS_RANGE[1], SPEED_RANGE[1]])
    return ((conditions - lo) / (hi - lo)).astype(np.float32)


class Surrogate(nn.Module):
    """Small MLP mapping a normalized 4-dim condition vector to predicted model disagreement.

    Stands in for the expensive spatial ensemble when scoring candidates
    that have not been solved via FEM yet.
    """

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_surrogate(
    model: Surrogate,
    normalized_conditions: np.ndarray,
    disagreement_targets: np.ndarray,
    epochs: int = 300,
    lr: float = 0.01,
) -> None:
    """Fit the surrogate on (condition, disagreement) pairs from already-solved cases."""
    x = torch.from_numpy(normalized_conditions).float()
    y = torch.from_numpy(disagreement_targets).float().unsqueeze(-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()


def ensemble_disagreement(predictions: list[torch.Tensor]) -> torch.Tensor:
    """Per-sample epistemic-uncertainty proxy: variance across ensemble-member predictions.

    predictions: list of (batch, channels, H, W) tensors, one per ensemble member.
    Returns: (batch,) tensor, mean predictive variance per sample.
    """
    stacked = torch.stack(predictions, dim=0)  # (n_members, batch, C, H, W)
    variance = stacked.var(dim=0, unbiased=True)  # (batch, C, H, W)
    return variance.mean(dim=(1, 2, 3))


@torch.no_grad()
def mc_dropout_disagreement(model: nn.Module, inputs: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Per-sample epistemic-uncertainty proxy via repeated stochastic forward passes.

    model must already be in .train() mode so its dropout layers are active;
    this function does not toggle mode, since callers may want to control
    that explicitly around batchnorm-free architectures like FNO2d.
    """
    predictions = [model(inputs) for _ in range(n_samples)]
    return ensemble_disagreement(predictions)


def select_top_k(scores: np.ndarray, k: int) -> list[int]:
    """Return the indices of the k highest-scoring candidates."""
    order = np.argsort(scores)[::-1]
    return order[:k].tolist()


def grid_solved_frame(
    case_dir: Path,
    grid_size: int,
    indentation: float,
    friction: float,
    youngs: float,
    contact_fraction: float,
    directed_phase: float,
    time_s: float = float("nan"),
    mean_speed_rpm: float = float("nan"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn an already-solved CalculiX case directory into (input, target, condition).

    Pure post-processing -- does not invoke CalculiX. ``case_dir`` must
    already contain a completed solve (``case.json``, ``mesh_data.npz``,
    ``case.frd``). ``indentation``/``friction``/``youngs``/``contact_fraction``
    are known before solving in this pipeline and are also recoverable from
    ``case.json`` after the fact; ``time_s``/``mean_speed_rpm`` are per-
    trajectory scheduling metadata that are not persisted to disk, so
    callers reconstructing datasets from disk alone (rather than from a
    live solve) should leave them as NaN -- neither feeds into the
    returned input features or target.
    """
    grid = project_case_to_grid(case_dir, grid_size)
    x_span = max(float(np.ptp(grid["x"])), 1e-6)
    y_span = max(float(np.ptp(grid["y"])), 1e-6)
    x_norm = (grid["x"] - grid["x"].mean()) / x_span
    y_norm = (grid["y"] - grid["y"].mean()) / y_span
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    contact_x, contact_y = metadata["contact_point_mm"]
    contact_sigma_mm = 0.30
    contact_map = np.exp(
        -((grid["x"] - contact_x) ** 2 + (grid["y"] - contact_y) ** 2) / (2.0 * contact_sigma_mm**2)
    ).astype(np.float32)
    contact_map *= grid["mask"]
    scalar_condition = np.asarray(
        [indentation / 0.025, friction / 0.10, youngs / 210_000.0, 2.0 * directed_phase - 1.0],
        dtype=np.float32,
    )
    condition_maps = np.broadcast_to(scalar_condition[:, None, None], (4, grid_size, grid_size))
    feature = np.concatenate(
        (grid["mask"][None], x_norm[None], y_norm[None], condition_maps, contact_map[None]), axis=0,
    ).astype(np.float32)
    target = grid["stress"]
    condition = np.asarray(
        [indentation, friction, youngs, contact_fraction, directed_phase, time_s, mean_speed_rpm],
        dtype=np.float32,
    )
    return feature, target, condition

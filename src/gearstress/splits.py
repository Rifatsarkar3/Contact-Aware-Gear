"""Leakage-resistant split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def grouped_trajectory_split(
    trajectory_ids: np.ndarray,
    train_fraction: float = 0.625,
    val_fraction: float = 0.1875,
    seed: int = 42,
) -> GroupSplit:
    """Split sample indices while keeping each trajectory wholly in one partition."""
    ids = np.asarray(trajectory_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("trajectory_ids must be a non-empty one-dimensional array")
    if not (0 < train_fraction < 1 and 0 <= val_fraction < 1):
        raise ValueError("invalid split fractions")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below one")

    groups = np.unique(ids)
    if groups.size < 3:
        raise ValueError("at least three trajectories are required")
    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    n_train = max(1, int(np.floor(groups.size * train_fraction)))
    n_val = max(1, int(np.floor(groups.size * val_fraction)))
    if n_train + n_val >= groups.size:
        n_val = 1
        n_train = groups.size - 2

    train_groups = set(groups[:n_train].tolist())
    val_groups = set(groups[n_train : n_train + n_val].tolist())
    test_groups = set(groups[n_train + n_val :].tolist())
    return GroupSplit(
        train=np.flatnonzero(np.isin(ids, list(train_groups))),
        val=np.flatnonzero(np.isin(ids, list(val_groups))),
        test=np.flatnonzero(np.isin(ids, list(test_groups))),
    )


def geometry_holdout_split(
    covariate: np.ndarray,
    holdout_low: float,
    holdout_high: float,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> GroupSplit:
    """Split by a continuous geometry covariate: test = an unseen covariate band.

    Every sample with ``covariate`` in ``[holdout_low, holdout_high]`` is
    withheld for test -- the model never sees that band of geometry during
    training. The remaining samples are randomly split into train/val by
    ``val_fraction``, matching the training-time distribution used elsewhere
    in this project. Unlike ``grouped_trajectory_split``, the held-out band
    is fixed by value, not chosen by a random group permutation, so it
    tests true out-of-distribution geometry generalization rather than an
    ordinary random split.
    """
    values = np.asarray(covariate)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("covariate must be a non-empty one-dimensional array")
    if holdout_low >= holdout_high:
        raise ValueError("holdout_low must be below holdout_high")
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be in (0, 1)")

    test = np.flatnonzero((values >= holdout_low) & (values <= holdout_high))
    remaining = np.flatnonzero((values < holdout_low) | (values > holdout_high))
    if test.size == 0:
        raise ValueError("no samples fall inside the holdout band")
    if remaining.size < 2:
        raise ValueError("too few non-held-out samples to form train/val")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(remaining)
    n_val = max(1, int(np.floor(shuffled.size * val_fraction)))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return GroupSplit(train=train, val=val, test=test)


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



def grouped_trajectory_fold_split(
    trajectory_ids: np.ndarray,
    fold: int,
    train_fraction: float = 0.625,
    val_fraction: float = 0.1875,
    reference_seed: int = 42,
    seed: int = 2026,
) -> GroupSplit:
    """Further trajectory-grouped partitions that test every trajectory at least once.

    Fold 0 is the reference partition ``grouped_trajectory_split(..., seed=reference_seed)``.
    The trajectories not tested there are shuffled once with ``seed`` and cut into
    test sets of the reference size; the last fold is completed with trajectories
    drawn from the other folds' test sets. Each fold draws its own validation
    trajectories from the remaining ones, and everything else is training data,
    so the partition sizes match the reference partition.
    """
    ids = np.asarray(trajectory_ids)
    reference = grouped_trajectory_split(ids, train_fraction, val_fraction, seed=reference_seed)
    if fold == 0:
        return reference
    groups = np.unique(ids)
    ref_test = sorted(set(ids[reference.test].tolist()))
    ref_val = sorted(set(ids[reference.val].tolist()))
    n_test, n_val = len(ref_test), len(ref_val)
    rng = np.random.default_rng(seed)
    untested = rng.permutation([g for g in groups if g not in ref_test])
    n_folds = -(-len(untested) // n_test)
    if not 1 <= fold <= n_folds:
        raise ValueError(f"fold must be in 0..{n_folds}")
    test = list(untested[(fold - 1) * n_test: fold * n_test])
    if len(test) < n_test:                         # complete the last fold from the other folds' test sets
        others = [g for g in untested if g not in test]
        test += list(rng.permutation(others)[: n_test - len(test)])
    fold_rng = np.random.default_rng(seed + fold)
    val = list(fold_rng.permutation([g for g in groups if g not in test])[:n_val])
    train = [g for g in groups if g not in test and g not in val]
    return GroupSplit(
        train=np.flatnonzero(np.isin(ids, train)),
        val=np.flatnonzero(np.isin(ids, val)),
        test=np.flatnonzero(np.isin(ids, test)),
    )

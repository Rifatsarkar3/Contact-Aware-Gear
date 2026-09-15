import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gearstress.active_learning import (
    FRICTION_RANGE,
    INDENT_RANGE,
    SPEED_RANGE,
    YOUNGS_RANGE,
    Surrogate,
    ensemble_disagreement,
    grid_solved_frame,
    mc_dropout_disagreement,
    normalize_conditions,
    sample_candidate_conditions,
    select_top_k,
    train_surrogate,
)
from gearstress.models import FNO2d


def test_sample_candidate_conditions_respects_ranges():
    rng = np.random.default_rng(0)
    candidates = sample_candidate_conditions(rng, n=500)
    assert candidates.shape == (500, 4)
    assert np.all(candidates[:, 0] >= INDENT_RANGE[0]) and np.all(candidates[:, 0] <= INDENT_RANGE[1])
    assert np.all(candidates[:, 1] >= FRICTION_RANGE[0]) and np.all(candidates[:, 1] <= FRICTION_RANGE[1])
    assert np.all(candidates[:, 2] >= YOUNGS_RANGE[0]) and np.all(candidates[:, 2] <= YOUNGS_RANGE[1])
    assert np.all(candidates[:, 3] >= SPEED_RANGE[0]) and np.all(candidates[:, 3] <= SPEED_RANGE[1])


def test_sample_candidate_conditions_is_reproducible_with_same_rng_seed():
    a = sample_candidate_conditions(np.random.default_rng(7), n=50)
    b = sample_candidate_conditions(np.random.default_rng(7), n=50)
    assert np.allclose(a, b)


def test_normalize_conditions_maps_ranges_to_unit_interval():
    conditions = np.array([
        [INDENT_RANGE[0], FRICTION_RANGE[0], YOUNGS_RANGE[0], SPEED_RANGE[0]],
        [INDENT_RANGE[1], FRICTION_RANGE[1], YOUNGS_RANGE[1], SPEED_RANGE[1]],
    ])
    normalized = normalize_conditions(conditions)
    assert np.allclose(normalized[0], 0.0, atol=1e-6)
    assert np.allclose(normalized[1], 1.0, atol=1e-6)


def test_surrogate_learns_a_simple_linear_relationship():
    rng = np.random.default_rng(1)
    conditions = sample_candidate_conditions(rng, n=200)
    normalized = normalize_conditions(conditions)
    # synthetic target: a known linear function of the normalized features,
    # standing in for "ensemble disagreement" during the unit test.
    true_weights = np.array([1.0, -0.5, 0.25, 2.0])
    targets = normalized @ true_weights

    model = Surrogate()
    train_surrogate(model, normalized, targets, epochs=300, lr=0.01)

    with torch.no_grad():
        pred = model(torch.from_numpy(normalized).float()).squeeze(-1).numpy()
    correlation = np.corrcoef(pred, targets)[0, 1]
    assert correlation > 0.9


def test_ensemble_disagreement_is_zero_for_identical_predictions():
    preds = [torch.ones(3, 3, 8, 8) for _ in range(5)]
    disagreement = ensemble_disagreement(preds)
    assert torch.allclose(disagreement, torch.zeros(3))


def test_ensemble_disagreement_is_positive_for_differing_predictions():
    torch.manual_seed(0)
    preds = [torch.randn(3, 3, 8, 8) for _ in range(5)]
    disagreement = ensemble_disagreement(preds)
    assert torch.all(disagreement > 0)


def test_mc_dropout_disagreement_matches_ensemble_disagreement_shape_and_semantics():
    torch.manual_seed(0)
    model = FNO2d(width=8, modes_x=4, modes_y=4, layers=2, dropout=0.5)
    model.train()
    x = torch.randn(3, 8, 16, 16)
    disagreement = mc_dropout_disagreement(model, x, n_samples=8)
    assert disagreement.shape == (3,)
    assert torch.all(disagreement >= 0)


def test_select_top_k_returns_k_highest_scoring_distinct_indices():
    scores = np.array([0.1, 0.9, 0.3, 0.7, 0.2])
    selected = select_top_k(scores, k=2)
    assert sorted(selected) == [1, 3]
    assert len(set(selected)) == 2


REAL_SOLVED_FRAME = (
    Path(__file__).resolve().parents[1]
    / "outputs" / "gear_pair_active_learning_dataset" / "ensemble_surrogate" / "trajectory_007" / "frame_00"
)


def test_grid_solved_frame_reads_an_already_solved_case_without_recalculix():
    if not REAL_SOLVED_FRAME.exists():
        pytest.skip("real solved case fixture not present in this environment")
    metadata = json.loads((REAL_SOLVED_FRAME / "case.json").read_text(encoding="utf-8"))
    contact_fraction = float(metadata["contact_fraction"])
    directed_phase = (contact_fraction - 0.38) / 0.24

    feature, target, condition = grid_solved_frame(
        REAL_SOLVED_FRAME, grid_size=64,
        indentation=float(metadata["indentation_mm"]),
        friction=float(metadata["friction"]),
        youngs=float(metadata["youngs_modulus_mpa"]),
        contact_fraction=contact_fraction,
        directed_phase=directed_phase,
    )

    assert feature.shape == (8, 64, 64)
    assert target.shape == (3, 64, 64)
    assert condition.shape == (7,)
    assert np.all(np.isfinite(feature))
    assert np.all(np.isfinite(target))
    assert condition[0] == float(metadata["indentation_mm"])
    assert condition[1] == float(metadata["friction"])
    assert condition[2] == float(metadata["youngs_modulus_mpa"])
    assert condition[3] == contact_fraction
    assert np.isnan(condition[5]), "time_s should default to NaN when not supplied"
    assert np.isnan(condition[6]), "mean_speed_rpm should default to NaN when not supplied"

import numpy as np
import pytest
import torch

from gearstress.losses import combined_loss, equilibrium_residual, hotspot_mae, relative_l2
from gearstress.inventions import (
    AdaptiveEquilibriumExchangeFNO,
    CompatibleContactPotentialFNO,
    ContactCenteredMultiscaleFNO,
    DCTFNO,
    DCTSpectralConv2d,
    DualWindowMultiscaleFNO,
    HotspotGuidedMultiscaleFNO,
    LoadPathModulatedFNO,
    MinimumInterventionContactEquilibrium,
    ResidualCompatibleRepairFNO,
    _dct_matrix,
    _soft_argmax_centroid,
    _von_mises_like,
    airy_stress,
    contact_centroid,
    spectral_equilibrium_projection,
    weighted_equilibrium_projection,
)
from gearstress.models import DeepONet, FNO2d, UNetSmall
from gearstress.fem import (
    ContactCase,
    GearPairCase,
    GearPairTransientCase,
    conjugate_involute_flank,
    contact_point_and_normal,
    gear_tooth_polygon,
    involute_flank,
    mating_node_targets,
    mating_tooth_transform,
    parse_final_nodal_force,
    parse_final_nodal_stress,
    parse_nodal_force_history,
    parse_nodal_stress_history,
    tooth_polygon,
    transient_time_samples,
)
from gearstress.proxy import make_proxy_dataset
from gearstress.splits import geometry_holdout_split, grouped_trajectory_split


def test_group_split_has_no_trajectory_leakage():
    ids = np.repeat(np.arange(12), 4)
    split = grouped_trajectory_split(ids, seed=42)
    partitions = [set(ids[index]) for index in (split.train, split.val, split.test)]
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])
    assert sum(map(len, (split.train, split.val, split.test))) == len(ids)


def test_geometry_holdout_split_isolates_the_band_to_test_only():
    rng = np.random.default_rng(0)
    covariate = rng.uniform(17.5, 22.5, size=120)
    split = geometry_holdout_split(covariate, holdout_low=20.8333, holdout_high=22.5, val_fraction=0.2, seed=1)
    assert np.all((covariate[split.test] >= 20.8333) & (covariate[split.test] <= 22.5))
    assert np.all((covariate[split.train] < 20.8333) | (covariate[split.train] > 22.5))
    assert np.all((covariate[split.val] < 20.8333) | (covariate[split.val] > 22.5))
    all_indices = np.concatenate([split.train, split.val, split.test])
    assert len(set(all_indices.tolist())) == covariate.size
    assert set(all_indices.tolist()) == set(range(covariate.size))


def test_models_preserve_field_shape():
    x = torch.randn(2, 8, 32, 32)
    for model in (
        FNO2d(width=8, modes_x=6, modes_y=6, layers=2),
        UNetSmall(width=8),
        DeepONet(width=8, basis_dim=16),
    ):
        assert model(x).shape == (2, 3, 32, 32)


def test_experimental_operators_preserve_field_shape_and_backpropagate():
    x = torch.randn(2, 8, 24, 24, requires_grad=True)
    x[:, 0].data.fill_(1.0)
    for model in (
        AdaptiveEquilibriumExchangeFNO(width=8, modes_x=5, modes_y=5, layers=2, dx=0.2, dy=0.3),
        CompatibleContactPotentialFNO(width=8, modes_x=5, modes_y=5, layers=2, dx=0.2, dy=0.3),
    ):
        prediction = model(x)
        assert prediction.shape == (2, 3, 24, 24)
        prediction.square().mean().backward(retain_graph=True)
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_equilibrium_projection_and_airy_head_are_discretely_compatible():
    stress = torch.randn(2, 3, 32, 32)
    mask = torch.ones(2, 1, 32, 32)
    projected = spectral_equilibrium_projection(stress, dx=0.2, dy=0.3)
    potential_stress = airy_stress(torch.randn(2, 1, 32, 32), dx=0.2, dy=0.3)
    assert equilibrium_residual(projected, mask, dx=0.2, dy=0.3).item() < 1e-8
    assert equilibrium_residual(potential_stress, mask, dx=0.2, dy=0.3).item() < 1e-8


def test_residual_compatible_repair_freezes_baseline_and_backpropagates_student():
    baseline = FNO2d(width=8, modes_x=5, modes_y=5, layers=2)
    model = ResidualCompatibleRepairFNO(
        baseline, width=6, modes_x=4, modes_y=4, layers=1, dx=0.2, dy=0.3
    )
    prediction = model(torch.randn(2, 8, 24, 24))
    prediction.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.baseline.parameters())
    assert any(parameter.grad is not None for parameter in model.repair_backbone.parameters())


def test_load_path_modulated_operator_preserves_shape():
    model = LoadPathModulatedFNO(width=8, modes_x=5, modes_y=5, layers=2)
    prediction = model(torch.randn(2, 8, 24, 24))
    assert prediction.shape == (2, 3, 24, 24)


def test_dct_matrix_is_orthonormal():
    basis = _dct_matrix(16)
    identity = basis @ basis.T
    assert torch.allclose(identity, torch.eye(16), atol=1e-5)


def test_dct_spectral_conv_reconstructs_input_at_full_modes_with_unit_weight():
    layer = DCTSpectralConv2d(1, 1, modes_x=20, modes_y=20, grid_h=20, grid_w=20)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    x = torch.randn(2, 1, 20, 20)
    out = layer(x)
    assert torch.allclose(out, x, atol=1e-4)


def test_dct_spectral_conv_preserves_shape_with_truncated_modes():
    layer = DCTSpectralConv2d(4, 6, modes_x=5, modes_y=5, grid_h=24, grid_w=24)
    x = torch.randn(2, 4, 24, 24, requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 6, 24, 24)
    out.square().mean().backward()
    assert layer.weight.grad is not None
    assert x.grad is not None


def test_dct_fno_preserves_field_shape():
    model = DCTFNO(width=8, modes_x=6, modes_y=6, layers=2, grid_h=32, grid_w=32)
    prediction = model(torch.randn(2, 8, 32, 32))
    assert prediction.shape == (2, 3, 32, 32)


def test_contact_centroid_matches_known_single_mass_location():
    batch, channels, size = 1, 8, 8
    inputs = torch.zeros(batch, channels, size, size)
    coord_extent = 0.5
    coords = torch.linspace(-coord_extent, coord_extent, size)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    inputs[:, 1] = grid_x
    inputs[:, 2] = grid_y
    row, col = 2, 5
    inputs[:, 7, row, col] = 1.0
    center = contact_centroid(inputs, contact_channel=7, x_channel=1, y_channel=2)
    assert torch.allclose(center[0, 0], grid_x[row, col], atol=1e-6)
    assert torch.allclose(center[0, 1], grid_y[row, col], atol=1e-6)


def test_local_sampling_grid_extracts_correct_region():
    from gearstress.inventions import _local_sampling_grid

    size = 16
    coord_extent = 0.5
    coords = torch.linspace(-coord_extent, coord_extent, size)
    field = coords.view(1, 1, 1, size).expand(1, 1, size, size).clone()
    center = torch.tensor([[0.1, 0.0]])
    window = 0.2
    patch_size = 5
    grid = _local_sampling_grid(center, window, patch_size, coord_extent)
    patch = torch.nn.functional.grid_sample(field, grid, mode="bilinear", align_corners=True)
    expected = torch.linspace(0.1 - window / 2, 0.1 + window / 2, patch_size)
    assert torch.allclose(patch[0, 0, 0, :], expected, atol=1e-3)


def test_ccm_fno_local_head_is_zero_initialized():
    model = ContactCenteredMultiscaleFNO(width=8, modes_x=5, modes_y=5, layers=2, local_width=6, local_size=8)
    patch = torch.randn(2, 8, 8, 8)
    delta = model.local_net(patch)
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_ccm_fno_preserves_shape_and_backpropagates():
    model = ContactCenteredMultiscaleFNO(
        width=8, modes_x=5, modes_y=5, layers=2, local_width=6, local_size=8, local_window=0.375
    )
    x = torch.randn(2, 8, 24, 24, requires_grad=True)
    x.data[:, 7] = torch.rand(2, 24, 24)
    prediction = model(x)
    assert prediction.shape == (2, 3, 24, 24)
    prediction.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_von_mises_like_is_nonnegative_and_zero_for_zero_stress():
    stress = torch.zeros(2, 3, 6, 6)
    assert torch.all(_von_mises_like(stress) >= 0.0)
    assert torch.allclose(_von_mises_like(stress), torch.zeros(2, 6, 6), atol=1e-4)


def test_soft_argmax_centroid_concentrates_on_sharp_peak():
    size = 20
    coord_extent = 0.5
    coords = torch.linspace(-coord_extent, coord_extent, size)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    row, col = 3, 15
    magnitude = torch.full((1, size, size), -10.0)
    magnitude[0, row, col] = 10.0
    center = _soft_argmax_centroid(magnitude, grid_x.unsqueeze(0), grid_y.unsqueeze(0), temperature=0.05)
    assert torch.allclose(center[0, 0], grid_x[row, col], atol=1e-3)
    assert torch.allclose(center[0, 1], grid_y[row, col], atol=1e-3)


def test_hgm_fno_local_head_is_zero_initialized():
    model = HotspotGuidedMultiscaleFNO(width=8, modes_x=5, modes_y=5, layers=2, local_width=6, local_size=8)
    patch = torch.randn(2, 8, 8, 8)
    delta = model.local_net(patch)
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_hgm_fno_preserves_shape_and_backpropagates():
    model = HotspotGuidedMultiscaleFNO(
        width=8, modes_x=5, modes_y=5, layers=2, local_width=6, local_size=8, local_window=0.375
    )
    x = torch.randn(2, 8, 24, 24, requires_grad=True)
    x.data[:, 7] = torch.rand(2, 24, 24)
    prediction = model(x)
    assert prediction.shape == (2, 3, 24, 24)
    prediction.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert model.last_hotspot_center is not None
    assert model.last_hotspot_center.shape == (2, 2)


def test_dwm_fno_both_local_heads_are_zero_initialized():
    model = DualWindowMultiscaleFNO(width=8, modes_x=5, modes_y=5, layers=2, local_width=6, local_size=8)
    patch = torch.randn(2, 8, 8, 8)
    assert torch.allclose(model.contact_local_net(patch), torch.zeros(2, 3, 8, 8))
    assert torch.allclose(model.hotspot_local_net(patch), torch.zeros(2, 3, 8, 8))


def test_dwm_fno_preserves_shape_and_backpropagates():
    model = DualWindowMultiscaleFNO(
        width=8, modes_x=5, modes_y=5, layers=2, local_width=6, local_size=8, local_window=0.375
    )
    x = torch.randn(2, 8, 24, 24, requires_grad=True)
    x.data[:, 7] = torch.rand(2, 24, 24)
    prediction = model(x)
    assert prediction.shape == (2, 3, 24, 24)
    prediction.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert model.last_contact_center is not None
    assert model.last_hotspot_center is not None


def test_weighted_projection_reduces_masked_equilibrium():
    stress = torch.randn(2, 3, 20, 20)
    mask = torch.ones(2, 1, 20, 20)
    weight = torch.ones_like(stress)
    before = equilibrium_residual(stress, mask, dx=0.2, dy=0.3)
    projected = weighted_equilibrium_projection(
        stress, mask, weight, dx=0.2, dy=0.3, iterations=100
    )
    after = equilibrium_residual(projected, mask, dx=0.2, dy=0.3)
    assert after < before * 1e-4


def test_minimum_intervention_operator_meets_budget_without_touching_valid_input():
    mask = torch.ones(2, 1, 20, 20)
    contact = torch.zeros_like(mask)
    layer = MinimumInterventionContactEquilibrium(
        dx=0.2,
        dy=0.3,
        equilibrium_budget=1.0,
        contact_weight=2.0,
        hotspot_weight=2.0,
        projection_iterations=100,
    )
    invalid = torch.randn(1, 3, 20, 20)
    valid = airy_stress(torch.randn(1, 1, 20, 20), dx=0.2, dy=0.3)
    inputs = torch.cat((invalid, valid), dim=0)
    repaired = layer(inputs, mask, contact)
    residual = equilibrium_residual(repaired[:1], mask[:1], dx=0.2, dy=0.3)
    assert residual <= 1.01
    assert torch.allclose(repaired[1:], valid, atol=1e-6)


def test_losses_are_zero_for_exact_prediction():
    y = torch.randn(2, 3, 16, 16)
    assert relative_l2(y, y).item() == pytest.approx(0.0)
    assert hotspot_mae(y, y).item() == pytest.approx(0.0)
    assert equilibrium_residual(torch.zeros_like(y)).item() == pytest.approx(0.0)


def test_equilibrium_residual_excludes_mask_boundary_stencils():
    stress = torch.zeros(1, 3, 12, 12)
    mask = torch.zeros(1, 1, 12, 12)
    mask[:, :, 2:10, 2:10] = 1.0
    stress[:, 0, 2:10, 2:10] = 2.0
    stress[:, 1, 2:10, 2:10] = -1.0
    assert equilibrium_residual(stress, mask, dx=0.2, dy=0.3).item() == pytest.approx(0.0)


def test_combined_loss_reference_normalizes_projected_equilibrium():
    target = torch.randn(2, 3, 12, 12)
    mask = torch.ones(2, 1, 12, 12)
    total, parts = combined_loss(
        target,
        target,
        mask,
        hotspot_weight=2.0,
        physics_weight=0.05,
        hotspot_quantile=0.95,
        dx=0.2,
        dy=0.3,
    )
    assert parts["physics"].item() == pytest.approx(1.0)
    assert total.item() == pytest.approx(0.05)


def test_proxy_is_grouped_and_finite():
    data = make_proxy_dataset(trajectories=3, frames_per_trajectory=2, grid_size=16)
    assert data["inputs"].shape == (6, 8, 16, 16)
    assert data["targets"].shape == (6, 3, 16, 16)
    assert np.isfinite(data["targets"]).all()
    assert np.array_equal(np.bincount(data["trajectory_ids"]), np.array([2, 2, 2]))


def test_involute_geometry_is_finite_and_radially_ordered():
    case = ContactCase()
    flank = involute_flank(case)
    radius = np.linalg.norm(flank, axis=1)
    assert np.all(np.diff(radius) >= -1e-9)
    assert radius[0] == pytest.approx(case.base_radius_mm)
    assert radius[-1] == pytest.approx(case.outer_radius_mm)
    polygon = tooth_polygon(case)
    assert polygon.shape[0] > 40
    assert np.isfinite(polygon).all()
    point, normal = contact_point_and_normal(case)
    assert np.isfinite(point).all()
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert point[0] * normal[0] > 0
    center = point + normal * (case.indenter_radius_mm + case.initial_gap_mm)
    assert np.linalg.norm(polygon - center, axis=1).min() >= case.indenter_radius_mm


def test_mating_tooth_transform_aligns_complementary_involutes():
    case = GearPairCase(contact_fraction=0.42)
    rotation, translation, point, normal = mating_tooth_transform(case)
    polygon = gear_tooth_polygon(case)
    mating = polygon @ rotation.T + translation
    clearance = np.linalg.norm(polygon[:, None, :] - mating[None, :, :], axis=2).min()
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.isfinite(point).all()
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert clearance == pytest.approx(case.initial_gap_mm, abs=5e-4)
    assert np.linalg.norm(translation) == pytest.approx(2.0 * case.pitch_radius_mm, rel=0.001)


def test_transient_time_samples_constant_speed_are_uniform():
    times = transient_time_samples(
        phase_start=0.38, phase_end=0.62, n_steps=4, mean_speed_rpm=3000.0,
    )
    assert times.shape == (5, 2)
    assert times[0, 0] == pytest.approx(0.0)
    assert np.all(np.diff(times[:, 0]) > 0)
    assert np.allclose(np.diff(times[:, 0]), np.diff(times[:, 0])[0], rtol=1e-6)
    assert times[0, 1] == pytest.approx(0.38)
    assert times[-1, 1] == pytest.approx(0.62)


def test_transient_time_samples_with_ripple_are_nonuniform_but_monotonic():
    times = transient_time_samples(
        phase_start=0.38, phase_end=0.62, n_steps=6, mean_speed_rpm=3000.0,
        speed_ripple_fraction=0.3,
    )
    assert np.all(np.diff(times[:, 0]) > 0)
    steps = np.diff(times[:, 0])
    assert steps.max() / steps.min() > 1.01


def test_transient_case_root_targets_match_static_transform_at_each_phase():
    case = GearPairTransientCase(phase_start=0.38, phase_end=0.62, n_time_steps=4)
    rotation0, translation0, _, _ = mating_tooth_transform(case.as_gear_pair_case(case.phase_start))
    polygon = gear_tooth_polygon(case.as_gear_pair_case(case.phase_start))
    mating0 = polygon @ rotation0.T + translation0
    reference_node_xy = {i: mating0[i] for i in range(0, len(mating0), 7)}
    samples = transient_time_samples(
        case.phase_start, case.phase_end, case.n_time_steps, case.mean_speed_rpm
    )
    targets = mating_node_targets(
        case, reference_node_xy, rotation0, translation0, phase=samples[-1, 1], load_fraction=1.0
    )
    rotation_end, translation_end, _, normal_end = mating_tooth_transform(
        case.as_gear_pair_case(case.phase_end)
    )
    expected = polygon @ rotation_end.T + translation_end - normal_end * case.indentation_mm
    for node_id, xy0 in reference_node_xy.items():
        assert np.allclose(targets[node_id], expected[node_id], atol=1e-8)


def test_corrected_external_gear_flank_thins_toward_tip():
    case = GearPairCase()
    flank = conjugate_involute_flank(case)
    angular_half_thickness = np.arctan2(np.abs(flank[:, 0]), flank[:, 1])
    assert angular_half_thickness[-1] < angular_half_thickness[0]


def test_frd_stress_parser_reads_last_block(tmp_path):
    frd = tmp_path / "tiny.frd"
    frd.write_text(
        " -4  STRESS      6    1\n"
        " -5  SXX         1    4    1    1\n"
        " -1         1" + "1.00000E+000" * 6 + "\n"
        " -3\n",
        encoding="ascii",
    )
    stress = parse_final_nodal_stress(frd)
    assert np.array_equal(stress[1], np.ones(6, dtype=np.float32))


def test_frd_stress_history_parser_reads_every_block(tmp_path):
    frd = tmp_path / "multi.frd"
    frd.write_text(
        " -4  STRESS      6    1\n"
        " -5  SXX         1    4    1    1\n"
        " -1         1" + "1.00000E+000" * 6 + "\n"
        " -3\n"
        " -4  STRESS      6    1\n"
        " -5  SXX         1    4    1    1\n"
        " -1         1" + "2.00000E+000" * 6 + "\n"
        " -3\n",
        encoding="ascii",
    )
    history = parse_nodal_stress_history(frd)
    assert len(history) == 2
    assert np.array_equal(history[0][1], np.ones(6, dtype=np.float32))
    assert np.array_equal(history[1][1], np.full(6, 2.0, dtype=np.float32))


def test_frd_force_parser_reads_last_and_every_block(tmp_path):
    frd = tmp_path / "force.frd"
    frd.write_text(
        " -4  FORC        4    1\n"
        " -5  F1          1    2    1    0\n"
        " -1         1" + "1.00000E+000" * 3 + "\n"
        " -3\n"
        " -4  FORC        4    1\n"
        " -5  F1          1    2    1    0\n"
        " -1         1" + "3.00000E+000" * 3 + "\n"
        " -3\n",
        encoding="ascii",
    )
    last = parse_final_nodal_force(frd)
    assert np.array_equal(last[1], np.full(3, 3.0, dtype=np.float32))
    history = parse_nodal_force_history(frd)
    assert len(history) == 2
    assert np.array_equal(history[0][1], np.ones(3, dtype=np.float32))


def test_fno2d_dropout_zero_matches_baseline_exactly():
    torch.manual_seed(0)
    baseline = FNO2d(width=8, modes_x=4, modes_y=4, layers=2)
    torch.manual_seed(0)
    zero_dropout = FNO2d(width=8, modes_x=4, modes_y=4, layers=2, dropout=0.0)
    x = torch.randn(2, 8, 16, 16)
    baseline.eval()
    zero_dropout.eval()
    assert torch.allclose(baseline(x), zero_dropout(x))


def test_fno2d_dropout_active_in_train_mode_gives_varying_output():
    torch.manual_seed(0)
    model = FNO2d(width=8, modes_x=4, modes_y=4, layers=2, dropout=0.3)
    model.train()
    x = torch.randn(2, 8, 16, 16)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2)


def test_fno2d_dropout_disabled_in_eval_mode():
    torch.manual_seed(0)
    model = FNO2d(width=8, modes_x=4, modes_y=4, layers=2, dropout=0.3)
    model.eval()
    x = torch.randn(2, 8, 16, 16)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)

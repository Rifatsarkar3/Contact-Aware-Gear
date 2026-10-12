import numpy as np
import pytest

from gearstress.fem import element_material_mask, gear_mesh_material_mask
from gearstress.splits import grouped_trajectory_fold_split, grouped_trajectory_split


def _ids(n_traj=16, per=9, variants=3):
    """Trajectory id of every case; the friction variants of a case keep its trajectory."""
    return np.concatenate([np.repeat(np.arange(n_traj), per)] * variants)


def test_fold_zero_is_the_reference_partition():
    ids = _ids()
    ref = grouped_trajectory_split(ids, 0.625, 0.1875, seed=42)
    fold = grouped_trajectory_fold_split(ids, 0)
    for part in ("train", "val", "test"):
        assert np.array_equal(getattr(ref, part), getattr(fold, part))


def test_folds_are_grouped_disjoint_and_test_every_trajectory():
    ids = _ids()
    ref = grouped_trajectory_split(ids, 0.625, 0.1875, seed=42)
    sizes = tuple(len(set(ids[getattr(ref, p)].tolist())) for p in ("train", "val", "test"))
    tested = set()
    for k in range(6):
        split = grouped_trajectory_fold_split(ids, k)
        groups = [set(ids[getattr(split, p)].tolist()) for p in ("train", "val", "test")]
        tr, va, te = groups
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert tr | va | te == set(range(16))
        assert tuple(len(g) for g in groups) == sizes
        for p in ("train", "val", "test"):
            idx = getattr(split, p)
            # every case of a trajectory, including its friction variants, lies in the same set
            assert set(np.flatnonzero(np.isin(ids, list(set(ids[idx].tolist())))).tolist()) == set(idx.tolist())
        tested |= te
    assert tested == set(range(16))


def test_fold_index_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        grouped_trajectory_fold_split(_ids(), 6)


def test_element_mask_excludes_points_outside_the_elements(tmp_path):
    # an L-shaped domain: its convex hull covers the notch at (1.5, 1.5), its elements do not
    coordinates = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]], dtype=float)
    np.savez(tmp_path / "mesh_data.npz", node_ids=np.arange(10, 16), coordinates=coordinates,
             driver_connectivity=np.array([[10, 11, 12], [10, 12, 13], [10, 13, 14], [10, 14, 15]]))
    xx = np.array([[0.5, 1.5, 1.5]])
    yy = np.array([[0.5, 0.5, 1.5]])
    assert element_material_mask(tmp_path, xx, yy).tolist() == [[True, True, False]]


def test_gear_mesh_mask_matches_the_datasets():
    mask = gear_mesh_material_mask(64)
    assert mask.shape == (64, 64) and int(mask.sum()) == 1832

"""Reconstruct an ML-ready dataset from FEM cases already solved during the active-learning study.

**No new FEM solving.** Re-grids already-solved case directories under
outputs/gear_pair_active_learning_dataset/<arm>/ via grid_solved_frame(),
and combines them with v1's original 63 samples into one dataset matching
the v1/v2/v3 schema, written to
data/processed/gear_pair_al_<arm>_extended.h5.

v1's original val/test trajectories are never touched by any arm's AL run
(only fresh trajectory IDs above max(v1 ids) ever get added), so this
script does not re-split the combined trajectory pool -- re-running
grouped_trajectory_split on a changed trajectory count would silently
scramble which trajectories land in test. Instead it recovers v1's
original train/val/test trajectory-ID sets directly from v1.h5 (which
never changes) and reuses them: train = v1's original train trajectories
+ every newly-added trajectory; val/test = v1's original val/test
trajectories, unchanged.

Which trajectories are "newly-added" differs by arm:
  - random_baseline, ensemble_surrogate: every trajectory either ever
    solved was immediately added to the training set (no rejection --
    ensemble_surrogate's surrogate model decides what to solve *before*
    solving, not after, so there is no solved-but-unused reserve pool
    either), so every trajectory_* directory on disk under either arm
    is, by construction, part of the training set. Directory listing
    alone is authoritative.
  - pool_based: pre-solves a larger candidate pool upfront and only
    *selects* a subset across its rounds -- directory listing alone
    is NOT authoritative (unselected candidates are also solved and
    sit on disk). This arm requires an explicit `labeled_trajectory_ids`
    list from its round_6_results.json (added in a later change to
    run_active_learning_study.py); older results predating that field
    will cause this script to fail loudly rather than silently include
    unselected candidates.

time_s and mean_speed_rpm are per-trajectory scheduling metadata that
were never persisted to disk during the AL run (not part of GearPairCase
/ case.json). They are recorded as NaN in the reconstructed `conditions`
array for newly-added trajectories -- disclosed and harmless, since no
training/evaluation script in this project reads the `conditions` array.

Usage:
    python scripts/reconstruct_al_dataset.py --arm random_baseline
    python scripts/reconstruct_al_dataset.py --arm pool_based
    python scripts/reconstruct_al_dataset.py --arm ensemble_surrogate
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.active_learning import grid_solved_frame  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

V1_DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v1.h5"
CONFIG = ROOT / "configs" / "smoke.yaml"
AL_CASE_ROOT = ROOT / "outputs" / "gear_pair_active_learning_dataset"
AL_RESULTS_ROOT = ROOT / "outputs" / "active_learning"
GRID_SIZE = 64
FRAMES_PER_TRAJECTORY = 9


def load_v1_split_trajectory_ids() -> dict[str, set[int]]:
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


def resolve_selected_trajectory_ids(arm: str, max_v1_id: int) -> list[int]:
    """Return the *newly-added* trajectory global indices actually used as training data by this arm's AL run.

    Excludes v1's own seed trajectories even where the source data
    technically includes them (run_active_learning_study.py's
    labeled_trajectory_ids is initialized from v1's seed pool, then
    grown -- it is not "newly-added trajectories only"), since v1's
    samples are already included separately via v1_inputs/v1_targets/...
    in main(). New AL trajectories always have an id > max(v1 ids) by
    construction (next_trajectory_index = max(v1 ids) + 1, incrementing).
    """
    arm_root = AL_CASE_ROOT / arm
    solved_dirs = sorted(arm_root.glob("trajectory_*"))
    if not solved_dirs:
        raise FileNotFoundError(f"no solved trajectories found under {arm_root}")
    solved_ids = [int(d.name.split("_")[-1]) for d in solved_dirs]

    if arm in ("random_baseline", "ensemble_surrogate"):
        new_ids = [i for i in solved_ids if i > max_v1_id]
        return new_ids

    if arm == "pool_based":
        results_path = AL_RESULTS_ROOT / arm / "round_6_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"no completed run found for arm={arm}: {results_path}")
        results = json.loads(results_path.read_text(encoding="utf-8"))
        if "labeled_trajectory_ids" not in results:
            raise KeyError(
                f"{results_path} has no 'labeled_trajectory_ids' field -- this run predates that field "
                f"being added (see run_active_learning_study.py). pool_based's upfront-solved candidate "
                f"pool includes trajectories that were never selected, so directory listing alone is not "
                f"a valid substitute. Re-run `python scripts/run_active_learning_study.py --method pool_based "
                f"--rounds 6` first, then retry this script."
            )
        labeled_ids = [int(t) for t in results["labeled_trajectory_ids"]]
        new_ids = [i for i in labeled_ids if i > max_v1_id]
        return new_ids

    raise ValueError(f"unknown arm: {arm}")


def grid_one_new_trajectory(trajectory_dir: Path, trajectory_global_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-grid every already-solved frame of one trajectory (no new FEM solving)."""
    frame_dirs = sorted(trajectory_dir.glob("frame_*"))
    if len(frame_dirs) != FRAMES_PER_TRAJECTORY:
        raise ValueError(f"{trajectory_dir}: expected {FRAMES_PER_TRAJECTORY} frames, found {len(frame_dirs)}")

    inputs, targets, conditions = [], [], []
    for frame_dir in frame_dirs:
        print(f"    {frame_dir.name}...", flush=True)
        metadata = json.loads((frame_dir / "case.json").read_text(encoding="utf-8"))
        indentation = float(metadata["indentation_mm"])
        friction = float(metadata["friction"])
        youngs = float(metadata["youngs_modulus_mpa"])
        contact_fraction = float(metadata["contact_fraction"])
        directed_phase = (contact_fraction - 0.38) / 0.24
        feature, target, condition = grid_solved_frame(
            frame_dir, GRID_SIZE, indentation, friction, youngs, contact_fraction, directed_phase,
        )
        inputs.append(feature)
        targets.append(target)
        conditions.append(condition)
        gc.collect()

    trajectory_ids = np.full(FRAMES_PER_TRAJECTORY, trajectory_global_index, dtype=np.int64)
    return np.stack(inputs), np.stack(targets), np.stack(conditions), trajectory_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("pool_based", "random_baseline", "ensemble_surrogate"))
    args = parser.parse_args()

    with h5py.File(V1_DATA, "r") as handle:
        v1_inputs = np.asarray(handle["inputs"], dtype=np.float32)
        v1_targets = np.asarray(handle["targets"], dtype=np.float32)
        v1_trajectory_ids = np.asarray(handle["trajectory_ids"])
        v1_conditions = np.asarray(handle["conditions"], dtype=np.float32)
        grid_dx = float(handle.attrs["grid_dx_mm"])
        grid_dy = float(handle.attrs["grid_dy_mm"])
        mesh_size = float(handle.attrs["mesh_size_mm"])

    v1_split_ids = load_v1_split_trajectory_ids()
    max_v1_id = int(v1_trajectory_ids.max())
    selected_ids = resolve_selected_trajectory_ids(args.arm, max_v1_id)

    all_inputs, all_targets, all_conditions, all_trajectory_ids = [v1_inputs], [v1_targets], [v1_conditions], [v1_trajectory_ids]
    arm_root = AL_CASE_ROOT / args.arm
    for i, trajectory_global_index in enumerate(selected_ids, start=1):
        print(f"[{i}/{len(selected_ids)}] gridding trajectory_{trajectory_global_index:03d}...", flush=True)
        trajectory_dir = arm_root / f"trajectory_{trajectory_global_index:03d}"
        inputs, targets, conditions, trajectory_ids = grid_one_new_trajectory(trajectory_dir, trajectory_global_index)
        all_inputs.append(inputs)
        all_targets.append(targets)
        all_conditions.append(conditions)
        all_trajectory_ids.append(trajectory_ids)
        gc.collect()

    combined_inputs = np.concatenate(all_inputs)
    combined_targets = np.concatenate(all_targets)
    combined_conditions = np.concatenate(all_conditions)
    combined_trajectory_ids = np.concatenate(all_trajectory_ids)

    combined_id_set = set(int(t) for t in combined_trajectory_ids.tolist())
    n_trajectories = len(combined_id_set)
    expected_trajectories = len(v1_split_ids["train"]) + len(v1_split_ids["val"]) + len(v1_split_ids["test"]) + len(selected_ids)
    assert n_trajectories == expected_trajectories, f"expected {expected_trajectories} trajectories, got {n_trajectories}"
    assert v1_split_ids["val"] <= combined_id_set, "v1's val trajectories missing from reconstructed dataset"
    assert v1_split_ids["test"] <= combined_id_set, "v1's test trajectories missing from reconstructed dataset"

    out_path = ROOT / "data" / "processed" / f"gear_pair_al_{args.arm}_extended.h5"
    with h5py.File(out_path, "w") as handle:
        handle.create_dataset("inputs", data=combined_inputs)
        handle.create_dataset("targets", data=combined_targets)
        handle.create_dataset("trajectory_ids", data=combined_trajectory_ids)
        handle.create_dataset("conditions", data=combined_conditions)
        handle.attrs["grid_dx_mm"] = grid_dx
        handle.attrs["grid_dy_mm"] = grid_dy
        handle.attrs["mesh_size_mm"] = mesh_size
        handle.attrs["source"] = (
            f"v1 (63 samples) + {args.arm} AL-selected trajectories "
            f"({len(selected_ids)} trajectories, {len(selected_ids) * FRAMES_PER_TRAJECTORY} samples)"
        )
        handle.attrs["condition_columns"] = (
            "indentation_mm, friction, youngs_modulus_mpa, contact_fraction, directed_phase, "
            "time_s (NaN for AL-added trajectories), mean_speed_rpm (NaN for AL-added trajectories)"
        )

    n_train_trajectories = len(v1_split_ids["train"]) + len(selected_ids)
    print(
        f"[written] {out_path}: {combined_inputs.shape[0]} samples, {n_trajectories} trajectories "
        f"({n_train_trajectories} train / {n_train_trajectories * FRAMES_PER_TRAJECTORY} samples, "
        f"{len(v1_split_ids['val'])} val, {len(v1_split_ids['test'])} test)"
    )


if __name__ == "__main__":
    main()

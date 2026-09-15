"""Uncertainty-guided active learning: all four arms, shared scaffold.

**Do NOT run this script until GPU is free and the user gives an explicit
go-ahead** -- see
docs/superpowers/specs/2026-08-04-uncertainty-guided-active-learning-design.md.

Starting from v1's 7 seed trajectories (36 samples), grows the training
pool by 2 trajectories/round for up to 6 rounds (checkpoints at rounds
2/4/6 -> 11/15/19 trajectories -> 99/135/171 samples), evaluating on v1's
own fixed, untouched test set throughout. Four acquisition strategies:

- ensemble_surrogate: 10-seed FNO ensemble; a small MLP surrogate maps
  the known trajectory-level scalar condition (base indentation, friction,
  Young's modulus, mean speed) to predicted ensemble disagreement, since
  the full spatial input for a candidate cannot be scored before it is
  solved via FEM (input and label come from the same solve in this
  pipeline). The top-2 surrogate-scored candidates are solved for real
  each round.
- mc_dropout: same surrogate-acquisition pattern, but the disagreement
  signal used to train the surrogate comes from repeated stochastic
  forward passes of ONE dropout-enabled model (seed 1) rather than
  across-seed ensemble variance. For a fair accuracy comparison against
  the other arms, 10 independently seeded dropout models are still
  trained and evaluated at each checkpoint -- only the *acquisition*
  signal differs, not the evaluation protocol.
- pool_based: pre-solves a larger candidate pool (~20 trajectories) via
  FEM upfront, then selects each round from the already-solved remainder
  using *real* ensemble disagreement (no surrogate needed, since inputs
  already exist).
- random_baseline: identical round structure, candidates chosen uniformly
  at random rather than by any uncertainty signal. Method-agnostic --
  generated once and reused as the common comparison arm for all three
  UQ methods above.

Usage (once launched):
    python scripts/run_active_learning_study.py --method ensemble_surrogate --rounds 6
    python scripts/run_active_learning_study.py --method mc_dropout --rounds 6
    python scripts/run_active_learning_study.py --method pool_based --rounds 6
    python scripts/run_active_learning_study.py --method random_baseline --rounds 6
"""

from __future__ import annotations

import argparse
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

from gearstress.active_learning import (  # noqa: E402
    Surrogate,
    ensemble_disagreement,
    grid_solved_frame,
    mc_dropout_disagreement,
    normalize_conditions,
    sample_candidate_conditions,
    select_top_k,
    train_surrogate,
)
from gearstress.fem import (  # noqa: E402
    GearPairCase,
    build_gear_pair_case,
    contact_element_counts,
    run_calculix,
    transient_time_samples,
)
from gearstress.losses import combined_loss, equilibrium_residual, hotspot_mae, relative_l2  # noqa: E402
from gearstress.models import FNO2d  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

V1_DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v1.h5"
CONFIG = ROOT / "configs" / "smoke.yaml"
CCX = ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe"
AL_CASE_ROOT = ROOT / "outputs" / "gear_pair_active_learning_dataset"
OUT_ROOT = ROOT / "outputs" / "active_learning"

FRAMES_PER_TRAJECTORY = 9
GRID_SIZE = 64
MESH_SIZE_MM = 0.12
ROUNDS_TOTAL = 6
TRAJECTORIES_PER_ROUND = 2
CHECKPOINT_ROUNDS = (2, 4, 6)
CANDIDATE_POOL_SIZE = 200
POOL_BASED_UPFRONT_SIZE = 20

SEEDS = list(range(1, 11))
EVAL_EPOCHS = 300
ACQUISITION_EPOCHS = 300  # models used purely for acquisition scoring still train fully -- no shortcuts that could bias the disagreement signal
MC_DROPOUT_RATE = 0.2
MC_DROPOUT_SAMPLES = 20

# Another job may already be resident on the GPU. Measured peak footprint of
# this workload (10-model ensemble resident + one model training) is ~180 MiB
# -- capping at 20% of total device memory (~2.4 GiB on a 12 GiB card) leaves
# over an order of magnitude of headroom while guaranteeing we can never grow
# into memory the other job needs.
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


# ---------------------------------------------------------------------------
# FEM solving for one new trajectory (mirrors
# generate_gear_pair_transient_quasistatic_dataset.py's per-trajectory body,
# parameterized so active learning can call it one trajectory at a time).
# ---------------------------------------------------------------------------


def solve_one_trajectory(
    trajectory_global_index: int,
    arm: str,
    base_indent: float,
    friction: float,
    youngs: float,
    mean_speed_rpm: float,
    direction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve all FRAMES_PER_TRAJECTORY frames of one trajectory via real CalculiX.

    Returns (inputs, targets, conditions) arrays of shape
    (frames, 8, GRID_SIZE, GRID_SIZE), (frames, 3, GRID_SIZE, GRID_SIZE),
    (frames, 7) respectively -- identical schema to v1/v2/v3.
    """
    case_root = AL_CASE_ROOT / arm / f"trajectory_{trajectory_global_index:03d}"
    inputs, targets, conditions = [], [], []

    samples = transient_time_samples(
        phase_start=0.38, phase_end=0.62, n_steps=FRAMES_PER_TRAJECTORY - 1, mean_speed_rpm=mean_speed_rpm,
    )
    time_s_by_frame = samples[:, 0]
    phase_by_frame = samples[:, 1] if direction > 0 else samples[::-1, 1]

    for frame in range(FRAMES_PER_TRAJECTORY):
        phase = frame / max(FRAMES_PER_TRAJECTORY - 1, 1)
        directed_phase = float((phase_by_frame[frame] - 0.38) / 0.24)
        contact_fraction = float(phase_by_frame[frame])
        time_s = float(time_s_by_frame[frame])
        indentation = float(base_indent * (0.85 + 0.30 * np.sin(np.pi * phase)))

        case = GearPairCase(
            friction=friction,
            youngs_modulus_mpa=youngs,
            indentation_mm=indentation,
            contact_fraction=contact_fraction,
            mesh_size_mm=MESH_SIZE_MM,
        )
        case_dir = case_root / f"frame_{frame:02d}"
        deck = build_gear_pair_case(case, case_dir)
        result = run_calculix(deck, CCX, timeout_seconds=600)
        if result.returncode != 0:
            raise RuntimeError(f"active-learning solve failed: arm={arm} trajectory={trajectory_global_index} frame={frame}")
        counts = contact_element_counts(case_dir / "solver.log")
        if not counts or max(counts) <= 0:
            raise RuntimeError(f"inactive contact: arm={arm} trajectory={trajectory_global_index} frame={frame}")

        feature, target, condition = grid_solved_frame(
            case_dir, GRID_SIZE, indentation, friction, youngs, contact_fraction, directed_phase, time_s, mean_speed_rpm,
        )
        inputs.append(feature)
        targets.append(target)
        conditions.append(condition)

    return np.stack(inputs), np.stack(targets), np.stack(conditions)


# ---------------------------------------------------------------------------
# Seed-set loading, recovering each v1 trajectory's true trajectory-level
# base indentation (not the phase-modulated per-frame value stored in the
# dataset): indentation = base_indent * (0.85 + 0.30*sin(pi*phase)), and at
# frame 0, phase = 0 -> indentation_frame0 = base_indent * 0.85.
# ---------------------------------------------------------------------------


def load_v1_seed_pool_and_fixed_sets() -> dict:
    with h5py.File(V1_DATA, "r") as handle:
        inputs = np.asarray(handle["inputs"], dtype=np.float32)
        targets = np.asarray(handle["targets"], dtype=np.float32)
        trajectory_ids = np.asarray(handle["trajectory_ids"])
        conditions = np.asarray(handle["conditions"], dtype=np.float32)
        grid_dx = float(handle.attrs.get("grid_dx_mm", 1.0))
        grid_dy = float(handle.attrs.get("grid_dy_mm", 1.0))

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    split = grouped_trajectory_split(
        trajectory_ids, train_fraction=float(cfg["train_fraction"]), val_fraction=float(cfg["val_fraction"]), seed=int(cfg["seed"]),
    )

    seed_trajectory_level_conditions = []
    for traj_id in sorted(set(trajectory_ids[split.train].tolist())):
        frame0 = np.flatnonzero((trajectory_ids == traj_id))[0]
        indentation_frame0, friction, youngs, _, _, _, mean_speed_rpm = conditions[frame0]
        base_indent = float(indentation_frame0) / 0.85
        seed_trajectory_level_conditions.append((base_indent, float(friction), float(youngs), float(mean_speed_rpm)))

    return {
        "inputs": inputs,
        "targets": targets,
        "trajectory_ids": trajectory_ids,
        "conditions": conditions,
        "train_idx": split.train,
        "val_idx": split.val,
        "test_idx": split.test,
        "grid_dx": grid_dx,
        "grid_dy": grid_dy,
        "seed_trajectory_level_conditions": seed_trajectory_level_conditions,
    }


# ---------------------------------------------------------------------------
# Training / evaluation, matching scripts/run_training_size_curve.py exactly
# (same loss config, optimizer, best-val checkpoint selection).
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, device, dx, dy) -> dict[str, float]:
    model.eval()
    rel, hot, eq, count = 0.0, 0.0, 0.0, 0
    for batch_inputs, batch_targets in loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        prediction = model(batch_inputs)
        batch = batch_inputs.shape[0]
        rel += float(relative_l2(prediction, batch_targets)) * batch
        hot += float(hotspot_mae(prediction, batch_targets)) * batch
        eq += float(equilibrium_residual(prediction, batch_inputs[:, :1], dx=dx, dy=dy)) * batch
        count += batch
    return {"relative_l2": rel / count, "hotspot_mae": hot / count, "equilibrium_residual": eq / count}


def make_loader(inputs, targets, indices, batch_size, shuffle, seed):
    dataset = TensorDataset(torch.from_numpy(inputs[indices]), torch.from_numpy(targets[indices]))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_one_model(
    inputs, targets_raw, train_idx, val_idx, seed, cfg, grid_dx, grid_dy, dropout: float = 0.0,
) -> tuple[FNO2d, float]:
    seed_all(seed)
    target_scale = max(float(np.max(np.abs(targets_raw[train_idx]))), 1e-8)
    targets = targets_raw / target_scale
    train_loader = make_loader(inputs, targets, train_idx, int(cfg["batch_size"]), True, seed)
    val_loader = make_loader(inputs, targets, val_idx, int(cfg["batch_size"]), False, seed)

    model_cfg = dict(cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNO2d(
        width=model_cfg["width"], modes_x=model_cfg["modes_x"], modes_y=model_cfg["modes_y"],
        layers=model_cfg["layers"], dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    best_state, best_val = None, float("inf")
    for _ in range(EVAL_EPOCHS):
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
    return model, target_scale


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    m = mean(diffs)
    s = stdev(diffs) if n > 1 else 0.0
    se = s / (n**0.5) if n > 1 else 0.0
    t = m / se if se > 0 else float("inf")
    return {"mean_diff": m, "sd_diff": s, "t": t, "n": n}


# ---------------------------------------------------------------------------
# Acquisition: score a pool of candidates and pick the top-k.
# ---------------------------------------------------------------------------


def acquire_via_surrogate(
    pool_state: dict, cfg, arm: str, round_seed: int,
) -> list[tuple[float, float, float, float]]:
    """Train models on the current pool, fit a surrogate on their disagreement,
    score a freshly sampled candidate pool, return the top TRAJECTORIES_PER_ROUND
    (base_indent, friction, youngs, speed) tuples to solve next.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arm == "ensemble_surrogate":
        models = []
        for seed in range(1, 4):  # 3 quick acquisition-scoring seeds, distinct from the 10 evaluation seeds -- keeps acquisition cheap relative to the full evaluation protocol
            model, _ = train_one_model(
                pool_state["inputs"], pool_state["targets"], pool_state["labeled_idx"], pool_state["val_idx"], seed, cfg, pool_state["grid_dx"], pool_state["grid_dy"],
            )
            models.append(model)
        labeled_disagreement = []
        for traj_id in pool_state["labeled_trajectory_ids"]:
            frame_idx = np.flatnonzero(pool_state["trajectory_ids"] == traj_id)
            x = torch.from_numpy(pool_state["inputs"][frame_idx]).float().to(device)
            preds = [m(x) for m in models]
            disagreement = ensemble_disagreement(preds).mean().item()
            labeled_disagreement.append(disagreement)
    else:  # mc_dropout
        model, _ = train_one_model(
            pool_state["inputs"], pool_state["targets"], pool_state["labeled_idx"], pool_state["val_idx"], 1, cfg, pool_state["grid_dx"], pool_state["grid_dy"], dropout=MC_DROPOUT_RATE,
        )
        model.train()  # dropout must be active for MC sampling
        labeled_disagreement = []
        for traj_id in pool_state["labeled_trajectory_ids"]:
            frame_idx = np.flatnonzero(pool_state["trajectory_ids"] == traj_id)
            x = torch.from_numpy(pool_state["inputs"][frame_idx]).float().to(device)
            disagreement = mc_dropout_disagreement(model, x, MC_DROPOUT_SAMPLES).mean().item()
            labeled_disagreement.append(disagreement)

    surrogate = Surrogate()
    normalized_labeled = normalize_conditions(np.array(pool_state["labeled_trajectory_level_conditions"]))
    train_surrogate(surrogate, normalized_labeled, np.array(labeled_disagreement))

    rng = np.random.default_rng(round_seed)
    candidates = sample_candidate_conditions(rng, CANDIDATE_POOL_SIZE)
    normalized_candidates = normalize_conditions(candidates)
    with torch.no_grad():
        scores = surrogate(torch.from_numpy(normalized_candidates).float()).squeeze(-1).numpy()
    top_idx = select_top_k(scores, TRAJECTORIES_PER_ROUND)
    return [tuple(candidates[i]) for i in top_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("ensemble_surrogate", "mc_dropout", "pool_based", "random_baseline"))
    parser.add_argument("--rounds", type=int, default=ROUNDS_TOTAL)
    args = parser.parse_args()

    cap_gpu_memory()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seed_pool = load_v1_seed_pool_and_fixed_sets()

    labeled_idx = seed_pool["train_idx"].copy()
    labeled_trajectory_ids = sorted(set(seed_pool["trajectory_ids"][labeled_idx].tolist()))
    labeled_trajectory_level_conditions = list(seed_pool["seed_trajectory_level_conditions"])
    inputs = seed_pool["inputs"]
    targets = seed_pool["targets"]
    trajectory_ids = seed_pool["trajectory_ids"]

    next_trajectory_index = max(trajectory_ids.tolist()) + 1
    rng = np.random.default_rng(hash(args.method) % (2**31))

    pool_based_reserve: list[tuple[int, tuple]] = []
    if args.method == "pool_based":
        for _ in range(POOL_BASED_UPFRONT_SIZE):
            base_indent, friction, youngs, speed = sample_candidate_conditions(rng, 1)[0]
            direction = -1.0 if next_trajectory_index % 2 else 1.0
            new_inputs, new_targets, new_conditions = solve_one_trajectory(
                next_trajectory_index, args.method, base_indent, friction, youngs, speed, direction,
            )
            inputs = np.concatenate([inputs, new_inputs])
            targets = np.concatenate([targets, new_targets])
            trajectory_ids = np.concatenate([trajectory_ids, np.full(FRAMES_PER_TRAJECTORY, next_trajectory_index, dtype=trajectory_ids.dtype)])
            pool_based_reserve.append((next_trajectory_index, (base_indent, friction, youngs, speed)))
            next_trajectory_index += 1

    summary = {}
    for round_number in range(1, args.rounds + 1):
        if args.method == "random_baseline":
            chosen = [tuple(sample_candidate_conditions(rng, 1)[0]) for _ in range(TRAJECTORIES_PER_ROUND)]
            for base_indent, friction, youngs, speed in chosen:
                direction = -1.0 if next_trajectory_index % 2 else 1.0
                new_inputs, new_targets, _ = solve_one_trajectory(next_trajectory_index, args.method, base_indent, friction, youngs, speed, direction)
                inputs = np.concatenate([inputs, new_inputs])
                targets = np.concatenate([targets, new_targets])
                trajectory_ids = np.concatenate([trajectory_ids, np.full(FRAMES_PER_TRAJECTORY, next_trajectory_index, dtype=trajectory_ids.dtype)])
                labeled_idx = np.concatenate([labeled_idx, np.flatnonzero(trajectory_ids == next_trajectory_index)])
                labeled_trajectory_ids.append(next_trajectory_index)
                labeled_trajectory_level_conditions.append((base_indent, friction, youngs, speed))
                next_trajectory_index += 1

        elif args.method == "pool_based":
            pool_state = {
                "inputs": inputs, "targets": targets, "trajectory_ids": trajectory_ids,
                "labeled_idx": labeled_idx, "val_idx": seed_pool["val_idx"], "grid_dx": seed_pool["grid_dx"], "grid_dy": seed_pool["grid_dy"],
            }
            models = []
            for s in range(1, 4):
                model, _ = train_one_model(inputs, targets, labeled_idx, seed_pool["val_idx"], s, cfg, seed_pool["grid_dx"], seed_pool["grid_dy"])
                models.append(model)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            scored = []
            for traj_id, condition_tuple in pool_based_reserve:
                frame_idx = np.flatnonzero(trajectory_ids == traj_id)
                x = torch.from_numpy(inputs[frame_idx]).float().to(device)
                preds = [m(x) for m in models]
                scored.append(ensemble_disagreement(preds).mean().item())
            top_idx = select_top_k(np.array(scored), TRAJECTORIES_PER_ROUND)
            for i in sorted(top_idx, reverse=True):
                traj_id, condition_tuple = pool_based_reserve.pop(i)
                labeled_idx = np.concatenate([labeled_idx, np.flatnonzero(trajectory_ids == traj_id)])
                labeled_trajectory_ids.append(traj_id)
                labeled_trajectory_level_conditions.append(condition_tuple)

        else:  # ensemble_surrogate or mc_dropout
            pool_state = {
                "inputs": inputs, "targets": targets, "trajectory_ids": trajectory_ids,
                "labeled_idx": labeled_idx, "val_idx": seed_pool["val_idx"], "grid_dx": seed_pool["grid_dx"], "grid_dy": seed_pool["grid_dy"],
                "labeled_trajectory_ids": labeled_trajectory_ids, "labeled_trajectory_level_conditions": labeled_trajectory_level_conditions,
            }
            chosen = acquire_via_surrogate(pool_state, cfg, args.method, round_seed=1000 + round_number)
            for base_indent, friction, youngs, speed in chosen:
                direction = -1.0 if next_trajectory_index % 2 else 1.0
                new_inputs, new_targets, _ = solve_one_trajectory(next_trajectory_index, args.method, base_indent, friction, youngs, speed, direction)
                inputs = np.concatenate([inputs, new_inputs])
                targets = np.concatenate([targets, new_targets])
                trajectory_ids = np.concatenate([trajectory_ids, np.full(FRAMES_PER_TRAJECTORY, next_trajectory_index, dtype=trajectory_ids.dtype)])
                labeled_idx = np.concatenate([labeled_idx, np.flatnonzero(trajectory_ids == next_trajectory_index)])
                labeled_trajectory_ids.append(next_trajectory_index)
                labeled_trajectory_level_conditions.append((base_indent, friction, youngs, speed))
                next_trajectory_index += 1

        if round_number in CHECKPOINT_ROUNDS:
            dropout_rate = MC_DROPOUT_RATE if args.method == "mc_dropout" else 0.0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            out_dir = OUT_ROOT / args.method
            checkpoint_dir = out_dir / f"round_{round_number}_models"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            per_seed_results = []
            for seed in SEEDS:
                model, target_scale = train_one_model(inputs, targets, labeled_idx, seed_pool["val_idx"], seed, cfg, seed_pool["grid_dx"], seed_pool["grid_dy"], dropout=dropout_rate)
                test_targets_scaled = targets / target_scale
                test_loader = make_loader(inputs, test_targets_scaled, seed_pool["test_idx"], int(cfg["batch_size"]), False, seed)
                metrics = evaluate(model, test_loader, device, seed_pool["grid_dx"], seed_pool["grid_dy"])
                per_seed_results.append({"seed": seed, "target_scale": target_scale, **metrics})
                # Saved (not just the aggregate metric) so run_uq_calibration_check.py can
                # reload every seed's weights and compute per-sample cross-seed disagreement
                # against true error, without retraining.
                torch.save(model.state_dict(), checkpoint_dir / f"seed{seed}.pt")
            summary[f"round_{round_number}"] = {
                "n_train": int(labeled_idx.size),
                "n_train_trajectories": len(labeled_trajectory_ids),
                "labeled_trajectory_ids": list(labeled_trajectory_ids),
                "dropout_rate": dropout_rate,
                "per_seed_results": per_seed_results,
                "relative_l2_mean": mean(r["relative_l2"] for r in per_seed_results),
                "relative_l2_std": stdev(r["relative_l2"] for r in per_seed_results),
                "hotspot_mae_mean": mean(r["hotspot_mae"] for r in per_seed_results),
                "hotspot_mae_std": stdev(r["hotspot_mae"] for r in per_seed_results),
            }
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"round_{round_number}_results.json").write_text(json.dumps(summary[f"round_{round_number}"], indent=2), encoding="utf-8")
            print(f"[checkpoint] {args.method} round {round_number}: N={labeled_idx.size} "
                  f"relative_l2={summary[f'round_{round_number}']['relative_l2_mean']:.4f} "
                  f"hotspot_mae={summary[f'round_{round_number}']['hotspot_mae_mean']:.4f}")

    out_dir = OUT_ROOT / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "full_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary written] {out_dir / 'full_summary.json'}")


if __name__ == "__main__":
    main()

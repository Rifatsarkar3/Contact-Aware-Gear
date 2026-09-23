"""Seed-42, <=10-epoch software smoke trainer."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.inventions import (  # noqa: E402
    AdaptiveEquilibriumExchangeFNO,
    CompatibleContactPotentialFNO,
    ContactCenteredMultiscaleFNO,
    DCTFNO,
    DualWindowMultiscaleFNO,
    HotspotGuidedMultiscaleFNO,
    LoadPathModulatedFNO,
    ResidualCompatibleRepairFNO,
)
from gearstress.losses import combined_loss, equilibrium_residual, hotspot_mae, relative_l2  # noqa: E402
from gearstress.models import DeepONet, FNO2d, UNetSmall  # noqa: E402
from gearstress.splits import geometry_holdout_split, grouped_trajectory_split  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dx: float,
    dy: float,
) -> dict[str, float]:
    model.eval()
    rel, hot, equilibrium, count = 0.0, 0.0, 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        prediction = model(inputs)
        batch = inputs.shape[0]
        rel += float(relative_l2(prediction, targets)) * batch
        hot += float(hotspot_mae(prediction, targets)) * batch
        equilibrium += float(equilibrium_residual(prediction, inputs[:, :1], dx=dx, dy=dy)) * batch
        count += batch
    metrics = {
        "relative_l2": rel / count,
        "hotspot_mae": hot / count,
        "equilibrium_residual": equilibrium / count,
    }
    if isinstance(model, AdaptiveEquilibriumExchangeFNO) and model.last_exchange is not None:
        metrics["mean_equilibrium_exchange"] = float(model.last_exchange.mean())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "analytic_proxy_v0.h5")
    parser.add_argument(
        "--model",
        choices=(
            "fno",
            "fno_physics",
            "unet",
            "deeponet",
            "aee_fno",
            "ccp_fno",
            "rcr_fno",
            "lpm_fno",
            "lpm_rcr_fno",
            "dct_fno",
            "ccm_fno",
            "hgm_fno",
            "dwm_fno",
        ),
        default="fno",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--physics-warmup-epochs", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--repair-fraction", type=float, default=None)
    parser.add_argument("--width", type=int, default=None, help="overrides configs/smoke.yaml's model width, for architecture search")
    parser.add_argument("--basis-dim", type=int, default=64, help="deeponet: shared branch/trunk basis dimension")
    parser.add_argument("--modes-x", type=int, default=None, help="overrides configs/smoke.yaml's model modes_x")
    parser.add_argument("--modes-y", type=int, default=None, help="overrides configs/smoke.yaml's model modes_y")
    parser.add_argument("--layers", type=int, default=None, help="overrides configs/smoke.yaml's model layers")
    parser.add_argument("--local-width", type=int, default=16, help="ccm_fno: local-branch channel width")
    parser.add_argument("--local-window", type=float, default=0.375, help="ccm_fno: local window side length in x/y-channel units")
    parser.add_argument("--local-size", type=int, default=16, help="ccm_fno: local window resolution")
    parser.add_argument("--gate-sigma-fraction", type=float, default=0.25, help="ccm_fno/hgm_fno/dwm_fno: Gaussian gate sigma as a fraction of local_window")
    parser.add_argument("--softmax-temperature", type=float, default=0.1, help="hgm_fno/dwm_fno: soft-argmax temperature for the predicted-hotspot window center")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="overrides the model init/training seed only; the train/val/test trajectory split always uses the config seed, so multi-seed runs stay leakage-free and comparable",
    )
    parser.add_argument(
        "--allow-extended-training",
        action="store_true",
        help="permit --epochs above the 10-epoch smoke cap for an explicit, deliberate diagnostic run",
    )
    parser.add_argument(
        "--split-scheme",
        choices=("grouped_trajectory", "geometry_holdout"),
        default="grouped_trajectory",
        help="geometry_holdout requires a 'pressure_angle_deg' dataset in --data and withholds a fixed band for test, testing true out-of-distribution geometry generalization instead of a random split",
    )
    parser.add_argument("--holdout-low", type=float, default=None, help="geometry_holdout: lower bound (inclusive) of the withheld pressure_angle_deg band")
    parser.add_argument("--holdout-high", type=float, default=None, help="geometry_holdout: upper bound (inclusive) of the withheld pressure_angle_deg band")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="overrides the train/val/test trajectory split seed only (default: the config seed); use this to test whether a finding is specific to one partition of the data, not model-init seed",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    epochs = args.epochs if args.epochs is not None else int(cfg["epochs"])
    learning_rate = args.learning_rate if args.learning_rate is not None else float(cfg["learning_rate"])
    if epochs > 10 and not args.allow_extended_training:
        raise ValueError("smoke runs are capped at 10 epochs unless --allow-extended-training is set")
    if not (0 <= args.physics_warmup_epochs < epochs):
        raise ValueError("physics warmup must be non-negative and below the epoch budget")
    split_seed = args.split_seed if args.split_seed is not None else int(cfg["seed"])
    run_seed = args.seed if args.seed is not None else split_seed
    seed_all(run_seed)
    run_suffix = ""
    if args.seed is not None:
        run_suffix += f"_seed{run_seed}"
    if args.allow_extended_training:
        run_suffix += f"_ep{epochs}"

    with h5py.File(args.data, "r") as handle:
        inputs = np.asarray(handle["inputs"], dtype=np.float32)
        targets = np.asarray(handle["targets"], dtype=np.float32)
        trajectory_ids = np.asarray(handle["trajectory_ids"])
        evidence_status = str(handle.attrs.get("evidence_status", "software smoke only"))
        grid_dx_mm = float(handle.attrs.get("grid_dx_mm", 1.0))
        grid_dy_mm = float(handle.attrs.get("grid_dy_mm", 1.0))
        pressure_angle_deg = (
            np.asarray(handle["pressure_angle_deg"], dtype=np.float32) if "pressure_angle_deg" in handle else None
        )
    if args.split_scheme == "geometry_holdout":
        if pressure_angle_deg is None:
            raise ValueError("--split-scheme geometry_holdout requires a 'pressure_angle_deg' dataset in --data")
        if args.holdout_low is None or args.holdout_high is None:
            raise ValueError("--split-scheme geometry_holdout requires --holdout-low and --holdout-high")
        split = geometry_holdout_split(
            pressure_angle_deg,
            holdout_low=args.holdout_low,
            holdout_high=args.holdout_high,
            val_fraction=float(cfg["val_fraction"]),
            seed=split_seed,
        )
    else:
        split = grouped_trajectory_split(
            trajectory_ids,
            train_fraction=float(cfg["train_fraction"]),
            val_fraction=float(cfg["val_fraction"]),
            seed=split_seed,
        )
    # Fit the stress scale on training trajectories only; validation/test labels
    # never influence preprocessing.
    target_scale = max(float(np.max(np.abs(targets[split.train]))), 1e-8)
    targets = targets / target_scale

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(inputs[indices]), torch.from_numpy(targets[indices]))
        generator = torch.Generator().manual_seed(run_seed)
        return DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=shuffle, generator=generator)

    train_loader = loader(split.train, True)
    val_loader = loader(split.val, False)
    test_loader = loader(split.test, False)
    model_cfg = dict(cfg["model"])
    if args.width is not None:
        model_cfg["width"] = args.width
    if args.modes_x is not None:
        model_cfg["modes_x"] = args.modes_x
    if args.modes_y is not None:
        model_cfg["modes_y"] = args.modes_y
    if args.layers is not None:
        model_cfg["layers"] = args.layers
    grid_size = int(cfg.get("grid_size", 64))
    if args.model.startswith("fno"):
        model = FNO2d(width=model_cfg["width"], modes_x=model_cfg["modes_x"], modes_y=model_cfg["modes_y"], layers=model_cfg["layers"])
    elif args.model == "aee_fno":
        model = AdaptiveEquilibriumExchangeFNO(
            width=model_cfg["width"],
            modes_x=model_cfg["modes_x"],
            modes_y=model_cfg["modes_y"],
            layers=model_cfg["layers"],
            dx=grid_dx_mm,
            dy=grid_dy_mm,
        )
    elif args.model == "ccp_fno":
        model = CompatibleContactPotentialFNO(
            width=model_cfg["width"],
            modes_x=model_cfg["modes_x"],
            modes_y=model_cfg["modes_y"],
            layers=model_cfg["layers"],
            dx=grid_dx_mm,
            dy=grid_dy_mm,
        )
    elif args.model == "rcr_fno":
        baseline = FNO2d(
            width=model_cfg["width"],
            modes_x=model_cfg["modes_x"],
            modes_y=model_cfg["modes_y"],
            layers=model_cfg["layers"],
        )
        baseline_path = ROOT / "outputs" / "smoke" / args.data.stem / f"fno{run_suffix}" / "model.pt"
        if not baseline_path.exists():
            raise FileNotFoundError(f"RCR-FNO requires the accepted baseline checkpoint: {baseline_path}")
        baseline.load_state_dict(torch.load(baseline_path, map_location="cpu"))
        model = ResidualCompatibleRepairFNO(
            baseline,
            dx=grid_dx_mm,
            dy=grid_dy_mm,
        )
    elif args.model == "lpm_fno":
        model = LoadPathModulatedFNO(
            width=model_cfg["width"],
            modes_x=model_cfg["modes_x"],
            modes_y=model_cfg["modes_y"],
            layers=model_cfg["layers"],
        )
    elif args.model == "lpm_rcr_fno":
        baseline = LoadPathModulatedFNO(
            width=model_cfg["width"],
            modes_x=model_cfg["modes_x"],
            modes_y=model_cfg["modes_y"],
            layers=model_cfg["layers"],
        )
        baseline_path = ROOT / "outputs" / "smoke" / args.data.stem / f"lpm_fno{run_suffix}" / "model.pt"
        if not baseline_path.exists():
            raise FileNotFoundError(f"LPM-RCR-FNO requires its frozen parent checkpoint: {baseline_path}")
        baseline.load_state_dict(torch.load(baseline_path, map_location="cpu"))
        model = ResidualCompatibleRepairFNO(
            baseline,
            dx=grid_dx_mm,
            dy=grid_dy_mm,
            repair_fraction=args.repair_fraction if args.repair_fraction is not None else 0.6,
        )
    elif args.model == "dct_fno":
        model = DCTFNO(
            width=model_cfg["width"],
            modes_x=model_cfg["modes_x"],
            modes_y=model_cfg["modes_y"],
            layers=model_cfg["layers"],
            grid_h=grid_size,
            grid_w=grid_size,
        )
    elif args.model == "ccm_fno":
        model = ContactCenteredMultiscaleFNO(
            width=model_cfg["width"],
            modes_x=model_cfg["modes_x"],
            modes_y=model_cfg["modes_y"],
            layers=model_cfg["layers"],
            local_width=args.local_width,
            local_window=args.local_window,
            local_size=args.local_size,
            gate_sigma_fraction=args.gate_sigma_fraction,
        )
    elif args.model == "hgm_fno":
        model = HotspotGuidedMultiscaleFNO(
            width=model_cfg["width"],
            modes_x=model_cfg["modes_x"],
            modes_y=model_cfg["modes_y"],
            layers=model_cfg["layers"],
            local_width=args.local_width,
            local_window=args.local_window,
            local_size=args.local_size,
            gate_sigma_fraction=args.gate_sigma_fraction,
            softmax_temperature=args.softmax_temperature,
        )
    elif args.model == "dwm_fno":
        model = DualWindowMultiscaleFNO(
            width=model_cfg["width"],
            modes_x=model_cfg["modes_x"],
            modes_y=model_cfg["modes_y"],
            layers=model_cfg["layers"],
            local_width=args.local_width,
            local_window=args.local_window,
            local_size=args.local_size,
            gate_sigma_fraction=args.gate_sigma_fraction,
            softmax_temperature=args.softmax_temperature,
        )
    elif args.model == "unet":
        model = UNetSmall(width=model_cfg["width"])
    else:
        model = DeepONet(width=model_cfg["width"], basis_dim=args.basis_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(cfg["weight_decay"]))
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            prediction = model(batch_inputs)
            physics_weight = 0.0
            if args.model == "fno_physics":
                base_physics_weight = float(cfg["loss"]["physics_weight"])
                if args.physics_warmup_epochs:
                    ramp = max(0.0, (epoch - args.physics_warmup_epochs) / (epochs - args.physics_warmup_epochs))
                    physics_weight = base_physics_weight * ramp
                else:
                    physics_weight = base_physics_weight
            loss, _ = combined_loss(
                prediction,
                batch_targets,
                batch_inputs[:, :1],
                hotspot_weight=float(cfg["loss"]["hotspot_weight"]),
                physics_weight=physics_weight,
                hotspot_quantile=float(cfg["loss"]["hotspot_quantile"]),
                dx=grid_dx_mm,
                dy=grid_dy_mm,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * batch_inputs.shape[0]
            seen += batch_inputs.shape[0]
        val = evaluate(model, val_loader, device, dx=grid_dx_mm, dy=grid_dy_mm)
        history.append({"epoch": epoch, "train_loss": running / seen, "physics_weight": physics_weight, **{f"val_{k}": v for k, v in val.items()}})
        if val["relative_l2"] < best_val:
            best_val = val["relative_l2"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(history[-1])
    assert best_state is not None
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, dx=grid_dx_mm, dy=grid_dy_mm)
    result = {
        "evidence_status": evidence_status,
        "model": args.model,
        "seed": run_seed,
        "split_seed": split_seed,
        "epochs": epochs,
        "physics_warmup_epochs": args.physics_warmup_epochs,
        "learning_rate": learning_rate,
        "split_samples": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "target_scale_mpa_or_native": target_scale,
        "grid_spacing_mm": {"dx": grid_dx_mm, "dy": grid_dy_mm},
        "model_architecture": {
            "width": model_cfg["width"],
            "modes_x": model_cfg["modes_x"],
            "modes_y": model_cfg["modes_y"],
            "layers": model_cfg["layers"],
        },
        "best_val_relative_l2": best_val,
        "test": test,
        "history": history,
    }
    if isinstance(model, ResidualCompatibleRepairFNO):
        result["repair_fraction"] = model.repair_fraction
    run_name = args.model
    if args.model == "fno_physics" and args.physics_warmup_epochs:
        run_name = f"fno_physics_warmup{args.physics_warmup_epochs}"
    if args.learning_rate is not None:
        run_name = f"{run_name}_lr{learning_rate:g}"
    if args.model == "lpm_rcr_fno" and args.repair_fraction is not None:
        run_name = f"{run_name}_rf{args.repair_fraction:g}"
    if any(v is not None for v in (args.width, args.modes_x, args.modes_y, args.layers)):
        run_name = (
            f"{run_name}_w{model_cfg['width']}_mx{model_cfg['modes_x']}"
            f"_my{model_cfg['modes_y']}_l{model_cfg['layers']}"
        )
    if args.model in ("ccm_fno", "hgm_fno", "dwm_fno"):
        run_name = f"{run_name}_lw{args.local_width}_lwin{args.local_window:g}_lsz{args.local_size}"
    if args.model == "deeponet":
        run_name = f"{run_name}_bd{args.basis_dim}"
    if args.split_scheme == "geometry_holdout":
        run_name = f"{run_name}_geomholdout{args.holdout_low:g}-{args.holdout_high:g}"
    if args.split_seed is not None:
        run_name = f"{run_name}_splitseed{split_seed}"
    run_name = f"{run_name}{run_suffix}"
    out_dir = ROOT / "outputs" / "smoke" / args.data.stem / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(best_state, out_dir / "model.pt")
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()

"""Generate the non-evidentiary analytic proxy dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.proxy import make_proxy_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "analytic_proxy_v0.h5")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = make_proxy_dataset(
        trajectories=config["trajectories"],
        frames_per_trajectory=config["frames_per_trajectory"],
        grid_size=config["grid_size"],
        seed=config["seed"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.attrs["evidence_status"] = "software-smoke-only; not FEM evidence"
        for key, value in data.items():
            handle.create_dataset(key, data=value, compression="gzip", shuffle=True)
    print(f"Wrote {args.output} with {data['inputs'].shape[0]} samples")


if __name__ == "__main__":
    main()


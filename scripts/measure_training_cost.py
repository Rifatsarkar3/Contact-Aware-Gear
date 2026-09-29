"""Wall-clock training cost of a ten-network surrogate on this machine's CPU.

Times a short run of the same training script, model and data used for the ten
v2 networks (vanilla FNO, 90 training cases), then scales the measured time per
epoch to 300 epochs and ten networks. The short run is written to a scratch
seed and removed afterwards, so it cannot be picked up as a trained member.
Run it on an otherwise idle machine.

Usage:
    python scripts/measure_training_cost.py [--epochs 20]
Output: outputs/surrogate/training_cost.json
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "surrogate"
SCRATCH_SEED = 9999


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dataset", default="gear_pair_transient_quasistatic_v2")
    a = ap.parse_args()
    DATA = ROOT / "data" / "processed" / f"{a.dataset}.h5"
    OUT = OUT_DIR / ("training_cost.json" if a.dataset.endswith("_v2") else f"training_cost_{a.dataset.rsplit('_', 1)[1]}.json")
    cmd = [sys.executable, str(ROOT / "scripts" / "train_smoke.py"), "--data", str(DATA), "--model", "fno",
           "--seed", str(SCRATCH_SEED), "--epochs", str(a.epochs), "--allow-extended-training"]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    wall = time.perf_counter() - t0
    run_dir = ROOT / "outputs" / "smoke" / DATA.stem / f"fno_seed{SCRATCH_SEED}_ep{a.epochs}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    per_epoch = wall / a.epochs          # includes process start-up, so slightly pessimistic
    result = {"epochs_timed": a.epochs, "wall_seconds": wall, "seconds_per_epoch": per_epoch,
              "one_network_300_epochs_hours": 300 * per_epoch / 3600,
              "ten_networks_hours": 10 * 300 * per_epoch / 3600,
              "ten_networks_v2_hours": 10 * 300 * per_epoch / 3600,  # kept for older readers
              "dataset": a.dataset,
              "hardware": {"processor": platform.processor(), "torch_threads": torch.get_num_threads(),
                           "torch": torch.__version__},
              "method": "one short training run of the v2 vanilla FNO, scaled linearly to 300 epochs x 10 seeds"}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{wall:.1f} s for {a.epochs} epochs -> {per_epoch:.2f} s/epoch; ten networks x 300 epochs = "
          f"{result['ten_networks_hours']:.2f} h. Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

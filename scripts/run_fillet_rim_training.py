"""Train the reference ensemble on the filleted-root, deformable-rim dataset v5f.

Same protocol as the reference ensemble on v4f: the vanilla FNO with the default
configuration (width 24, 12 Fourier modes in each direction, four layers), 300
epochs, seeds 1 to 10, the reference partition (trajectory-grouped split, seed 42),
each run single-threaded on the CPU through scripts/train_smoke.py. Runs whose
results.json exists are skipped.

Usage: python scripts/run_fillet_rim_training.py [--workers 10]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "gear_mesh_quasistatic_v5f.h5"
RUNS = ROOT / "outputs" / "smoke" / "gear_mesh_quasistatic_v5f"
SEEDS = tuple(range(1, 11))
LOG = ROOT / "outputs" / "fillet_rim" / "train_queue.log"


def train(seed: int) -> str:
    run = RUNS / f"fno_seed{seed}_ep300"
    if (run / "results.json").exists():
        return f"seed {seed}: done earlier"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, str(ROOT / "scripts" / "train_smoke.py"), "--data", str(DATA), "--model", "fno",
           "--seed", str(seed), "--epochs", "300", "--allow-extended-training"]
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%m-%d %H:%M:%S')} start seed {seed}\n")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%m-%d %H:%M:%S')} end seed {seed} rc={r.returncode} ({time.time() - t0:.0f} s)\n")
        if r.returncode:
            f.write(r.stdout[-2000:] + r.stderr[-2000:] + "\n")
    return f"seed {seed}: rc={r.returncode} ({time.time() - t0:.0f} s)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for msg in ex.map(train, SEEDS):
            print(msg, flush=True)


if __name__ == "__main__":
    main()

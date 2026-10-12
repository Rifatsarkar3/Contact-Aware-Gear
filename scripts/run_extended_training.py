"""Training queue for the corrected datasets (v4, v4f after the material-mask correction).

Stages, each job a call of scripts/train_smoke.py with one CPU thread:

1. Screening with a matched budget: every model family gets six configurations,
   each trained with two screening seeds (101, 102) for 100 epochs on the
   reference partition of v4f. The configuration with the lowest mean
   validation relative L2 (full grid, the checkpoint-selection criterion of
   every run) is selected.
     FNO          width {16, 24, 32} x Fourier modes {8, 12}, four layers
     Physics-FNO  equilibrium-penalty weight {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
     U-Net        base width {16, 24, 32, 38, 48, 64}
     DeepONet     width {32, 64, 98} x basis size {32, 128}
     POD-DeepONet branch width {32, 64, 98} x POD rank {32, 128} (trunk fixed to the POD modes)
2. Reference partition: the vanilla FNO (width 24, 12 modes) with seeds 1-10 for
   300 epochs on v4 and on v4f.
3. Finals: each selected configuration with seeds 1-10 for 300 epochs on v4f
   (the FNO reuses stage 2 when the default configuration is selected).
4. Partitions: the vanilla FNO with seeds 1-10 for 300 epochs on each of the
   five further trajectory-grouped partitions of v4f (gearstress.splits).

Resumable: a job whose results.json exists is skipped. The number of parallel
jobs is re-read from outputs/extended_study/train_workers.txt whenever a slot frees.

Usage: python scripts/run_extended_training.py
Outputs: outputs/smoke/gear_mesh_quasistatic_v4{,f}/..., outputs/extended_study/model_selection.json
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
SMOKE = ROOT / "outputs" / "smoke"
EXT = ROOT / "outputs" / "extended_study"
LOG = EXT / "logs" / "train_queue.log"
WORKERS_FILE = EXT / "train_workers.txt"
V4, V4F = "gear_mesh_quasistatic_v4", "gear_mesh_quasistatic_v4f"
SCREEN_SEEDS, FINAL_SEEDS = (101, 102), tuple(range(1, 11))

GRIDS = {
    "fno": [{"model": "fno", "width": w, "modes": m} for w in (16, 24, 32) for m in (8, 12)],
    "fno_physics": [{"model": "fno_physics", "physics_weight": w} for w in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)],
    "unet": [{"model": "unet", "width": w} for w in (16, 24, 32, 38, 48, 64)],
    "deeponet": [{"model": "deeponet", "width": w, "basis": b} for w in (32, 64, 98) for b in (32, 128)],
    "pod_deeponet": [{"model": "pod_deeponet", "width": w, "basis": b} for w in (32, 64, 98) for b in (32, 128)],
}


def job(dataset: str, cfg: dict, seed: int, epochs: int, fold: int | None = None, explicit: bool = True):
    """(run directory, command arguments) for one training run, mirroring train_smoke's run naming."""
    args = ["--data", str(DATA / f"{dataset}.h5"), "--model", cfg["model"], "--seed", str(seed),
            "--epochs", str(epochs), "--allow-extended-training"]
    name = cfg["model"]
    if cfg["model"] == "fno_physics" and "physics_weight" in cfg:
        args += ["--physics-weight", f"{cfg['physics_weight']:g}"]
        name = f"fno_physics_pw{cfg['physics_weight']:g}"
    if explicit and "width" in cfg:
        modes = cfg.get("modes", 12)
        args += ["--width", str(cfg["width"]), "--modes-x", str(modes), "--modes-y", str(modes), "--layers", "4"]
        name = f"{name}_w{cfg['width']}_mx{modes}_my{modes}_l4"
    if cfg["model"] in ("deeponet", "pod_deeponet"):
        args += ["--basis-dim", str(cfg["basis"])]
        name = f"{name}_bd{cfg['basis']}"
    if fold is not None:
        args += ["--fold", str(fold)]
        name = f"{name}_fold{fold}"
    return SMOKE / dataset / f"{name}_seed{seed}_ep{epochs}", args


def screening_jobs():
    return [job(V4F, cfg, s, 100) for fam in GRIDS.values() for cfg in fam for s in SCREEN_SEEDS]


def reference_jobs():
    default = {"model": "fno"}
    return [job(ds, default, s, 300, explicit=False) for ds in (V4F, V4) for s in FINAL_SEEDS]


def partition_jobs():
    return [job(V4F, {"model": "fno"}, s, 300, fold=k, explicit=False) for k in range(1, 6) for s in FINAL_SEEDS]


def select():
    """Configuration with the lowest mean validation error over the screening seeds, per family."""
    out = {}
    for fam, grid in GRIDS.items():
        scores = []
        for cfg in grid:
            vals = [json.loads((job(V4F, cfg, s, 100)[0] / "results.json").read_text())["best_val_relative_l2"]
                    for s in SCREEN_SEEDS]
            scores.append({"config": cfg, "val_relative_l2": vals, "mean": statistics.fmean(vals)})
        best = min(scores, key=lambda r: r["mean"])
        out[fam] = {"selected": best["config"], "scores": scores}
    (EXT / "model_selection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def final_jobs(selection):
    jobs = []
    for fam, sel in selection.items():
        cfg = sel["selected"]
        if fam == "fno" and cfg["width"] == 24 and cfg["modes"] == 12:
            continue                                     # identical to the reference runs of stage 2
        jobs += [job(V4F, cfg, s, 300) for s in FINAL_SEEDS]
    return jobs


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%m-%d %H:%M:%S')} {msg}\n")


def workers() -> int:
    try:
        return max(1, int(WORKERS_FILE.read_text().strip()))
    except (OSError, ValueError):
        return 4


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    screen = screening_jobs()
    queue = screen + reference_jobs()
    finals_added = False
    tail = partition_jobs()
    running: dict[subprocess.Popen, tuple[Path, float]] = {}
    log(f"start: {len(queue)} jobs queued, {len(tail)} partition jobs pending")
    while queue or tail or running or not finals_added:
        for p in [p for p in running if p.poll() is not None]:
            out, t0 = running.pop(p)
            ok = (out / "results.json").exists()
            log(f"end {out.parent.name}/{out.name} rc={p.returncode} {(time.time() - t0) / 60:.1f} min{'' if ok else ' MISSING'}")
        if not finals_added and all((o / "results.json").exists() for o, _ in screen):
            sel = select()
            log("selection: " + json.dumps({k: v["selected"] for k, v in sel.items()}))
            queue += final_jobs(sel)
            finals_added = True
        if not queue and finals_added and tail:
            queue, tail = tail, []
        while queue and len(running) < workers():
            out, args = queue.pop(0)
            if (out / "results.json").exists():
                continue
            p = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "train_smoke.py"), *args], cwd=ROOT, env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running[p] = (out, time.time())
            log(f"start {out.parent.name}/{out.name}")
        if not running and not queue and not finals_added:
            # screening incomplete but nothing running: a screening run failed
            log("screening incomplete with nothing running; stopping")
            break
        time.sleep(10)
    log("all done")


if __name__ == "__main__":
    main()

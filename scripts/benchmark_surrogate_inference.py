"""Inference cost of the released surrogate against the finite-element solve it
replaces.

Two measurements:

1. Solver cost. Every CalculiX solve behind the v1, v2 and v3 datasets wrote a
   ``solver.log`` ending in ``Total CalculiX Time: <seconds>``. That line is
   parsed for every case; the result is the solver alone and excludes meshing,
   so it understates the cost of the pipeline a surrogate replaces.
2. Surrogate cost, timed on the same class of hardware (CPU). A single network
   is timed on one case and on the 27-case v2 held-out partition as a batch;
   the ten-member ensemble is timed by actually running all ten checkpoints on
   that batch, not by multiplying a single model's time. Each figure is the
   median of repeated timings after warm-up.

The raw solver logs are not in the public repository. When they are absent
the script reads the per-case times already extracted from them in
outputs/surrogate/fem_solve_times.json, and times the surrogate afresh.
Surrogate timings depend on the machine; the committed JSON records the
hardware it was measured on.

Usage:
    python scripts/benchmark_surrogate_inference.py

Writes outputs/surrogate/inference_benchmark.json and
outputs/surrogate/fem_solve_times.json.
"""

from __future__ import annotations

import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.models import FNO2d  # noqa: E402
from gearstress.splits import grouped_trajectory_split  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
OUT = ROOT / "outputs" / "surrogate"
# case folder -> the processed dataset whose trajectories it must belong to
DATASETS = {"gear_pair_transient_quasistatic_dataset": "gear_pair_transient_quasistatic_v1",
            "gear_pair_transient_quasistatic_v2_dataset": "gear_pair_transient_quasistatic_v2",
            "gear_pair_transient_quasistatic_v3_dataset": "gear_pair_transient_quasistatic_v3"}
TOTAL = re.compile(r"Total CalculiX Time:\s*([0-9.]+)")
REPEATS = 7


def solver_times() -> list[dict]:
    """Solver time of every case that is in a processed dataset.

    Case folders can also hold aborted, unused trajectories; those are skipped
    so the figure describes exactly the solves behind the datasets.
    """
    rows = []
    for ds, processed in DATASETS.items():
        with h5py.File(ROOT / "data" / "processed" / f"{processed}.h5", "r") as f:
            keep = {f"trajectory_{int(t):03d}" for t in np.unique(f["trajectory_ids"][:])}
        for log in sorted((ROOT / "outputs" / ds).glob("trajectory_*/frame_*/solver.log")):
            if log.parent.parent.name not in keep:
                continue
            m = TOTAL.search(log.read_text(encoding="utf-8", errors="ignore"))
            if m:
                rows.append({"dataset": ds, "case": f"{log.parent.parent.name}/{log.parent.name}",
                             "seconds": float(m.group(1))})
    return rows


def timed(fn, warmup: int, iters: int) -> float:
    """Median wall-clock seconds per call over REPEATS blocks of `iters` calls."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        blocks = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            blocks.append((time.perf_counter() - t0) / iters)
    return statistics.median(blocks)


def main() -> None:
    fem = solver_times()
    OUT.mkdir(parents=True, exist_ok=True)
    extracted = OUT / "fem_solve_times.json"
    if fem:
        extracted.write_text(json.dumps(fem, indent=1), encoding="utf-8")
    elif extracted.exists():
        # Raw solver logs are not in the public repository; the per-case times
        # extracted from them are, so the solver column is still inspectable.
        fem = json.loads(extracted.read_text(encoding="utf-8"))
        print(f"solver logs absent; using {len(fem)} extracted times from {extracted.name}")
    else:
        raise FileNotFoundError("no solver.log files and no extracted fem_solve_times.json")
    secs = np.array([r["seconds"] for r in fem])

    with h5py.File(ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v2.h5", "r") as f:
        x = np.asarray(f["inputs"], dtype=np.float32)
        traj = np.asarray(f["trajectory_ids"])
    split = grouped_trajectory_split(traj, train_fraction=float(CFG["train_fraction"]),
                                     val_fraction=float(CFG["val_fraction"]), seed=42)
    batch = torch.from_numpy(x[split.test])
    one = batch[:1]

    runs = sorted((ROOT / "outputs" / "smoke" / "gear_pair_transient_quasistatic_v2").glob("fno_seed*_ep300"),
                  key=lambda p: int(re.search(r"_seed(\d+)_", p.name).group(1)))
    w = CFG["model"]
    models = []
    for run in runs:
        m = FNO2d(width=w["width"], modes_x=w["modes_x"], modes_y=w["modes_y"], layers=w["layers"])
        m.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
        m.eval()
        models.append(m)
    assert len(models) == 10, f"expected 10 v2 checkpoints, found {len(models)}"
    single = models[0]
    n = len(batch)

    t_single_one = timed(lambda: single(one), warmup=10, iters=50)
    t_single_batch = timed(lambda: single(batch), warmup=3, iters=10)
    t_ens_batch = timed(lambda: [m(batch) for m in models], warmup=2, iters=3)
    # one design query at a time, all ten members: the latency a user waits for
    t_ens_one = timed(lambda: [m(one) for m in models], warmup=5, iters=20)
    # the same unbatched query for every held-out case, to show the spread
    per_case = []
    with torch.no_grad():
        for k in range(n):
            q = batch[k:k + 1]
            for _ in range(2):
                [m(q) for m in models]
            reps = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                [m(q) for m in models]
                reps.append(time.perf_counter() - t0)
            per_case.append(1e3 * statistics.median(reps))

    solve = float(np.median(secs))
    per = {"single_network_one_case_ms": 1e3 * t_single_one,
           "single_network_batched_ms_per_case": 1e3 * t_single_batch / n,
           "ten_member_ensemble_batched_ms_per_case": 1e3 * t_ens_batch / n,
           "ten_member_ensemble_one_case_ms": 1e3 * t_ens_one,
           "ten_member_ensemble_one_case_ms_range_over_test_cases": [min(per_case), max(per_case)]}
    result = {
        "hardware": {"device": "cpu", "torch_threads": torch.get_num_threads(),
                     "processor": platform.processor(), "torch": torch.__version__},
        "fem_solver": {"n_solves": int(len(secs)), "median_s": solve, "mean_s": float(secs.mean()),
                       "p10_s": float(np.percentile(secs, 10)), "p90_s": float(np.percentile(secs, 90)),
                       "max_s": float(secs.max()), "scope": "CalculiX solver only; meshing excluded"},
        "surrogate": {"parameters": int(sum(p.numel() for p in single.parameters())),
                      "batch_size": n, **per},
        "speedup_vs_median_solve": {
            "single_network_one_case": solve / (per["single_network_one_case_ms"] / 1e3),
            "single_network_batched": solve / (per["single_network_batched_ms_per_case"] / 1e3),
            "ten_member_ensemble_batched": solve / (per["ten_member_ensemble_batched_ms_per_case"] / 1e3),
            "ten_member_ensemble_one_case": solve / (per["ten_member_ensemble_one_case_ms"] / 1e3)},
        "timing_method": f"median of {REPEATS} repeated blocks after warm-up",
    }
    (OUT / "inference_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"FEM solve  n={len(secs)}  median {solve:.1f}s  mean {secs.mean():.1f}s  "
          f"p90 {np.percentile(secs, 90):.1f}s")
    for k, v in per.items():
        print(f"{k:42s} {v} ms" if isinstance(v, list) else f"{k:42s} {v:8.2f} ms")
    for k, v in result["speedup_vs_median_solve"].items():
        print(f"speedup {k:32s} {v:10,.0f}x")
    print(f"threads={torch.get_num_threads()}  wrote {(OUT / 'inference_benchmark.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

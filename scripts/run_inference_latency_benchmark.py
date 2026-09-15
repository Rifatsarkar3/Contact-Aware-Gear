"""Inference-latency benchmark, closing MSSP's required-but-never-measured efficiency metric.

**Do NOT run this until GPU is free** -- see
docs/superpowers/specs/2026-08-04-uncertainty-guided-active-learning-design.md.

Times three deployment configurations on whatever device is available
(CPU and, if present, GPU): a single vanilla FNO forward pass, a 10-model
ensemble (sequential forward passes -- matches how the ensemble is
actually used elsewhere in this project, no claim of batched-parallel
ensembling unless the arms above also do that), and one MC-Dropout model
sampled MC_DROPOUT_SAMPLES times. Reports ms/sample and samples/sec for
each, with a 5-iteration warmup before timing (standard latency-
benchmarking practice -- excludes one-time CUDA kernel compilation/
caching effects from the measured numbers).

Usage (once launched):
    python scripts/run_inference_latency_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.models import FNO2d  # noqa: E402

CONFIG = ROOT / "configs" / "smoke.yaml"
OUT_ROOT = ROOT / "outputs" / "active_learning"
WARMUP_ITERS = 5
TIMED_ITERS = 50
BATCH_SIZE = 4
GRID_SIZE = 64
ENSEMBLE_SIZE = 10
MC_DROPOUT_SAMPLES = 20
MC_DROPOUT_RATE = 0.2

# Same rationale/value as run_active_learning_study.py -- another job may
# already be resident on the GPU; cap our footprint to 20% of total device
# memory so we can never grow into memory it needs.
GPU_MEMORY_FRACTION = 0.20


def time_forward_passes(forward_fn, device: torch.device) -> dict[str, float]:
    for _ in range(WARMUP_ITERS):
        forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(TIMED_ITERS):
        forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms_per_call = 1000.0 * elapsed / TIMED_ITERS
    samples_per_sec = (BATCH_SIZE * TIMED_ITERS) / elapsed
    return {"ms_per_call": ms_per_call, "ms_per_sample": ms_per_call / BATCH_SIZE, "samples_per_sec": samples_per_sec}


def benchmark_on_device(device: torch.device) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model_cfg = dict(cfg["model"])
    x = torch.randn(BATCH_SIZE, 8, GRID_SIZE, GRID_SIZE, device=device)

    single = FNO2d(width=model_cfg["width"], modes_x=model_cfg["modes_x"], modes_y=model_cfg["modes_y"], layers=model_cfg["layers"]).to(device).eval()

    @torch.no_grad()
    def single_forward():
        single(x)

    ensemble = [
        FNO2d(width=model_cfg["width"], modes_x=model_cfg["modes_x"], modes_y=model_cfg["modes_y"], layers=model_cfg["layers"]).to(device).eval()
        for _ in range(ENSEMBLE_SIZE)
    ]

    @torch.no_grad()
    def ensemble_forward():
        for m in ensemble:
            m(x)

    dropout_model = FNO2d(width=model_cfg["width"], modes_x=model_cfg["modes_x"], modes_y=model_cfg["modes_y"], layers=model_cfg["layers"], dropout=MC_DROPOUT_RATE).to(device)
    dropout_model.train()  # dropout must be active for MC sampling

    @torch.no_grad()
    def mc_dropout_forward():
        for _ in range(MC_DROPOUT_SAMPLES):
            dropout_model(x)

    return {
        "single_model": time_forward_passes(single_forward, device),
        "ensemble_10": time_forward_passes(ensemble_forward, device),
        "mc_dropout_20": time_forward_passes(mc_dropout_forward, device),
    }


def main() -> None:
    results = {"cpu": benchmark_on_device(torch.device("cpu"))}
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
        results["cuda"] = benchmark_on_device(torch.device("cuda"))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / "inference_latency_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()

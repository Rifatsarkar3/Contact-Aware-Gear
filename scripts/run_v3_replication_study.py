"""Replicate the core v1 comparison on v3, a second independent 36-sample draw.

`v1` (seed=42) showed AEE-FNO and DWM-FNO as clean, cost-free mechanisms;
`v2` (seed=7, 2.5x more data) showed both as significant regressions. That
leaves open whether v1 was simply an unlucky draw, or whether *any*
independently-sampled 36-training-sample dataset from this same FEM
generation pipeline would similarly fail to detect the effect. `v3`
(seed=13, same generator, same 7-trajectory/9-frame scale as v1, zero
overlap with v1 or v2) tests this directly: same 10-seed/300-epoch
protocol, same models, on a dataset nobody has looked at before this run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = ROOT / "scripts" / "train_smoke.py"
DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v3.h5"

SEEDS = list(range(1, 11))
EPOCHS = 300

MODELS = [
    {"key": "fno", "model": "fno", "extra": []},
    {"key": "aee_fno", "model": "aee_fno", "extra": []},
    {"key": "dwm_fno", "model": "dwm_fno", "extra": ["--local-window", "0.5", "--local-width", "8", "--local-size", "16"]},
]


def run_name_for(spec: dict, seed: int) -> str:
    name = spec["model"]
    if spec["model"] == "dwm_fno":
        name = f"{name}_lw8_lwin0.5_lsz16"
    return f"{name}_seed{seed}_ep{EPOCHS}"


def train(spec: dict, seed: int) -> Path:
    run_name = run_name_for(spec, seed)
    out_dir = ROOT / "outputs" / "smoke" / DATA.stem / run_name
    results_path = out_dir / "results.json"
    if results_path.exists():
        print(f"[skip] {run_name}")
        return results_path
    cmd = [
        PY, str(TRAIN),
        "--data", str(DATA),
        "--model", spec["model"],
        "--epochs", str(EPOCHS),
        "--allow-extended-training",
        "--seed", str(seed),
    ] + spec["extra"]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return results_path


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    m = mean(diffs)
    s = stdev(diffs) if n > 1 else 0.0
    se = s / (n**0.5) if n > 1 else 0.0
    t = m / se if se > 0 else float("inf")
    return {"mean_diff": m, "sd_diff": s, "t": t, "n": n}


def main() -> None:
    for seed in SEEDS:
        for spec in MODELS:
            train(spec, seed)

    runs_by_key: dict[str, dict[int, dict]] = {}
    for spec in MODELS:
        by_seed = {}
        for seed in SEEDS:
            results_path = ROOT / "outputs" / "smoke" / DATA.stem / run_name_for(spec, seed) / "results.json"
            data = json.loads(results_path.read_text(encoding="utf-8"))
            by_seed[seed] = data["test"]
        runs_by_key[spec["key"]] = by_seed

    baseline = runs_by_key["fno"]
    summary = {}
    for spec in MODELS:
        key = spec["key"]
        by_seed = runs_by_key[key]
        entry = {
            "n_seeds": len(by_seed),
            "relative_l2_mean": mean(v["relative_l2"] for v in by_seed.values()),
            "relative_l2_std": stdev(v["relative_l2"] for v in by_seed.values()),
            "hotspot_mae_mean": mean(v["hotspot_mae"] for v in by_seed.values()),
            "hotspot_mae_std": stdev(v["hotspot_mae"] for v in by_seed.values()),
            "equilibrium_residual_mean": mean(v["equilibrium_residual"] for v in by_seed.values()),
            "equilibrium_residual_std": stdev(v["equilibrium_residual"] for v in by_seed.values()),
        }
        if key != "fno":
            diffs_l2 = [by_seed[s]["relative_l2"] - baseline[s]["relative_l2"] for s in SEEDS]
            diffs_hot = [by_seed[s]["hotspot_mae"] - baseline[s]["hotspot_mae"] for s in SEEDS]
            diffs_eq = [by_seed[s]["equilibrium_residual"] - baseline[s]["equilibrium_residual"] for s in SEEDS]
            entry["paired_vs_fno"] = {
                "relative_l2": paired_stats(diffs_l2),
                "hotspot_mae": paired_stats(diffs_hot),
                "equilibrium_residual": paired_stats(diffs_eq),
            }
        summary[key] = entry

    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "v3_replication_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary written] {out_path}")
    for key, stats in summary.items():
        print(f"{key}: relative_l2 {stats['relative_l2_mean']:.4f} +/- {stats['relative_l2_std']:.4f}, "
              f"hotspot_mae {stats['hotspot_mae_mean']:.4f} +/- {stats['hotspot_mae_std']:.4f}")
        if "paired_vs_fno" in stats:
            print(json.dumps(stats["paired_vs_fno"], indent=2))


if __name__ == "__main__":
    main()

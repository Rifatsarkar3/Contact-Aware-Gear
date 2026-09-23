"""Re-run the core Stage M comparison on a larger, independently-sampled dataset.

User-directed: after nine structurally different mechanisms all gave clean or
reversed results on the 63-case ``gear_pair_transient_quasistatic_v1`` dataset
(36 training samples), the consistent conclusion across
`docs/EXTENDED_MULTISEED_STUDY_2026-07-17.md`, `docs/DCT_FNO_STUDY_2026-07-17.md`,
and `docs/CCM_FNO_STUDY_2026-07-24.md` was that training-data quantity, not
architecture, is the likely bottleneck. This script tests that hypothesis
directly: same 10-seed/300-epoch protocol, same models, but on
``gear_pair_transient_quasistatic_v2.h5`` -- 144 cases (16 trajectories x 9
phases, independently sampled with seed=7, not an extension of v1's seed=42
sampling), giving 90 training samples (2.5x v1's 36).

Model selection here is deliberately narrower than the original six-invention
sweep: vanilla FNO at both its inherited default capacity and the
capacity-matched width=32/modes=16 config the DCT-FNO study showed helps on
v1 (same values, so any change is attributable to data, not a new search);
Physics-FNO and AEE-FNO, Stage M's only two survivors; and LPM-FNO, whose
smoke-scale win reversed into the strongest regression found in the whole
campaign once trained properly on v1 -- the single most informative case to
re-check, since a genuine undertraining/data-starvation story would predict
its instability might behave differently with more data.
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
DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v2.h5"

SEEDS = list(range(1, 11))
EPOCHS = 300

MODELS = [
    {"key": "fno", "model": "fno", "width": None, "modes": None},
    {"key": "fno_w32", "model": "fno", "width": 32, "modes": 16},
    {"key": "fno_physics", "model": "fno_physics", "width": None, "modes": None},
    {"key": "aee_fno", "model": "aee_fno", "width": None, "modes": None},
    {"key": "lpm_fno", "model": "lpm_fno", "width": None, "modes": None},
]


def run_suffix(seed: int) -> str:
    return f"_seed{seed}_ep{EPOCHS}"


def run_name_for(spec: dict, seed: int) -> str:
    name = spec["model"]
    if spec["width"] is not None:
        name = f"{name}_w{spec['width']}_mx{spec['modes']}_my{spec['modes']}_l4"
    return f"{name}{run_suffix(seed)}"


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
    ]
    if spec["width"] is not None:
        cmd += ["--width", str(spec["width"]), "--modes-x", str(spec["modes"]), "--modes-y", str(spec["modes"])]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return results_path


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    m = mean(diffs)
    s = stdev(diffs) if n > 1 else 0.0
    se = s / (n ** 0.5) if n > 1 else 0.0
    t = m / se if se > 0 else float("inf")
    return {"mean_diff": m, "sd_diff": s, "t": t, "n": n}


def main() -> None:
    for seed in SEEDS:
        for spec in MODELS:
            train(spec, seed)

    runs_by_key: dict[str, list[dict]] = {}
    for spec in MODELS:
        runs = []
        for seed in SEEDS:
            results_path = ROOT / "outputs" / "smoke" / DATA.stem / run_name_for(spec, seed) / "results.json"
            data = json.loads(results_path.read_text(encoding="utf-8"))
            test = data["test"]
            runs.append(
                {
                    "seed": seed,
                    "relative_l2": test["relative_l2"],
                    "hotspot_mae": test["hotspot_mae"],
                    "equilibrium_residual": test["equilibrium_residual"],
                }
            )
        runs_by_key[spec["key"]] = runs

    baseline = runs_by_key["fno"]
    baseline_by_seed = {r["seed"]: r for r in baseline}

    summary = {}
    for spec in MODELS:
        key = spec["key"]
        runs = runs_by_key[key]
        entry = {
            "n_seeds": len(runs),
            "relative_l2_mean": mean(r["relative_l2"] for r in runs),
            "relative_l2_std": stdev(r["relative_l2"] for r in runs),
            "hotspot_mae_mean": mean(r["hotspot_mae"] for r in runs),
            "hotspot_mae_std": stdev(r["hotspot_mae"] for r in runs),
            "equilibrium_residual_mean": mean(r["equilibrium_residual"] for r in runs),
            "equilibrium_residual_std": stdev(r["equilibrium_residual"] for r in runs),
            "runs": runs,
        }
        if key != "fno":
            diffs_l2 = [r["relative_l2"] - baseline_by_seed[r["seed"]]["relative_l2"] for r in runs]
            diffs_hot = [r["hotspot_mae"] - baseline_by_seed[r["seed"]]["hotspot_mae"] for r in runs]
            diffs_eq = [r["equilibrium_residual"] - baseline_by_seed[r["seed"]]["equilibrium_residual"] for r in runs]
            entry["paired_vs_fno_default"] = {
                "relative_l2": paired_stats(diffs_l2),
                "hotspot_mae": paired_stats(diffs_hot),
                "equilibrium_residual": paired_stats(diffs_eq),
            }
        summary[key] = entry

    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "v2_multiseed_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary written] {out_path}")
    for key, stats in summary.items():
        print(
            f"{key}: relative_l2 {stats['relative_l2_mean']:.4f} +/- {stats['relative_l2_std']:.4f}, "
            f"hotspot_mae {stats['hotspot_mae_mean']:.4f} +/- {stats['hotspot_mae_std']:.4f}, "
            f"equilibrium {stats['equilibrium_residual_mean']:.2f} +/- {stats['equilibrium_residual_std']:.2f}"
        )
        if "paired_vs_fno_default" in stats:
            print(json.dumps(stats["paired_vs_fno_default"], indent=2))


if __name__ == "__main__":
    main()

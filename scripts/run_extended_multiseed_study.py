"""Diagnostic escalation beyond the 10-epoch/single-seed smoke gate.

User-directed follow-up to the six-mechanism comparison
(`docs/INVENTION_MECHANISM_SELECTION_2026-07-17.md`): the smoke gate's
10-epoch cap and single seed cannot distinguish a genuine field/hotspot
accuracy tradeoff from simple undertraining or seed noise, especially for
LPM-FNO's documented training instability. This script trains every
directly-trainable model from both invention families for 300 epochs across
10 independent seeds (model init/training seed only -- the train/val/test
trajectory split is always the config's seed-42 split, so every run is
evaluated on identical, leakage-free data) and aggregates mean/std test
metrics per model.

This remains a diagnostic run on the existing 63-case phase-dense dataset,
not new FEM evidence or a claim of final paper-grade statistics.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = ROOT / "scripts" / "train_smoke.py"
MICE = ROOT / "scripts" / "evaluate_mice_smoke.py"

SEEDS = list(range(1, 11))
EPOCHS = 300
DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v1.h5"

INDEPENDENT_MODELS = ["fno", "fno_physics", "aee_fno", "ccp_fno", "lpm_fno"]
DEPENDENT_MODELS = [("rcr_fno", "fno"), ("lpm_rcr_fno", "lpm_fno")]


def run_suffix(seed: int) -> str:
    return f"_seed{seed}_ep{EPOCHS}"


def train(model: str, seed: int) -> None:
    run_name = f"{model}{run_suffix(seed)}"
    out_dir = ROOT / "outputs" / "smoke" / DATA.stem / run_name
    if (out_dir / "results.json").exists():
        print(f"[skip] {run_name} already exists")
        return
    cmd = [
        PY, str(TRAIN),
        "--data", str(DATA),
        "--model", model,
        "--epochs", str(EPOCHS),
        "--allow-extended-training",
        "--seed", str(seed),
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def evaluate_mice(seed: int) -> None:
    run_name = f"mice_lpm{run_suffix(seed)}"
    out_dir = ROOT / "outputs" / "smoke" / DATA.stem / run_name
    if (out_dir / "results.json").exists():
        print(f"[skip] {run_name} already exists")
        return
    cmd = [
        PY, str(MICE),
        "--data", str(DATA),
        "--lpm-run-name", f"lpm_fno{run_suffix(seed)}",
        "--output-run-name", run_name,
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def aggregate() -> dict:
    families = {
        "fno": [],
        "fno_physics": [],
        "aee_fno": [],
        "ccp_fno": [],
        "rcr_fno": [],
        "lpm_fno": [],
        "lpm_rcr_fno": [],
        "mice_lpm": [],
    }
    for model in families:
        for seed in SEEDS:
            run_name = f"{model}{run_suffix(seed)}"
            results_path = ROOT / "outputs" / "smoke" / DATA.stem / run_name / "results.json"
            if not results_path.exists():
                continue
            data = json.loads(results_path.read_text(encoding="utf-8"))
            test = data["test"] if model != "mice_lpm" else data["test"]
            families[model].append(
                {
                    "seed": seed,
                    "relative_l2": test["relative_l2"],
                    "hotspot_mae": test["hotspot_mae"],
                    "equilibrium_residual": test["equilibrium_residual"],
                }
            )
    summary = {}
    for model, runs in families.items():
        if not runs:
            continue
        rel = [r["relative_l2"] for r in runs]
        hot = [r["hotspot_mae"] for r in runs]
        eq = [r["equilibrium_residual"] for r in runs]
        summary[model] = {
            "n_seeds": len(runs),
            "relative_l2_mean": mean(rel),
            "relative_l2_std": pstdev(rel) if len(rel) > 1 else 0.0,
            "hotspot_mae_mean": mean(hot),
            "hotspot_mae_std": pstdev(hot) if len(hot) > 1 else 0.0,
            "equilibrium_residual_mean": mean(eq),
            "equilibrium_residual_std": pstdev(eq) if len(eq) > 1 else 0.0,
            "runs": runs,
        }
    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "extended_multiseed_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary written] {out_path}")
    return summary


def main() -> None:
    for seed in SEEDS:
        for model in INDEPENDENT_MODELS:
            train(model, seed)
        for model, _baseline in DEPENDENT_MODELS:
            train(model, seed)
        evaluate_mice(seed)
    summary = aggregate()
    for model, stats in summary.items():
        print(
            f"{model}: relative_l2 {stats['relative_l2_mean']:.4f} +/- {stats['relative_l2_std']:.4f}, "
            f"hotspot_mae {stats['hotspot_mae_mean']:.4f} +/- {stats['hotspot_mae_std']:.4f}, "
            f"equilibrium {stats['equilibrium_residual_mean']:.2f} +/- {stats['equilibrium_residual_std']:.2f} "
            f"(n={stats['n_seeds']})"
        )


if __name__ == "__main__":
    main()

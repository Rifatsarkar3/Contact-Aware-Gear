"""Master orchestrator: run the entire uncertainty-guided active-learning study.

**Do NOT run this until GPU is free and the user gives an explicit
go-ahead** -- see
docs/superpowers/specs/2026-08-04-uncertainty-guided-active-learning-design.md
and docs/superpowers/plans/2026-08-04-uncertainty-guided-active-learning.md.

Runs, in order: the shared random-selection baseline; all three UQ arms
(ensemble_surrogate, mc_dropout, pool_based); a calibration check for each
UQ arm at every checkpoint; the inference-latency benchmark. Then
aggregates everything into one summary and prints a comparison table --
which arm wins at which sample budget, against random, with what
calibration and what latency cost.

This is the single command referred to throughout this project's planning
as "launch everything":

    python scripts/run_active_learning_full_study.py

Estimated scope (see the design spec): ~500+ new CalculiX solves across
the three UQ arms plus the shared random baseline, on top of everything
already run for the four-pronged validation study on 2026-08-03. Expect
this to take a long time -- run it in the background and check back
rather than waiting synchronously.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
STUDY = ROOT / "scripts" / "run_active_learning_study.py"
CALIBRATION = ROOT / "scripts" / "run_uq_calibration_check.py"
LATENCY = ROOT / "scripts" / "run_inference_latency_benchmark.py"
OUT_ROOT = ROOT / "outputs" / "active_learning"

ROUNDS = 6
CHECKPOINT_ROUNDS = (2, 4, 6)
UQ_METHODS = ("ensemble_surrogate", "mc_dropout", "pool_based")
ALL_METHODS = ("random_baseline",) + UQ_METHODS


def run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def paired_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    m = mean(diffs)
    s = stdev(diffs) if n > 1 else 0.0
    se = s / (n**0.5) if n > 1 else 0.0
    t = m / se if se > 0 else float("inf")
    return {"mean_diff": m, "sd_diff": s, "t": t, "n": n}


def main() -> None:
    # 1. Shared random baseline first -- every UQ arm's comparison depends on it.
    run([PY, str(STUDY), "--method", "random_baseline", "--rounds", str(ROUNDS)])

    # 2. Three competing UQ arms.
    for method in UQ_METHODS:
        run([PY, str(STUDY), "--method", method, "--rounds", str(ROUNDS)])

    # 3. Calibration check for each UQ arm at every checkpoint.
    for method in UQ_METHODS:
        for round_number in CHECKPOINT_ROUNDS:
            run([PY, str(CALIBRATION), "--method", method, "--round", str(round_number)])

    # 4. Inference latency benchmark.
    run([PY, str(LATENCY)])

    # 5. Aggregate everything into one comparison.
    random_summary = json.loads((OUT_ROOT / "random_baseline" / "full_summary.json").read_text(encoding="utf-8"))
    comparison = {}
    for method in UQ_METHODS:
        method_summary = json.loads((OUT_ROOT / method / "full_summary.json").read_text(encoding="utf-8"))
        comparison[method] = {}
        for round_number in CHECKPOINT_ROUNDS:
            key = f"round_{round_number}"
            method_round = method_summary[key]
            random_round = random_summary[key]
            method_by_seed = {r["seed"]: r for r in method_round["per_seed_results"]}
            random_by_seed = {r["seed"]: r for r in random_round["per_seed_results"]}
            shared_seeds = sorted(set(method_by_seed) & set(random_by_seed))
            diffs_l2 = [method_by_seed[s]["relative_l2"] - random_by_seed[s]["relative_l2"] for s in shared_seeds]
            diffs_hot = [method_by_seed[s]["hotspot_mae"] - random_by_seed[s]["hotspot_mae"] for s in shared_seeds]
            calibration_path = OUT_ROOT / method / f"round_{round_number}_calibration.json"
            calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else None
            comparison[method][key] = {
                "n_train": method_round["n_train"],
                "relative_l2_mean": method_round["relative_l2_mean"],
                "random_relative_l2_mean": random_round["relative_l2_mean"],
                "paired_vs_random": {
                    "relative_l2": paired_stats(diffs_l2),
                    "hotspot_mae": paired_stats(diffs_hot),
                },
                "calibration_spearman": calibration["spearman_correlation_variance_vs_error"] if calibration else None,
            }

    latency = json.loads((OUT_ROOT / "inference_latency_benchmark.json").read_text(encoding="utf-8"))

    full = {"comparison": comparison, "random_baseline": random_summary, "latency": latency}
    out_path = OUT_ROOT / "full_study_summary.json"
    out_path.write_text(json.dumps(full, indent=2), encoding="utf-8")
    print(f"\n[summary written] {out_path}\n")

    print(f"{'Method':<20}{'N':>6}{'Rel.L2':>10}{'vs random t':>14}{'Calibration r':>16}")
    for method in UQ_METHODS:
        for round_number in CHECKPOINT_ROUNDS:
            row = comparison[method][f"round_{round_number}"]
            t = row["paired_vs_random"]["relative_l2"]["t"]
            corr = row["calibration_spearman"]
            corr_str = f"{corr:.3f}" if corr is not None else "n/a"
            print(f"{method:<20}{row['n_train']:>6}{row['relative_l2_mean']:>10.4f}{t:>14.2f}{corr_str:>16}")

    print("\nA negative t means the UQ arm beats random at that sample budget (lower relative L2 is better).")
    print("Decision rule (per the design spec): the arm with the strongest combination of")
    print("(a) beating random with significance and (b) high calibration correlation becomes")
    print("the manuscript's primary contribution; the others are reported as comparative evidence.")


if __name__ == "__main__":
    main()

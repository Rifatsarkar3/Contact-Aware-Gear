"""Reflexive power analysis: does AL-selected data detect the known
AEE-FNO/DWM-FNO regression more reliably than randomly-selected data,
at the same N?

**Do NOT run until scripts/run_al_misranking_comparison.py has produced
results for both datasets** -- see
docs/superpowers/specs/2026-08-04-al-misranking-combined-study-design.md.

Same Monte Carlo methodology as scripts/run_posthoc_power_analysis.py,
applied here to the two reconstructed datasets instead of v1. The "true
effect" is estimated from v2 (independent of both -- no overlap with
either dataset's trajectories), combined with each dataset's own
observed noise (SD of its 10 paired seed differences) to simulate
200,000 synthetic 10-seed paired t-tests. This asks the paper's actual
question directly and quantitatively: at n=10, is pool_based's
retrospective power to detect AEE-FNO/DWM-FNO's regression higher than
random_baseline's?

Usage (once scripts/run_al_misranking_comparison.py has been run):
    python scripts/run_al_misranking_power_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "smoke" / "gear_pair_transient_quasistatic_v2"
COMPARISON_DIR = ROOT / "outputs" / "smoke" / "al_misranking_comparison"
SEEDS = list(range(1, 11))
T_CRIT = 2.262  # two-tailed, df=9, alpha=0.05
N_MC = 200_000
RNG = np.random.default_rng(0)

MECHANISMS = ["aee_fno", "dwm_fno"]
DATASET_KEYS = ["pool_based", "random_baseline", "ensemble_surrogate"]

# dwm_fno's v2 results directory includes its already-selected leakage-safe-
# search hyperparameters (see run_v2_remaining_mechanisms_study.py); other
# models here use their plain name. Matches run_posthoc_power_analysis.py's
# MECHANISMS dict for the same reason.
V2_DIR_STUB = {"fno": "fno", "aee_fno": "aee_fno", "dwm_fno": "dwm_fno_lw8_lwin0.5_lsz16"}


def load_v2_metric(model: str, metric: str) -> list[float]:
    stub = V2_DIR_STUB[model]
    values = []
    for seed in SEEDS:
        path = V2_DIR / f"{stub}_seed{seed}_ep300" / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        values.append(data["test"][metric])
    return values


def load_comparison_metric(dataset_key: str, model: str, metric: str) -> list[float]:
    values = []
    for seed in SEEDS:
        path = COMPARISON_DIR / dataset_key / f"{model}_seed{seed}_ep300" / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        values.append(data["test"][metric])
    return values


def paired_t(diffs: list[float]) -> dict:
    n = len(diffs)
    m = mean(diffs)
    s = stdev(diffs) if n > 1 else 0.0
    se = s / (n**0.5) if n > 1 else 0.0
    t = m / se if se > 0 else float("inf")
    return {"mean_diff": m, "sd_diff": s, "t": t, "n": n}


def mc_power(true_effect: float, noise_sd: float, n: int, n_mc: int = N_MC) -> float:
    if noise_sd <= 0:
        return 1.0 if abs(true_effect) > 0 else 0.0
    samples = RNG.normal(loc=true_effect, scale=noise_sd, size=(n_mc, n))
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    se = sds / (n**0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, means / se, np.inf * np.sign(means))
    return float(np.mean(np.abs(t) >= T_CRIT))


def required_n_for_power(true_effect: float, noise_sd: float, target_power: float = 0.80, n_max: int = 500) -> int | None:
    for n in range(3, n_max + 1):
        if mc_power(true_effect, noise_sd, n, n_mc=20_000) >= target_power:
            return n
    return None


def mde(noise_sd: float, n: int) -> float:
    z_alpha = 1.96
    z_beta = 0.84
    return (z_alpha + z_beta) * noise_sd / (n**0.5)


def main() -> None:
    report: dict = {}
    for mechanism in MECHANISMS:
        entry: dict = {}
        for metric in ("relative_l2", "hotspot_mae"):
            v2_model = load_v2_metric(mechanism, metric)
            v2_fno = load_v2_metric("fno", metric)
            v2_diffs = [m - f for m, f in zip(v2_model, v2_fno)]
            v2_stats = paired_t(v2_diffs)
            true_effect = v2_stats["mean_diff"]

            per_dataset = {}
            for dataset_key in DATASET_KEYS:
                model_vals = load_comparison_metric(dataset_key, mechanism, metric)
                fno_vals = load_comparison_metric(dataset_key, "fno", metric)
                diffs = [m - f for m, f in zip(model_vals, fno_vals)]
                stats = paired_t(diffs)
                noise_sd = stats["sd_diff"]
                per_dataset[dataset_key] = {
                    "observed": stats,
                    "minimum_detectable_effect_at_n10_80pct_power": mde(noise_sd, n=10),
                    "retrospective_power_n10_for_v2_sized_effect": mc_power(true_effect, noise_sd, n=10),
                    "seeds_needed_for_80pct_power_on_v2_effect": required_n_for_power(true_effect, noise_sd, target_power=0.80),
                }
            entry[metric] = {"v2_true_effect": v2_stats, "by_dataset": per_dataset}
        report[mechanism] = entry

    out_path = COMPARISON_DIR / "al_misranking_power_analysis.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[written] {out_path}\n")
    for mechanism, entry in report.items():
        print(f"== {mechanism} ==")
        for metric, stats in entry.items():
            print(f"  {metric}: v2 true effect mean_diff={stats['v2_true_effect']['mean_diff']:+.4f}")
            for dataset_key in DATASET_KEYS:
                d = stats["by_dataset"][dataset_key]
                print(
                    f"    {dataset_key}: observed t={d['observed']['t']:+.2f}, "
                    f"retrospective power for v2-sized effect={d['retrospective_power_n10_for_v2_sized_effect']*100:.1f}%, "
                    f"seeds needed for 80% power={d['seeds_needed_for_80pct_power_on_v2_effect']}"
                )
        print()


if __name__ == "__main__":
    main()

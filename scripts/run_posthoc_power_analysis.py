"""Retrospective power / minimum-detectable-effect analysis for the v1-to-v2 reversals.

`docs/DATASET_SCALE_STUDY_2026-07-24.md` found that AEE-FNO and DWM-FNO looked
like clean, cost-free mechanisms on the 36-training-sample dataset (v1) but
show statistically significant field-accuracy regressions on the 90-sample
dataset (v2). This script asks the natural follow-up question quantitatively,
using no new training runs: given v1's own seed-to-seed noise level (observed
SD of the paired per-seed difference at n=10 seeds), how likely was a v1-style
study to ever detect an effect the size of what v2 independently revealed?

Method: Monte Carlo retrospective power. The "true effect" is estimated from
v2 (an independent dataset, not the same data being tested here, so this
avoids the circular "observed power from the same sample" fallacy). v1's own
noise level (SD of its 10 paired seed differences) is combined with that
effect size to simulate 200,000 synthetic 10-seed paired t-tests, and the
fraction reaching significance (|t| >= 2.262, alpha=0.05, df=9) is reported
as v1's retrospective power. A minimum-detectable-effect (MDE) and a required
sample size for 80% power are reported alongside for a more standard,
non-circular framing.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1_DIR = ROOT / "outputs" / "smoke" / "gear_pair_transient_quasistatic_v1"
V2_DIR = ROOT / "outputs" / "smoke" / "gear_pair_transient_quasistatic_v2"
SEEDS = list(range(1, 11))
T_CRIT = 2.262  # two-tailed, df=9, alpha=0.05
N_MC = 200_000
RNG = np.random.default_rng(0)

MECHANISMS = {
    "aee_fno": {"v1_dir": "aee_fno", "v2_dir": "aee_fno"},
    "dwm_fno": {"v1_dir": "dwm_fno_lw8_lwin0.5_lsz16", "v2_dir": "dwm_fno_lw8_lwin0.5_lsz16"},
}


def load_metric(run_dir: Path, run_stub: str, metric: str) -> list[float]:
    values = []
    for seed in SEEDS:
        path = run_dir / f"{run_stub}_seed{seed}_ep300" / "results.json"
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
    """Fraction of simulated n-seed paired t-tests reaching |t| >= T_CRIT."""
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
    """Minimum detectable effect at alpha=0.05, ~80% power, normal approximation."""
    z_alpha = 1.96
    z_beta = 0.84
    return (z_alpha + z_beta) * noise_sd / (n**0.5)


def main() -> None:
    report: dict = {}
    for key, dirs in MECHANISMS.items():
        entry: dict = {}
        for metric in ("relative_l2", "hotspot_mae"):
            v1_model = load_metric(V1_DIR, dirs["v1_dir"], metric)
            v1_fno = load_metric(V1_DIR, "fno", metric)
            v2_model = load_metric(V2_DIR, dirs["v2_dir"], metric)
            v2_fno = load_metric(V2_DIR, "fno", metric)

            v1_diffs = [m - f for m, f in zip(v1_model, v1_fno)]
            v2_diffs = [m - f for m, f in zip(v2_model, v2_fno)]

            v1_stats = paired_t(v1_diffs)
            v2_stats = paired_t(v2_diffs)

            true_effect = v2_stats["mean_diff"]
            v1_noise_sd = v1_stats["sd_diff"]

            power_v1_given_v2_effect = mc_power(true_effect, v1_noise_sd, n=10)
            v1_mde = mde(v1_noise_sd, n=10)
            n_needed = required_n_for_power(true_effect, v1_noise_sd, target_power=0.80)

            entry[metric] = {
                "v1_observed": v1_stats,
                "v2_observed": v2_stats,
                "v1_minimum_detectable_effect_at_n10_80pct_power": v1_mde,
                "v2_effect_vs_v1_mde_ratio": (abs(true_effect) / v1_mde) if v1_mde else None,
                "retrospective_power_v1_n10_for_v2_sized_effect": power_v1_given_v2_effect,
                "seeds_needed_at_v1_noise_for_80pct_power_on_v2_effect": n_needed,
            }
        report[key] = entry

    out_path = V1_DIR / "posthoc_power_analysis.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[written] {out_path}\n")
    for key, entry in report.items():
        print(f"== {key} ==")
        for metric, stats in entry.items():
            print(f"  {metric}:")
            print(f"    v1 observed: mean_diff={stats['v1_observed']['mean_diff']:+.4f} sd={stats['v1_observed']['sd_diff']:.4f} t={stats['v1_observed']['t']:+.2f}")
            print(f"    v2 observed: mean_diff={stats['v2_observed']['mean_diff']:+.4f} sd={stats['v2_observed']['sd_diff']:.4f} t={stats['v2_observed']['t']:+.2f}")
            print(f"    v1 minimum detectable effect (n=10, 80% power): {stats['v1_minimum_detectable_effect_at_n10_80pct_power']:.4f}")
            print(f"    v2 effect / v1 MDE ratio: {stats['v2_effect_vs_v1_mde_ratio']:.2f}" if stats["v2_effect_vs_v1_mde_ratio"] is not None else "    v2 effect / v1 MDE ratio: n/a")
            print(f"    retrospective power of v1's n=10 protocol to catch a v2-sized effect: {stats['retrospective_power_v1_n10_for_v2_sized_effect']*100:.1f}%")
            print(f"    seeds needed at v1's noise level for 80% power: {stats['seeds_needed_at_v1_noise_for_80pct_power_on_v2_effect']}")
        print()


if __name__ == "__main__":
    main()

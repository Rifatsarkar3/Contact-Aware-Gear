"""Disentangle seed count from data quantity for the v1-to-v2 reversals.

`docs/DATASET_SCALE_STUDY_2026-07-24.md` found AEE-FNO and DWM-FNO look like
clean, cost-free mechanisms at v1's 10-seed protocol but show significant
regressions on v2 (2.5x more training data, same 10 seeds). That leaves an
obvious alternative explanation unresolved: maybe v1 just needed more seeds,
not more data. `scripts/run_posthoc_power_analysis.py` estimated v1 would
need ~29 seeds (AEE-FNO) / ~132 seeds (DWM-FNO) for 80% power against a
v2-sized effect, using v1's own noise level. This script tests that directly,
empirically: extend vanilla FNO, AEE-FNO, and DWM-FNO on v1 (still 36
training samples, zero new FEM data) from 10 to 30 seeds, and recompute the
paired significance test at n=30. If the regression remains undetectable at
30 seeds on the *same* small dataset, that rules out "just needed more
seeds" and confirms the effect is specifically about training-data quantity.
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
DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v1.h5"

EXISTING_SEEDS = list(range(1, 11))
NEW_SEEDS = list(range(11, 31))
ALL_SEEDS = EXISTING_SEEDS + NEW_SEEDS
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
    for seed in NEW_SEEDS:
        for spec in MODELS:
            train(spec, seed)

    runs_by_key: dict[str, dict[int, dict]] = {}
    for spec in MODELS:
        by_seed = {}
        for seed in ALL_SEEDS:
            results_path = ROOT / "outputs" / "smoke" / DATA.stem / run_name_for(spec, seed) / "results.json"
            data = json.loads(results_path.read_text(encoding="utf-8"))
            test = data["test"]
            by_seed[seed] = {
                "relative_l2": test["relative_l2"],
                "hotspot_mae": test["hotspot_mae"],
                "equilibrium_residual": test["equilibrium_residual"],
            }
        runs_by_key[spec["key"]] = by_seed

    baseline = runs_by_key["fno"]

    summary = {}
    for spec in MODELS:
        key = spec["key"]
        by_seed = runs_by_key[key]
        if key == "fno":
            continue
        for n_label, seed_subset in (("n10", EXISTING_SEEDS), ("n20_new_only", NEW_SEEDS), ("n30_pooled", ALL_SEEDS)):
            diffs_l2 = [by_seed[s]["relative_l2"] - baseline[s]["relative_l2"] for s in seed_subset]
            diffs_hot = [by_seed[s]["hotspot_mae"] - baseline[s]["hotspot_mae"] for s in seed_subset]
            diffs_eq = [by_seed[s]["equilibrium_residual"] - baseline[s]["equilibrium_residual"] for s in seed_subset]
            summary.setdefault(key, {})[n_label] = {
                "seeds": seed_subset,
                "relative_l2": paired_stats(diffs_l2),
                "hotspot_mae": paired_stats(diffs_hot),
                "equilibrium_residual": paired_stats(diffs_eq),
            }

    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "v1_seed_disentangling_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary written] {out_path}")
    for key, by_n in summary.items():
        print(f"== {key} ==")
        for n_label, stats in by_n.items():
            print(f"  {n_label} (n={stats['relative_l2']['n']}): "
                  f"rel_l2 diff={stats['relative_l2']['mean_diff']:+.4f} t={stats['relative_l2']['t']:+.2f}  "
                  f"hotspot diff={stats['hotspot_mae']['mean_diff']:+.4f} t={stats['hotspot_mae']['t']:+.2f}")


if __name__ == "__main__":
    main()

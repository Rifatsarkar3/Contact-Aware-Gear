"""Re-test the eight mechanisms not yet covered by the v2 data-scale study.

`docs/DATASET_SCALE_STUDY_2026-07-24.md` re-tested vanilla FNO (default and
width=32), Physics-FNO, AEE-FNO, and LPM-FNO on the independently-sampled,
2.5x larger `gear_pair_transient_quasistatic_v2.h5` dataset. This script
covers the remaining nine-minus-one mechanisms that were null or reversed on
`v1` (`docs/CCM_FNO_STUDY_2026-07-24.md` and predecessors): RCR-FNO,
CCP-FNO, LPM-RCR-FNO, MICE-LPM, DCT-FNO, CCM-FNO, HGM-FNO, DWM-FNO.

Scope decision, disclosed rather than hidden: for DCT-FNO, CCM-FNO,
HGM-FNO, and DWM-FNO, this reuses each mechanism's already-selected
architecture from its `v1` leakage-safe search (DCT-FNO: width=32/modes=16;
CCM-FNO: window=0.5/local_width=16; HGM-FNO: window=0.25/local_width=24;
DWM-FNO: window=0.5/local_width=8) rather than re-running a fresh
leakage-safe architecture search on `v2`. This directly answers "does this
already-selected architecture behave differently with more data" -- it does
not answer "would a `v2`-specific search have found something different."
That is a real, separate open question, not addressed here.

RCR-FNO and LPM-RCR-FNO require frozen fno/lpm_fno checkpoints at the same
seed to repair; MICE-LPM requires a frozen lpm_fno checkpoint. All of these
already exist on `v2` from the prior study, so no redundant training is
needed for the baselines.
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
MICE = ROOT / "scripts" / "evaluate_mice_smoke.py"
DATA = ROOT / "data" / "processed" / "gear_pair_transient_quasistatic_v2.h5"

SEEDS = list(range(1, 11))
EPOCHS = 300

SIMPLE_MODELS = ["rcr_fno", "ccp_fno", "lpm_rcr_fno"]

CAPACITY_MODELS = [
    {"key": "dct_fno", "model": "dct_fno", "width": 32, "modes": 16},
]

LOCAL_WINDOW_MODELS = [
    {"key": "ccm_fno", "model": "ccm_fno", "local_window": 0.5, "local_width": 16, "local_size": 16},
    {"key": "hgm_fno", "model": "hgm_fno", "local_window": 0.25, "local_width": 24, "local_size": 16},
    {"key": "dwm_fno", "model": "dwm_fno", "local_window": 0.5, "local_width": 8, "local_size": 16},
]


def run_suffix(seed: int) -> str:
    return f"_seed{seed}_ep{EPOCHS}"


def run_name_simple(model: str, seed: int) -> str:
    return f"{model}{run_suffix(seed)}"


def run_name_capacity(spec: dict, seed: int) -> str:
    return f"{spec['model']}_w{spec['width']}_mx{spec['modes']}_my{spec['modes']}_l4{run_suffix(seed)}"


def run_name_local(spec: dict, seed: int) -> str:
    return f"{spec['model']}_lw{spec['local_width']}_lwin{spec['local_window']:g}_lsz{spec['local_size']}{run_suffix(seed)}"


def train_simple(model: str, seed: int) -> Path:
    run_name = run_name_simple(model, seed)
    out_dir = ROOT / "outputs" / "smoke" / DATA.stem / run_name
    results_path = out_dir / "results.json"
    if results_path.exists():
        print(f"[skip] {run_name}")
        return results_path
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
    return results_path


def train_capacity(spec: dict, seed: int) -> Path:
    run_name = run_name_capacity(spec, seed)
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
        "--width", str(spec["width"]),
        "--modes-x", str(spec["modes"]),
        "--modes-y", str(spec["modes"]),
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return results_path


def train_local(spec: dict, seed: int) -> Path:
    run_name = run_name_local(spec, seed)
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
        "--local-window", str(spec["local_window"]),
        "--local-width", str(spec["local_width"]),
        "--local-size", str(spec["local_size"]),
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return results_path


def evaluate_mice(seed: int) -> Path:
    run_name = f"mice_lpm{run_suffix(seed)}"
    out_dir = ROOT / "outputs" / "smoke" / DATA.stem / run_name
    results_path = out_dir / "results.json"
    if results_path.exists():
        print(f"[skip] {run_name}")
        return results_path
    cmd = [
        PY, str(MICE),
        "--data", str(DATA),
        "--lpm-run-name", f"lpm_fno{run_suffix(seed)}",
        "--output-run-name", run_name,
    ]
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


def collect(results_path_fn) -> list[dict]:
    runs = []
    for seed in SEEDS:
        results_path = results_path_fn(seed)
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
    return runs


def main() -> None:
    for seed in SEEDS:
        for model in SIMPLE_MODELS:
            train_simple(model, seed)
        for spec in CAPACITY_MODELS:
            train_capacity(spec, seed)
        for spec in LOCAL_WINDOW_MODELS:
            train_local(spec, seed)
        evaluate_mice(seed)

    baseline_path = ROOT / "outputs" / "smoke" / DATA.stem / "fno_seed1_ep300" / "results.json"
    assert baseline_path.parent.parent.exists(), "run run_v2_multiseed_study.py first for the fno baseline"
    baseline_runs = collect(lambda seed: ROOT / "outputs" / "smoke" / DATA.stem / f"fno{run_suffix(seed)}" / "results.json")
    baseline_by_seed = {r["seed"]: r for r in baseline_runs}

    summary = {}
    all_specs: list[tuple[str, object]] = (
        [(m, lambda seed, m=m: ROOT / "outputs" / "smoke" / DATA.stem / run_name_simple(m, seed) / "results.json") for m in SIMPLE_MODELS]
        + [(s["key"], lambda seed, s=s: ROOT / "outputs" / "smoke" / DATA.stem / run_name_capacity(s, seed) / "results.json") for s in CAPACITY_MODELS]
        + [(s["key"], lambda seed, s=s: ROOT / "outputs" / "smoke" / DATA.stem / run_name_local(s, seed) / "results.json") for s in LOCAL_WINDOW_MODELS]
        + [("mice_lpm", lambda seed: ROOT / "outputs" / "smoke" / DATA.stem / f"mice_lpm{run_suffix(seed)}" / "results.json")]
    )
    for key, path_fn in all_specs:
        runs = collect(path_fn)
        diffs_l2 = [r["relative_l2"] - baseline_by_seed[r["seed"]]["relative_l2"] for r in runs]
        diffs_hot = [r["hotspot_mae"] - baseline_by_seed[r["seed"]]["hotspot_mae"] for r in runs]
        diffs_eq = [r["equilibrium_residual"] - baseline_by_seed[r["seed"]]["equilibrium_residual"] for r in runs]
        summary[key] = {
            "n_seeds": len(runs),
            "relative_l2_mean": mean(r["relative_l2"] for r in runs),
            "relative_l2_std": stdev(r["relative_l2"] for r in runs),
            "hotspot_mae_mean": mean(r["hotspot_mae"] for r in runs),
            "hotspot_mae_std": stdev(r["hotspot_mae"] for r in runs),
            "equilibrium_residual_mean": mean(r["equilibrium_residual"] for r in runs),
            "equilibrium_residual_std": stdev(r["equilibrium_residual"] for r in runs),
            "runs": runs,
            "paired_vs_fno_default": {
                "relative_l2": paired_stats(diffs_l2),
                "hotspot_mae": paired_stats(diffs_hot),
                "equilibrium_residual": paired_stats(diffs_eq),
            },
        }

    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "v2_remaining_mechanisms_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary written] {out_path}")
    for key, stats in summary.items():
        print(
            f"{key}: relative_l2 {stats['relative_l2_mean']:.4f} +/- {stats['relative_l2_std']:.4f}, "
            f"hotspot_mae {stats['hotspot_mae_mean']:.4f} +/- {stats['hotspot_mae_std']:.4f}, "
            f"equilibrium {stats['equilibrium_residual_mean']:.2f} +/- {stats['equilibrium_residual_std']:.2f}"
        )
        print(json.dumps(stats["paired_vs_fno_default"], indent=2))


if __name__ == "__main__":
    main()

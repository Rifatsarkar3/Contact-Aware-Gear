"""Does the v2 misranking-reversal finding hold under a different train/val/test
split, or is it specific to the one partition (split seed 42) used everywhere
else in this paper?

Every multi-seed comparison in this project randomizes only the model-init
seed; the trajectory split itself is always fixed at seed 42 for
cross-run comparability (see train_smoke.py's --seed docstring). An ARS
Stage-3 methodology review flagged this as an open robustness question: the
headline v2 result (vanilla FNO significantly beats AEE-FNO and DWM-FNO on
relative_l2, paired t-test n=10 seeds, at the 144-case v2 scale) has only
ever been tested against that one partition.

This script re-runs the same three models (fno, aee_fno, dwm_fno) at the
same capacity-matched hyperparameters, 10 model-init seeds each, 300 epochs,
on v2 -- but with the trajectory split re-seeded (SPLIT_SEED=123 instead of
the config default 42). If the paired t-tests (fno vs aee_fno, fno vs
dwm_fno) still reject at alpha=0.05 under this different partition, that is
real evidence the finding isn't an artifact of one particular train/test
split. If they don't, that is reported honestly too.
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
SPLIT_SEED = 123

MODELS = ["fno", "aee_fno", "dwm_fno"]
DWM_ARGS = ["--local-width", "8", "--local-window", "0.5", "--local-size", "16"]


def run_name_for(model: str, seed: int) -> str:
    base = model
    if model == "dwm_fno":
        base = f"{model}_lw8_lwin0.5_lsz16"
    return f"{base}_splitseed{SPLIT_SEED}_seed{seed}_ep{EPOCHS}"


def train(model: str, seed: int) -> Path:
    run_name = run_name_for(model, seed)
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
        "--split-seed", str(SPLIT_SEED),
    ]
    if model == "dwm_fno":
        cmd += DWM_ARGS
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
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found")

    for model in MODELS:
        for seed in SEEDS:
            train(model, seed)

    runs_by_model: dict[str, list[dict]] = {}
    for model in MODELS:
        runs = []
        for seed in SEEDS:
            results_path = ROOT / "outputs" / "smoke" / DATA.stem / run_name_for(model, seed) / "results.json"
            data = json.loads(results_path.read_text(encoding="utf-8"))
            test = data["test"]
            runs.append({"seed": seed, "relative_l2": test["relative_l2"], "hotspot_mae": test["hotspot_mae"]})
        runs_by_model[model] = runs

    baseline_by_seed = {r["seed"]: r for r in runs_by_model["fno"]}

    summary: dict = {"split_seed": SPLIT_SEED, "epochs": EPOCHS, "n_seeds": len(SEEDS)}
    for model in MODELS:
        runs = runs_by_model[model]
        entry = {
            "relative_l2_mean": mean(r["relative_l2"] for r in runs),
            "relative_l2_std": stdev(r["relative_l2"] for r in runs),
            "hotspot_mae_mean": mean(r["hotspot_mae"] for r in runs),
            "hotspot_mae_std": stdev(r["hotspot_mae"] for r in runs),
            "runs": runs,
        }
        if model != "fno":
            diffs_l2 = [r["relative_l2"] - baseline_by_seed[r["seed"]]["relative_l2"] for r in runs]
            entry["paired_vs_fno"] = {"relative_l2": paired_stats(diffs_l2)}
        summary[model] = entry

    out_path = ROOT / "outputs" / "smoke" / DATA.stem / "split_seed_robustness_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary written] {out_path}")
    for model in MODELS:
        s = summary[model]
        print(f"{model}: relative_l2 {s['relative_l2_mean']:.4f} +/- {s['relative_l2_std']:.4f}")
        if "paired_vs_fno" in s:
            print(json.dumps(s["paired_vs_fno"], indent=2))


if __name__ == "__main__":
    main()

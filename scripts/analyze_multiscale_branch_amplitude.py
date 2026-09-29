"""How much do the trained multiscale local branches actually contribute?

CCM-FNO, HGM-FNO and DWM-FNO each add a zero-initialized local branch to an
FNO global path, so every seed starts exactly at the global path and any
departure has to be learned. This measures, on the v2 held-out test set, the
root-mean-square size of the learned local correction relative to the global
path's output: output(model) - output(model with the local branch's final layer
zeroed), divided by the RMS of the latter. One ratio per seed; the summary
reports the mean over the ten final-validation seeds.

Usage:
    python scripts/analyze_multiscale_branch_amplitude.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_surrogate_accuracy import CFG, load_split  # noqa: E402
from gearstress.inventions import (  # noqa: E402
    ContactCenteredMultiscaleFNO,
    DualWindowMultiscaleFNO,
    HotspotGuidedMultiscaleFNO,
)

RUNS = ROOT / "outputs" / "smoke" / "gear_pair_transient_quasistatic_v2"
OUT = RUNS / "multiscale_branch_amplitude.json"
MODELS = {
    "ccm_fno": ("ccm_fno_lw16_lwin0.5_lsz16", ContactCenteredMultiscaleFNO,
                dict(local_width=16, local_window=0.5, local_size=16, gate_sigma_fraction=0.25), ["local_net"]),
    "hgm_fno": ("hgm_fno_lw24_lwin0.25_lsz16", HotspotGuidedMultiscaleFNO,
                dict(local_width=24, local_window=0.25, local_size=16, gate_sigma_fraction=0.25,
                     softmax_temperature=0.1), ["local_net"]),
    "dwm_fno": ("dwm_fno_lw8_lwin0.5_lsz16", DualWindowMultiscaleFNO,
                dict(local_width=8, local_window=0.5, local_size=16, gate_sigma_fraction=0.25,
                     softmax_temperature=0.1), ["contact_local_net", "hotspot_local_net"]),
}


def rms(t: torch.Tensor) -> float:
    return float(t.square().mean().sqrt())


def main() -> None:
    data = load_split("gear_pair_transient_quasistatic_v2", 42)
    w = CFG["model"]
    summary = {}
    for name, (run_prefix, cls, kwargs, branches) in MODELS.items():
        ratios = []
        for seed in range(1, 11):
            model = cls(width=w["width"], modes_x=w["modes_x"], modes_y=w["modes_y"], layers=w["layers"], **kwargs)
            state = torch.load(RUNS / f"{run_prefix}_seed{seed}_ep300" / "model.pt", map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                full = model(data["x"])
                for b in branches:
                    last = getattr(model, b)[-1]
                    last.weight.zero_()
                    last.bias.zero_()
                global_only = model(data["x"])
            ratios.append(rms(full - global_only) / rms(global_only))
        summary[name] = {"per_seed_ratio": ratios, "mean_ratio": float(np.mean(ratios)),
                         "min_ratio": float(np.min(ratios)), "max_ratio": float(np.max(ratios)),
                         "run_prefix": run_prefix}
        print(f"{name}: local branch RMS = {100 * np.mean(ratios):.2f}% of the global path "
              f"(range {100 * np.min(ratios):.2f}-{100 * np.max(ratios):.2f}% over 10 seeds)")
    summary["_metadata"] = {"dataset": "gear_pair_transient_quasistatic_v2", "split_seed": 42,
                            "n_test_cases": data["n_test_cases"],
                            "definition": "RMS(full - global-only) / RMS(global-only) over the test set, all pixels"}
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

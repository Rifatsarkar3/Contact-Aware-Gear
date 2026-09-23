"""Build and solve one local involute-tooth contact case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.fem import ContactCase, build_case, contact_element_counts, parse_final_nodal_stress, run_calculix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "fem_smoke" / "case_000")
    parser.add_argument("--mesh-size", type=float, default=0.65)
    parser.add_argument("--indentation", type=float, default=0.025)
    parser.add_argument(
        "--ccx",
        type=Path,
        default=ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe",
    )
    args = parser.parse_args()
    case = ContactCase(mesh_size_mm=args.mesh_size, indentation_mm=args.indentation)
    deck = build_case(case, args.output)
    result = run_calculix(deck, args.ccx)
    print(f"CalculiX exit code: {result.returncode}")
    print(result.stdout[-2000:])
    if result.returncode != 0 or not deck.with_suffix(".frd").exists():
        raise SystemExit("FEM smoke solve failed; inspect solver.log and case.sta")
    counts = contact_element_counts(deck.parent / "solver.log")
    if not counts or max(counts) <= 0:
        raise SystemExit("FEM solve completed without active contact elements")
    stress = parse_final_nodal_stress(deck.with_suffix(".frd"))
    peak = max(float(abs(values[[0, 1, 3]]).max()) for values in stress.values())
    if peak <= 0:
        raise SystemExit("FEM solve produced a zero stress field")
    print(f"Active contact elements (max): {max(counts)}; peak in-plane nodal stress: {peak:.3f} MPa")



if __name__ == "__main__":
    main()

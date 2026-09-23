"""One-off correctness gate for the plate-hole FEM backend: build a circular
hole case (a=b, Kt_theory=3.0, Kirsch's classical result) and a couple of
elliptical cases, run CalculiX, and check the measured Kt against Inglis's
closed-form formula before any dataset generation is trusted."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gearstress.fem import run_calculix  # noqa: E402
from gearstress.plate_hole_fem import (  # noqa: E402
    PlateHoleCase,
    build_plate_hole_case,
    measure_stress_concentration,
)

CCX = ROOT / "tools" / "CalculiX-2.23.0-win-x64" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe"
OUT = ROOT / "outputs" / "plate_hole_validation"


def check(case: PlateHoleCase, name: str) -> None:
    case_dir = OUT / name
    deck = build_plate_hole_case(case, case_dir)
    result = run_calculix(deck, CCX, timeout_seconds=120)
    if result.returncode != 0 or not (case_dir / "case.frd").exists():
        print(f"[{name}] CalculiX FAILED (returncode={result.returncode})")
        print((case_dir / "solver.log").read_text(encoding="utf-8", errors="replace")[-2000:])
        return
    m = measure_stress_concentration(case_dir)
    err_pct = 100.0 * abs(m["kt_fem"] - m["kt_theory"]) / m["kt_theory"]
    print(
        f"[{name}] a/b={case.aspect_ratio:.2f}  Kt_theory={m['kt_theory']:.3f}  "
        f"Kt_FEM={m['kt_fem']:.3f}  error={err_pct:.2f}%  "
        f"(peak={m['peak_syy_mpa']:.2f} MPa, farfield={m['farfield_syy_mpa']:.2f} MPa)"
    )


if __name__ == "__main__":
    check(PlateHoleCase(a_mm=3.0, b_mm=3.0), "circular_kt3")
    check(PlateHoleCase(a_mm=6.0, b_mm=3.0), "ellipse_ar2")
    check(PlateHoleCase(a_mm=9.0, b_mm=3.0), "ellipse_ar3")

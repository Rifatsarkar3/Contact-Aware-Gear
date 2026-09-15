# Misranked by Small Samples

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22761338.svg)](https://doi.org/10.5281/zenodo.22761338)

Code, datasets, and complete result logs for:

> **Misranked by Small Samples: Underpowered Architecture Comparisons in
> Physics-Informed Neural Operator Learning for Stress-Field Prediction**
>
> Tao Zhang, Mohammad Raziul Hasan Rifat Sarkar\*
> Faculty of Mechanical and Material Engineering, Huai'an University (HAU),
> Huai'an 223003, Jiangsu, China
> \*Corresponding author

**Status: under review at *Neural Networks*.** This release is archived on
Zenodo with a citable DOI (see [Citation](#citation)); the article citation
will be added on acceptance.

## What this paper claims

Architecture comparisons in physics-informed neural operator learning are
routinely run at the sample sizes nonlinear finite-element analysis
realistically supplies, tens to low hundreds of cases rather than thousands.
At that scale, **such comparisons can rank mechanisms wrongly**. The true
effect is not absent; the sample simply lacks the statistical power to detect
it.

The demonstration is direct rather than inferential:

- **Ten structurally distinct architectural interventions** are tested under a
  leakage-safe, capacity-matched, multi-seed protocol. All regress or add
  nothing. So do a classical proper-orthogonal-decomposition baseline, a
  U-Net, and a from-scratch DeepONet given its own leakage-safe
  hyperparameter search.
- **One mechanism survives**: a minimal, loss-only physics-consistency penalty
  (Physics-FNO), which cuts a Fourier Neural Operator's equilibrium residual
  22-32% across two datasets at no accuracy cost.
- **Replication at 2.5x scale exposes the misranking.** AEE-FNO, reported as
  clean at the smaller scale, carries a real accuracy cost that scale lacked
  the power to detect. DWM-FNO shows the same pattern but fails a split-seed
  partition test, and is reported as unreplicated rather than as a second
  confirmed case.
- **Data scale is the reliable lever, not architecture.** With zero
  architecture change, the same FNO baseline's field error more than halves
  and its hotspot error improves sixfold on 2.5x more data.
- **Active learning has a hidden statistical cost.** Uncertainty-guided
  acquisition lowers prediction error 6-13% across three architectures at
  matched labeling budget, a genuine gain, yet the same actively-selected data
  makes the paper's own misranking check *less* statistically reliable, shown
  via two independently built active-learning systems.

Transient gear-tooth contact is the testbed, with a plate-with-elliptical-hole
pilot as a second domain. The evidence comes from one finite-element pipeline
sampled three times, not from independent physical systems, and the paper says
so in its abstract, contributions, and limitations.

## Quick start

```bash
git clone https://github.com/Rifatsarkar3/Contact-Aware-Gear.git
cd Contact-Aware-Gear

# Python >= 3.11. With uv (uv.lock is committed for an exact environment):
uv sync --extra dev
uv run pytest            # 47 passed, 1 skipped

# or with pip:
pip install -e ".[dev]"
pytest
```

The one skipped test (`test_grid_solved_frame_reads_an_already_solved_case_without_recalculix`)
reads a real CalculiX-solved case directory to check the solver-output reader
without invoking CalculiX. That fixture is 84 MB of raw solver output and is
not published here for the same reason the rest of the raw output is not; the
test skips cleanly in its absence.

Training and dataset generation are driven from `scripts/`. Every script's
own module docstring states which study in the manuscript it implements, so
`head -30 scripts/<name>.py` is the fastest way to find the entry point you
want.

Note that regenerating the datasets from scratch requires
[CalculiX](http://www.calculix.de/) and [Gmsh](https://gmsh.info/) on your
PATH. The processed datasets in `data/processed/` are committed here
precisely so that the machine-learning results can be reproduced **without**
re-running any finite-element solve.

## Repository layout

| Path | Contents |
|---|---|
| `src/gearstress/` | The package. FEM case-building and CalculiX I/O (`fem.py`, `plate_hole_fem.py`), models (`models.py`), physics losses (`losses.py`), the architectural mechanisms under test (`inventions.py`), active-learning acquisition strategies (`active_learning.py`), leakage-safe dataset splitting (`splits.py`), and an analytic debug proxy (`proxy.py`). |
| `scripts/` | All 47 dataset-generation, training, study, and analysis entry points used to produce the manuscript's results. |
| `tests/` | Unit and shape tests for the pipeline and the active-learning code. 47 run anywhere; 1 needs a solver-output fixture and skips without it. |
| `configs/` | Reproducible experiment settings. |
| `data/processed/` | Model-ready HDF5 datasets (see below). |
| `results/` | Every per-seed result log and study summary the manuscript's tables and figures were computed from (2,205 JSON files). |

## Datasets (`data/processed/`)

All datasets are generated by the scripts in this repository from CalculiX
finite-element solves. None are downloaded from a third-party source. Each
`.h5` file follows the same convention: an `inputs` array `(N, C, H, W)`, a
`targets` array `(N, 3, H, W)` of stress components, and a `conditions` array
of per-case scalars.

| File | Role in the paper |
|---|---|
| `gear_pair_transient_quasistatic_v1.h5` | Primary dataset **v1**: 63 cases across seven trajectories. |
| `gear_pair_transient_quasistatic_v2.h5` | Independently sampled **v2**: 144 cases across sixteen trajectories, 2.5x v1. The replication that exposes the misranking. |
| `gear_pair_transient_quasistatic_v3.h5` | Third independent replication (**v3**), one of the five statistical checks. |
| `gear_pair_geometry_v1.h5` | 120-case geometry-varying dataset for the out-of-distribution check (Section 5.10). |
| `gear_pair_al_pool_based_extended.h5`, `gear_pair_al_ensemble_surrogate_extended.h5`, `gear_pair_al_random_baseline_extended.h5` | Reconstructed active-learning datasets underlying Sections 5.7-5.8 and the reflexive power analysis. |
| `plate_hole_small_v1.h5`, `plate_hole_large_v1.h5` | Second-domain plate-with-elliptical-hole pilot (Section 6.2). |
| `gear_pair_smoke_v1.h5`, `fem_smoke_v0.h5`, `analytic_proxy_v0.h5` | Small smoke/debug fixtures used by the test suite. |

## Results (`results/`)

`results/` mirrors the pipeline's own output tree. The directories that map
most directly onto the manuscript:

| Path | Used for |
|---|---|
| `results/smoke/gear_pair_transient_quasistatic_v1/` | v1 multi-seed runs, including the 30-seed control. |
| `results/smoke/gear_pair_transient_quasistatic_v2/` | v2 multi-seed runs and the split-seed robustness check. |
| `results/active_learning/` | All four acquisition arms (pool-based, ensemble-surrogate, MC-dropout, random baseline) with per-checkpoint per-seed logs. |
| `results/smoke/al_misranking_comparison/` | The active-learning reflexive-power analysis. |
| `results/gear_pair_penalty_convergence/` | Contact-penalty-stiffness convergence sweep. |
| `results/gear_pair_mesh_convergence/` | Mesh-convergence study. |
| `results/plate_hole_results/` | Second-domain pilot results. |

## What is deliberately not in this repository

Two classes of artifact are too large for GitHub and are available from the
corresponding author on reasonable request:

- **Trained-model checkpoints** (~3.5 GB across 750 runs).
- **Raw CalculiX solver output** (~10 GB of intermediate `.12d`, `.inp`,
  `.msh`, `.frd`, and `.npz` files).

Neither is needed to reproduce the paper's analysis. The processed datasets
and the complete result logs in this repository are sufficient to recompute
every table and figure.

## Reproducibility notes

- The evaluation protocol is leakage-safe by construction: screening seeds are
  disjoint from final-validation seeds, and hyperparameter searches for any
  mechanism are run only on screening seeds. See `src/gearstress/splits.py`
  and Section 3.6 of the manuscript.
- Capacity is matched explicitly where a mechanism changes model width or mode
  count. DCT-FNO in particular must be compared against a capacity-matched FNO
  retrain (`fno_w32`), not the default-capacity baseline. Comparing it against
  the default baseline reproduces neither the manuscript's numbers nor the
  correct conclusion.
- CalculiX solves in this pipeline are deterministic: re-solving the same
  `.inp` deck reproduces the same `.frd` output.

## License

- **Code** (`src/`, `scripts/`, `tests/`, `configs/`): MIT, see [LICENSE](LICENSE).
- **Datasets and result logs** (`data/`, `results/`): Creative Commons
  Attribution 4.0 International (CC BY 4.0), see [DATA_LICENSE](DATA_LICENSE).

## Citation

This release is archived on Zenodo. Cite the deposit as:

> Zhang, T., & Sarkar, M. R. H. R. (2026). *Misranked by Small Samples: code,
> datasets and result logs for underpowered architecture comparisons in
> physics-informed neural operator learning* (Version 1.0.0) [Computer
> software]. Zenodo. https://doi.org/10.5281/zenodo.22761338

- **Version DOI** (this exact release, use for reproducibility):
  [`10.5281/zenodo.22761338`](https://doi.org/10.5281/zenodo.22761338)
- **Concept DOI** (always resolves to the latest version):
  [`10.5281/zenodo.22761337`](https://doi.org/10.5281/zenodo.22761337)

See [CITATION.cff](CITATION.cff) for machine-readable metadata. The
accompanying manuscript is under review at *Neural Networks*; this section
will be updated with the article citation on acceptance.

## Funding

Supported by the 2025 Huai'an City Basic Research Program (Joint Special
Project, Grant No. HABL202501).

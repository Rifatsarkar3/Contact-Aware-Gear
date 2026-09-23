# Contact-Aware Gear

Code, datasets, trained models and result logs for a neural-operator surrogate
of the transient stress field in a meshing gear tooth: a Fourier Neural
Operator trained on nonlinear frictional contact solutions from CalculiX and
evaluated under a leakage-safe, trajectory-grouped protocol.

## Headline results

Every figure below is produced by a script in this repository, and the
committed output is under `outputs/surrogate/`. All accuracy figures are on
held-out trajectories of the 144-case v2 dataset.

| Result | Value | Produced by |
|---|---|---|
| Field accuracy, 10-seed ensemble + void mask | R² = 0.9978, relative L2 = 0.0480 | `evaluate_surrogate_accuracy.py` |
| Same, single network (mean over 10 seeds) | R² = 0.9951, relative L2 = 0.0901 | `evaluate_surrogate_accuracy.py` |
| Label discretization uncertainty (0.12 mm vs 0.08 mm, n = 8) | relative L2 = 0.0440 (mean) | `measure_label_uncertainty.py` |
| Surrogate error relative to that floor | 1.09x | ratio of the two rows above |
| Finite-element solve (median of 270) | 25.3 s per case | `benchmark_surrogate_inference.py` |
| 10-member ensemble, CPU, batched | 27.8 ms per case, about 900-fold faster | `benchmark_surrogate_inference.py` |
| Peak root-fillet von Mises error, best configuration | 4.11%, under-predicting by 2.62% | `evaluate_surrogate_accuracy.py` |
| Ensembling gain at 90 vs 36 training cases | 43.2% vs 18.9% | `evaluate_surrogate_accuracy.py` |
| Data-scaling law over 36 to 90 training cases | relative L2 = 8.55 N^-1.03, R² = 0.94 | `evaluate_surrogate_accuracy.py` |

Read the table with its limits:

- **The field error sits at the labels' own floor.** 0.0480 is 1.09 times the
  discrepancy between the 0.12 mm training labels and a 0.08 mm re-solve.
  Pushing the field error much lower against these labels would mean fitting
  mesh artifacts, not gear physics. Peak root-fillet stress is different: its
  label uncertainty is a median 0.9%, so that metric still has real headroom.
- **Peak root stress is under-predicted**, by 5.35% for vanilla FNO and 2.62%
  for AEE-FNO on average. For a bending-fatigue quantity that is the
  non-conservative direction; deployment would need a calibration offset or a
  safety margin.
- **One tooth geometry.** The headline numbers are for a single geometry under
  randomized loading. Withholding an entire pressure-angle band costs 19% in
  median error (`outputs/smoke/gear_pair_geometry_v1/`).
- **Three test trajectories.** Complete trajectories are the unit of the
  leakage-safe split, so the v2 held-out partition is 27 cases from 3
  independent trajectories, not 27 independent cases.
- **The speedup is not circular.** Every network input (material mask,
  coordinates, four scalar load conditions, and a contact-location map computed
  analytically from involute geometry) exists before the solve.
- **Computational only.** No experimental strain-gauge validation.

## Reproduce the headline numbers

About a minute on a CPU, using the 20 checkpoints shipped in this repository:

```bash
git clone https://github.com/Rifatsarkar3/Contact-Aware-Gear.git
cd Contact-Aware-Gear
uv sync --extra dev

uv run python scripts/evaluate_surrogate_accuracy.py     # accuracy, ensembling, root fillet, scaling law
uv run python scripts/benchmark_surrogate_inference.py   # inference cost against the solver
```

`evaluate_surrogate_accuracy.py` reproduces the v2 field accuracy, the v2
ensembling gain, every root-fillet figure and the scaling law exactly. The v1,
v3 and alternate-partition ensembling results and the Physics-FNO ensemble
need checkpoints that are not shipped (see below); the script reports those
blocks as unavailable rather than failing. The committed
`surrogate_accuracy.json` was produced with all checkpoints present.

`benchmark_surrogate_inference.py` times the surrogate afresh on your machine,
so its millisecond figures will differ from the committed ones, which record
the hardware they were measured on; repeated runs on one machine differed by
about 20%. Solver times come from the per-case values
extracted from the original CalculiX logs into
`outputs/surrogate/fem_solve_times.json`.

`measure_label_uncertainty.py` re-solves eight cases at two mesh sizes and
takes about 20 minutes. It needs [CalculiX](http://www.calculix.de/): pass its
path with `--ccx`, or put `ccx` on your PATH. Gmsh comes in through the Python
dependencies. It samples only cases that belong to the v2 dataset,
and records any case whose solve exceeds its time limit as dropped rather than
skipping it silently; all eight complete.

## Tests

```bash
uv run pytest            # 47 passed, 1 skipped
# or: pip install -e ".[dev]" && pytest
```

The skipped test reads raw CalculiX output (about 84 MB) that is not
published here; it skips cleanly without it.

## Repository layout

| Path | Contents |
|---|---|
| `src/gearstress/` | The package: finite-element case building and CalculiX I/O (`fem.py`, `plate_hole_fem.py`), models (`models.py`), physics losses (`losses.py`), the alternative architectural mechanisms evaluated in this study (`inventions.py`), active-learning acquisition (`active_learning.py`), leakage-safe splitting (`splits.py`), and an analytic debug proxy (`proxy.py`). |
| `scripts/` | All 49 dataset-generation, training, study and analysis entry points. Each module docstring states which study it implements. |
| `tests/` | Unit and shape tests for the pipeline and the active-learning code. |
| `configs/` | Experiment settings. |
| `data/processed/` | Model-ready HDF5 datasets. |
| `outputs/` | Every per-seed result log, study summary and surrogate result, plus the 20 headline checkpoints. Laid out exactly as the scripts write it, so they run from a fresh clone. |

## Datasets (`data/processed/`)

| File | Contents |
|---|---|
| `gear_pair_transient_quasistatic_v2.h5` | Primary dataset: 144 converged contact cases, 16 trajectories x 9 phases. The headline results use it. |
| `gear_pair_transient_quasistatic_v1.h5` | 63 cases, 7 trajectories, a smaller independent draw from the same generator. |
| `gear_pair_transient_quasistatic_v3.h5` | 63 cases, a second independent small draw. |
| `gear_pair_geometry_v1.h5` | 120 cases with randomized pressure angle and root clearance, for the geometry study. |
| `gear_pair_al_*_extended.h5` | Datasets reconstructed from each active-learning acquisition strategy. |
| `plate_hole_*_v1.h5` | Plate-with-elliptical-hole second-domain pilot. |
| remaining files | Early smoke-test datasets from pipeline development. |

Every file holds `inputs` (8 x 64 x 64), `targets` (sigma_xx, sigma_yy, sigma_xy
on a 64 x 64 grid) and `conditions`. The gear datasets also hold
`trajectory_ids`, which the leakage-safe split groups on; the plate-hole files
do not, because each of their cases is an independent solve. The geometry
dataset additionally records `pressure_angle_deg` and `root_clearance_module`.

## Outputs (`outputs/`)

| Path | Contents |
|---|---|
| `outputs/surrogate/` | Headline accuracy, label uncertainty, inference cost and per-case solver times |
| `outputs/smoke/gear_pair_transient_quasistatic_v2/` | v2 multi-seed runs, the 20 shipped checkpoints, the split-seed check and the training-size curve |
| `outputs/smoke/gear_pair_transient_quasistatic_v1/`, `..._v3/` | v1 including the 30-seed control, and the v3 replication |
| `outputs/gear_pair_penalty_convergence/` | Contact-penalty convergence |
| `outputs/gear_pair_mesh_convergence/` | Mesh-convergence study behind the production element size |
| `outputs/active_learning/` | Active-learning arms with per-checkpoint, per-seed logs |
| `outputs/smoke/al_misranking_comparison/` | Active-learning reflexive power analysis |
| `outputs/smoke/gear_pair_transient_quasistatic_v2/unet*`, `deeponet*` | U-Net and DeepONet baselines |
| `outputs/smoke/gear_pair_geometry_v1/` | Geometry generalization |
| `outputs/plate_hole_results/` | Second-domain pilot: plate with an elliptical hole |

Ten alternative architectural mechanisms were also evaluated under the same
protocol. None improved on the baseline; their per-seed logs are in
`outputs/smoke/`.

## What is deliberately not in this repository

Available on reasonable request:

- **The remaining trained checkpoints**, about 3.4 GB across roughly 730 runs,
  including those behind the v1, v3 and alternate-partition ensembling results.
- **Raw CalculiX output**, about 10 GB of `.frd`, `.inp`, `.msh`, `.12d` and
  `.npz` files. The solver's own per-case wall-clock times are extracted into
  `outputs/surrogate/fem_solve_times.json`.

## Reproducibility notes

- **Leakage-safe split.** Complete finite-element trajectories, never single
  frames, go to train, validation or test. The target scale is fitted on the
  training partition only. See `src/gearstress/splits.py`.
- **Capacity is matched** wherever a compared mechanism changes width or mode
  count.
- **CalculiX is deterministic here.** Re-solving the same cases in two
  separate runs reproduced every value to four decimal places.

## License

- **Code** (`src/`, `scripts/`, `tests/`, `configs/`): MIT, see [LICENSE](LICENSE).
- **Datasets, result logs and trained checkpoints** (`data/`, `outputs/`):
  CC BY 4.0, see [DATA_LICENSE](DATA_LICENSE).

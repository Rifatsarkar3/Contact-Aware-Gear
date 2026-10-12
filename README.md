# Contact-Aware Gear

Code, datasets, trained models and result logs for a neural-operator surrogate
of the in-plane stress field in a spur-gear tooth under quasi-static frictional
contact: a ten-member Fourier Neural Operator ensemble trained on nonlinear
contact solutions from CalculiX and evaluated on held-out loading trajectories.

## Version 3.1.0: material mask, robustness checks and matched baselines

- **Material mask.** The grid mask of the gear-mesh datasets was the convex hull
  of the tooth nodes (2074 grid points). It also covered 242 points
  beside each lower flank and under the rim arc, where the labels held values
  interpolated across the gap between flank and rim. The datasets v4 and v4f now
  keep the 1832 grid points inside a finite element of the driving tooth
  (`fem.element_material_mask`, `fem.gear_mesh_material_mask`), with zero targets
  elsewhere. The finite-element solutions are unchanged; the datasets were
  re-projected, every network was retrained and every analysis repeated. The
  ensemble error against the labels changed from 0.0140 to 0.0097, and the change of
  the labels under the 1.5-fold refinement from 0.0900 to 0.0556. The version 3.0.0
  results are kept in `outputs/surrogate/gear_mesh_v4_results_hullmask.json`. The
  earlier datasets v1 to v3 keep their convex-hull mask.
- **Partitions and unseen positions.** Five further trajectory-grouped partitions
  of v4f (`splits.grouped_trajectory_fold_split`), ten networks each, so that
  every trajectory is tested at least once: ensemble error 0.0057 to 0.0231 over
  the six partitions. Six additional trajectories (54 cases,
  `gear_mesh_quasistatic_v4_extra.h5`), with contact positions between the nine
  training positions and an approach drawn at every position: 0.0195 for the
  reference ensemble.
- **Finite-element checks** (`run_fe_checks.py`, `analyze_fe_checks.py`): the
  contact status of all 432 cases of v4f from the tangential slip against the
  elastic stick limit; a third discretization of the 27 test cases (0.06 mm
  elements, 160-vertex flank), under which the labels change by 0.0741; the root
  peak on 128 x 128 and 256 x 256 grids; and the finite-element contact width
  against the Hertz width.
- **Matched comparison on v4f** (`run_extended_training.py`,
  `analyze_extended_study.py`): FNO, Physics-FNO, U-Net, DeepONet and a
  POD-DeepONet (trunk fixed to the POD modes of the training fields), each with
  six screened configurations (two seeds, 100 epochs) and the selected one
  retrained with ten seeds for 300 epochs, and a POD baseline. Mean
  single-network error on the 27 test cases: FNO 0.0167, Physics-FNO 0.0170,
  POD-DeepONet 0.0238, U-Net 0.0262; the DeepONet did not fit the fields
  (0.60).
- **Earlier comparisons** on v1 and v2 re-evaluated from the stored checkpoints
  under the full-grid and the material-point metric (`analyze_earlier_variants.py`).
- **Figure legends** now give the training cases of the two ensembles (90 and 270).

## Version 3.0.0: corrected finite-element model

The headline results now come from a corrected gear-mesh model
(`fem.GearMeshCase`, `fem.build_gear_mesh_case`) and its datasets v4 and v4f.
Two errors in the earlier tooth-pair model (`fem.build_gear_pair_case`,
`fem.build_gear_pair_transient_case`, used for datasets v1 to v3) were found
and corrected:

- **Load direction.** The earlier model displaced the mating rim along the line
  joining the closest vertices of the two faceted flank polygons. That line
  deviates from the flank normal by up to about 90 degrees, so the load and the
  direction of friction depended on the flank discretization; with a 120-vertex
  flank the stress changed by about 50% on average
  (`analyze_contact_load_path.py`, `summarize_contact_load_path.py`). The
  corrected model places the teeth in conjugate mesh and loads them by a normal
  approach along the line of action plus a tangential slip with the sign of the
  kinematic sliding, which reverses at the pitch point.
- **Contact-node adjustment.** The earlier model used the CalculiX `ADJUST`
  option, which moves every contact node within a set distance of the opposing
  flank onto it before loading. For two involute flanks this closes the
  curvature gap over a few tenths of a millimeter and turns the Hertzian contact
  into a wide, mesh-dependent one. The corrected model starts from point contact
  with no adjustment.

The earlier datasets v1 to v3 stay in this repository for the method
comparisons (FNO variants, Physics-FNO, U-Net, DeepONet, active learning,
geometry variation, training-set size). Those comparisons rank learning methods
on the same data; their absolute stresses carry the earlier model's errors.

## Headline results

Unless a row says otherwise, accuracy figures are on the 27 test cases of the three
held-out trajectories of the reference partition.
The committed outputs are under `outputs/surrogate/`.

| Result | Value | Produced by |
|---|---|---|
| Field accuracy, ten-member ensemble trained on v4f (270 of its 432 cases) | relative L2 = 0.0097, R² = 0.9999 | `analyze_gear_mesh_surrogate.py` |
| Same, single network (mean over ten seeds) | relative L2 = 0.0165 | same |
| Same, ensemble trained on v4 only (90 of its 144 cases) | relative L2 = 0.0573 | same |
| Ensemble on six trajectory-grouped partitions | relative L2 = 0.0057 to 0.0231 | `analyze_extended_study.py` |
| Ensemble on six additional trajectories (unseen positions and approaches) | relative L2 = 0.0195 | same |
| Labels against references with every length scale refined 1.5-fold and 2-fold | relative L2 = 0.0556 and 0.0741 | `analyze_gear_mesh_surrogate.py` |
| Peak root tensile stress (max. principal), surrogate against labels | 0.5% mean absolute error | same |
| Friction response, mu = 0.04 to 0.12 | FE field changes 8.7%, surrogate 8.7%; sign of the root-stress change correct in 27 of 27 cases | same |
| Contact: FE peak pressure against Hertz line contact, all 432 cases | 0.98 to 1.09 | same |
| Contact: resultant tangential force / (mu x normal force), sliding cases | 0.96 to 1.00 | same |
| Contact penalty 50E against 200E per mm | load within 1.2%, root stress within 1.2%, field within 0.027 | `check_gear_mesh_penalty.py` |
| CalculiX solve, median of nine run one at a time | 27.1 s per case | `time_gear_mesh_solves.py` |
| Ten-member ensemble, one case on CPU | 76 ms, about 360-fold faster | `benchmark_surrogate_inference.py` |

Read the table with its limits:

- **The labels are the accuracy limit.** Refining every length scale 1.5-fold
  and 2-fold changes the labels by 0.056 and 0.074 in relative L2, about
  6 and 8 times the surrogate's error against them; 42% of the
  1.5-fold difference lies within 1 mm of the contact point and nearly all of the
  rest next to the sharp root corner. The labels are not converged there.
- **The contact is not represented on the prediction grid.** The contact
  spans one to two pixels of the 64 x 64 grid and a few flank segments of the
  finite-element polygon. The surrogate predicts tooth-body stress, not contact
  pressure or subsurface stress.
- **Model simplifications.** Sharp root corner (no trochoidal fillet) on a
  clamped rim, a constant Coulomb coefficient without a lubricant film, and
  sliding represented by one imposed slip. Absolute root stresses are model
  values, not design stresses.
- **One tooth geometry.** Whole trajectories are the unit of every split, so
  the 27 test cases of a partition are not independent. The largest partition
  error occurred where the test loads exceeded those of the training
  trajectories; use beyond the trained load range is not established.
- **The speedup is not circular.** Every network input (material mask,
  coordinates, approach, friction coefficient, modulus, contact position and a
  contact-location map computed from involute geometry) exists before the solve.
- **Computational only.** No experimental validation.

## Reproduce the headline numbers

About a minute on a CPU, with the 20 checkpoints shipped for v4 and v4f:

```bash
git clone https://github.com/Rifatsarkar3/Contact-Aware-Gear.git
cd Contact-Aware-Gear
uv sync --extra dev

uv run python scripts/analyze_gear_mesh_surrogate.py    # accuracy, refined reference, root stress, friction, contact
uv run python scripts/generate_figures.py               # figures of the gear-mesh study
uv run python scripts/generate_comparison_figures.py    # figures of the comparisons on v1 to v3
uv run python scripts/benchmark_surrogate_inference.py  # network timing on your machine
```

`analyze_gear_mesh_surrogate.py` reproduces `gear_mesh_v4_results.json` value
for value from the shipped datasets, refined solutions and checkpoints.

The robustness and comparison analyses of version 3.1.0:

```bash
uv run python scripts/generate_extended_figures.py      # figures from the shipped result files
uv run python scripts/analyze_fe_checks.py              # finite-element checks from the shipped solutions
uv run python scripts/run_fe_checks.py --workers 8      # the solves behind them (CalculiX, resumable)
uv run python scripts/run_extended_training.py          # every training run of the partitions and the comparison
uv run python scripts/analyze_extended_study.py         # needs the checkpoints written by the previous step
```
Figures are written to `outputs/figures/`.

The finite-element steps need [CalculiX](http://www.calculix.de/) (pass its
path with `--ccx` or put `ccx` on your PATH); Gmsh comes in through the Python
dependencies.

```bash
# 492 solves: 144 cases at their own friction, the same cases at mu = 0.04 and 0.12,
# and the refined references of the 27 test cases (about 1.2 h with six workers)
for k in 0 1 2 3 4 5; do uv run python scripts/generate_gear_mesh_dataset.py --workers 6 --worker-id $k & done; wait
uv run python scripts/generate_gear_mesh_dataset.py --assemble

uv run python scripts/check_gear_mesh_penalty.py        # contact-penalty sensitivity
uv run python scripts/time_gear_mesh_solves.py          # solver time, one solve at a time

# one of the ten networks; seeds 1 to 10 for each dataset
uv run python scripts/train_smoke.py --data data/processed/gear_mesh_quasistatic_v4f.h5 \
    --model fno --seed 1 --epochs 300 --allow-extended-training
uv run python scripts/measure_training_cost.py --dataset gear_mesh_quasistatic_v4f
```

`benchmark_surrogate_inference.py` times the networks afresh, so its
millisecond figures differ from machine to machine; repeated runs on one
machine agreed to within about 15%. The networks it times have the same
architecture and size as the gear-mesh surrogate. `training_cost_v2f.json`
records a timed 20-epoch run on a 432-case dataset of the same grid, from which
the ten-network training time (about 3 h on the benchmark CPU) is extrapolated.

`analyze_comparison_statistics.py` (ensembling over material pixels, confidence
intervals of the Physics-FNO cost and the scaling exponent) needs checkpoints
of v1 and v3 that are not shipped; the committed `comparison_statistics.json`
was produced with all checkpoints present.

## Tests

```bash
uv run pytest            # 56 passed, 1 skipped
```

The skipped test reads raw CalculiX output that is not published here; it
skips cleanly without it.

## Repository layout

| Path | Contents |
|---|---|
| `src/gearstress/` | The package: finite-element case building and CalculiX I/O (`fem.py`, including the corrected `GearMeshCase`; `plate_hole_fem.py`), models (`models.py`), physics losses (`losses.py`), the alternative architectural mechanisms (`inventions.py`), active-learning acquisition (`active_learning.py`), leakage-safe splitting (`splits.py`), and an analytic debug proxy (`proxy.py`). |
| `scripts/` | All 67 dataset-generation, training, study, analysis and figure entry points. Each module docstring states what it does. |
| `tests/` | Unit and shape tests for the pipeline and the active-learning code. |
| `configs/` | Experiment settings. |
| `data/processed/` | Model-ready HDF5 datasets. |
| `outputs/` | Result logs, study summaries, solver outputs and the shipped checkpoints, laid out as the scripts write them. |

## Datasets (`data/processed/`)

| File | Contents |
|---|---|
| `gear_mesh_quasistatic_v4.h5` | Corrected model: 144 cases, 16 trajectories x 9 contact positions in the single-tooth contact zone. |
| `gear_mesh_quasistatic_v4f.h5` | v4 plus every case re-solved at mu = 0.04 and 0.12 (432 cases, same partition). The surrogate is trained on it. |
| `gear_mesh_quasistatic_v4_extra.h5` | Six additional test trajectories (54 cases) with contact positions between the nine of v4 and an approach drawn at every position; test only. |
| `gear_pair_transient_quasistatic_v1.h5`, `_v2.h5`, `_v3.h5` | Earlier model: 63, 144 and 63 cases, used for the method comparisons. |
| `gear_pair_geometry_v1.h5` | Earlier model, 120 cases with varied pressure angle and root clearance. |
| `gear_pair_al_*_extended.h5` | Datasets reconstructed from each active-learning acquisition strategy. |
| `plate_hole_*_v1.h5` | Plate-with-elliptical-hole second-domain pilot. |
| remaining files | Early smoke-test datasets from pipeline development. |

Every file holds `inputs` (8 x 64 x 64), `targets` (sigma_xx, sigma_yy, sigma_xy
on a 64 x 64 grid) and `conditions`. The gear datasets also hold
`trajectory_ids`, which the split groups on. The gear-mesh files name their
condition columns in the file attributes, and v4f adds `friction_variant`.

## Outputs (`outputs/`)

| Path | Contents |
|---|---|
| `outputs/surrogate/gear_mesh_v4_*.json` | Gear-mesh results, contact metrics of every solve, penalty sensitivity |
| `outputs/surrogate/gear_mesh_solve_time_idle.json`, `inference_benchmark.json`, `training_cost*.json` | Solver time, network timing, training time |
| `outputs/gear_mesh_v4_dataset/` | Every gear-mesh solve on the prediction grid with its contact metrics: production cases (`base_*`), 0.08 mm elements (`fine_*`) and 0.08 mm elements with a 120-vertex flank (`fineflank_*`) |
| `outputs/smoke/gear_mesh_quasistatic_v4/`, `..._v4f/` | The 20 headline gear-mesh networks (checkpoints and per-seed logs), and the per-seed logs of every run of the partitions, the screening and the matched comparison |
| `outputs/extended_study/` | `extended_results.json` (partitions, additional trajectories, matched comparison, POD, error against distance from the contact), `model_selection.json` (screening), `earlier_variants.json` (earlier comparisons under both metrics), `fe_summary.json` and `fe/` (finite-element checks: contact status and solutions at three discretizations) |
| `outputs/surrogate/gear_mesh_v4_results_hullmask.json` | Results of version 3.0.0 (convex-hull mask) |
| `outputs/surrogate/contact_load_path*`, `refined_flank_test/`, `hertz_contact_check.json` | Diagnosis of the earlier model's load path |
| `outputs/surrogate/comparison_statistics.json` | Supplementary statistics of the method comparisons |
| `outputs/smoke/gear_pair_transient_quasistatic_v1/`, `_v2/`, `_v3/` | Multi-seed runs on the earlier datasets, including the ten FNO variants, Physics-FNO, U-Net and DeepONet, the split-seed check and the training-size curve |
| `outputs/active_learning/` | Active-learning arms with per-checkpoint, per-seed logs |
| `outputs/smoke/al_misranking_comparison/` | Active-learning power analysis |
| `outputs/smoke/gear_pair_geometry_v1/` | Geometry variation |
| `outputs/gear_pair_penalty_convergence/`, `gear_pair_mesh_convergence/` | Penalty and mesh convergence of the earlier model |
| `outputs/plate_hole_results/` | Second-domain pilot: plate with an elliptical hole |
| `outputs/plate_hole_validation/` | Stress-concentration check of the plate model against theory |
| `outputs/figures/` | Figures written by the two figure scripts |

## What is deliberately not in this repository

Available on reasonable request:

- **The remaining trained checkpoints** of the earlier datasets, about 3.4 GB,
  including those behind the v1, v3 and alternate-partition ensembling results,
  and the checkpoints of the partition and comparison runs on v4f, whose
  per-seed logs are included.
- **Raw CalculiX output** (`.frd`, `.inp`, `.msh`, `.12d`), which the scripts
  write to temporary directories and reduce to the grid solutions and contact
  metrics shipped here.

## Reproducibility notes

- **Leakage-safe split.** Complete trajectories, never single cases, go to
  train, validation or test, and the partition is fixed before training. The
  friction variants of v4f keep the trajectory of their original case. The
  target scale is fitted on the training partition only. See
  `src/gearstress/splits.py`.
- **No input needs a solve.** The contact point follows from the involute
  geometry and the contact position.

## License

- **Code** (`src/`, `scripts/`, `tests/`, `configs/`): MIT, see [LICENSE](LICENSE).
- **Datasets, result logs and trained checkpoints** (`data/`, `outputs/`):
  CC BY 4.0, see [DATA_LICENSE](DATA_LICENSE).

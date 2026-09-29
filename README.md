# Contact-Aware Gear

Code, datasets, trained models and result logs for a neural-operator surrogate
of the in-plane stress field in a spur-gear tooth under quasi-static frictional
contact: a ten-member Fourier Neural Operator ensemble trained on nonlinear
contact solutions from CalculiX and evaluated on held-out loading trajectories.

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

All accuracy figures are on the 27 test cases of three held-out trajectories.
The committed outputs are under `outputs/surrogate/`.

| Result | Value | Produced by |
|---|---|---|
| Field accuracy, ten-member ensemble trained on v4f (432 cases) | relative L2 = 0.0140, R² = 0.9999 | `analyze_gear_mesh_surrogate.py` |
| Same, single network (mean over ten seeds) | relative L2 = 0.0239 | same |
| Same, ensemble trained on v4 only (144 cases) | relative L2 = 0.0534 | same |
| Labels against a reference with every length scale refined 1.5-fold | relative L2 = 0.0900 | same |
| Surrogate against that refined reference | relative L2 = 0.0894 | same |
| Peak root tensile stress (max. principal), surrogate against labels | 1.0% mean absolute error | same |
| Friction response, mu = 0.04 to 0.12 | FE field changes 9.7%, surrogate 9.4%; sign of the root-stress change correct in 27 of 27 cases | same |
| Contact: FE peak pressure against Hertz line contact, all 144 cases | 0.98 to 1.09 | same |
| Contact: resultant tangential force / (mu x normal force), sliding cases | 0.96 to 1.00 | same |
| Contact penalty 50E against 200E per mm | load within 1.3%, root stress within 1.2%, field within 0.030 | `check_gear_mesh_penalty.py` |
| CalculiX solve, median of nine run one at a time | 27.1 s per case | `time_gear_mesh_solves.py` |
| Ten-member ensemble, one case on CPU | 76 ms, about 360-fold faster | `benchmark_surrogate_inference.py` |

Read the table with its limits:

- **The labels are the accuracy limit.** Refining the finite-element model
  changes the labels by 0.090 in relative L2, about six times the surrogate's
  error against them; 69% of that difference lies within 1 mm of the contact
  point and nearly all of the rest next to the sharp root corner.
- **The contact is not represented on the prediction grid.** The Hertz width
  spans about one pixel of the 64 x 64 grid and three flank segments of the
  finite-element polygon. The surrogate predicts tooth-body stress, not contact
  pressure or subsurface stress.
- **Model simplifications.** Sharp root corner (no trochoidal fillet) on a
  clamped rim, a constant Coulomb coefficient without a lubricant film, and
  sliding represented by one imposed slip. Absolute root stresses are model
  values, not design stresses.
- **One tooth geometry and three test trajectories.** Whole trajectories are
  the unit of the split, so the 27 test cases are not independent.
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
uv run pytest            # 47 passed, 1 skipped
```

The skipped test reads raw CalculiX output that is not published here; it
skips cleanly without it.

## Repository layout

| Path | Contents |
|---|---|
| `src/gearstress/` | The package: finite-element case building and CalculiX I/O (`fem.py`, including the corrected `GearMeshCase`; `plate_hole_fem.py`), models (`models.py`), physics losses (`losses.py`), the alternative architectural mechanisms (`inventions.py`), active-learning acquisition (`active_learning.py`), leakage-safe splitting (`splits.py`), and an analytic debug proxy (`proxy.py`). |
| `scripts/` | All 61 dataset-generation, training, study, analysis and figure entry points. Each module docstring states what it does. |
| `tests/` | Unit and shape tests for the pipeline and the active-learning code. |
| `configs/` | Experiment settings. |
| `data/processed/` | Model-ready HDF5 datasets. |
| `outputs/` | Result logs, study summaries, solver outputs and the shipped checkpoints, laid out as the scripts write them. |

## Datasets (`data/processed/`)

| File | Contents |
|---|---|
| `gear_mesh_quasistatic_v4.h5` | Corrected model: 144 cases, 16 trajectories x 9 contact positions in the single-tooth contact zone. |
| `gear_mesh_quasistatic_v4f.h5` | v4 plus every case re-solved at mu = 0.04 and 0.12 (432 cases, same partition). The surrogate is trained on it. |
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
| `outputs/smoke/gear_mesh_quasistatic_v4/`, `..._v4f/` | The 20 gear-mesh networks (checkpoints and per-seed logs) |
| `outputs/surrogate/contact_load_path*`, `refined_flank_test/`, `hertz_contact_check.json` | Diagnosis of the earlier model's load path |
| `outputs/surrogate/comparison_statistics.json` | Supplementary statistics of the method comparisons |
| `outputs/smoke/gear_pair_transient_quasistatic_v1/`, `_v2/`, `_v3/` | Multi-seed runs on the earlier datasets, including the ten FNO variants, Physics-FNO, U-Net and DeepONet, the split-seed check and the training-size curve |
| `outputs/active_learning/` | Active-learning arms with per-checkpoint, per-seed logs |
| `outputs/smoke/al_misranking_comparison/` | Active-learning power analysis |
| `outputs/smoke/gear_pair_geometry_v1/` | Geometry variation |
| `outputs/gear_pair_penalty_convergence/`, `gear_pair_mesh_convergence/` | Penalty and mesh convergence of the earlier model |
| `outputs/plate_hole_results/` | Second-domain pilot: plate with an elliptical hole |
| `outputs/figures/` | Figures written by the two figure scripts |

## What is deliberately not in this repository

Available on reasonable request:

- **The remaining trained checkpoints** of the earlier datasets, about 3.4 GB,
  including those behind the v1, v3 and alternate-partition ensembling results.
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

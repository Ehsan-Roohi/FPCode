# Cylinder Fokker--Planck and neural ESML snapshots

This directory preserves the cylinder-flow solver snapshots identified in the
Unity project while auditing the JCP manuscript cases. The source directory is:

```text
/project/pi_roohie_umass_edu/fokkerplanckDeeponet/
```

## Contents

- `../../FP_PINN/legacy_source/147CylFP.py`: the canonical 5,000-step paper
  configuration.  It runs the exact cubic-FP and the 16-to-9 GPU-native DNN
  sequentially and writes both field files and the surface-coefficient
  comparison.  The accidental Stage12A insertion in the Unity top-level copy
  has been removed; the original 16-feature inference path is restored.

- `exact_fp/147CylFP_ENTROPY_60371609.py`: syntax-valid pre-Stage12A snapshot
  associated with Unity job `60371609` (3,000 steps, 500 warm-up steps,
  600,000 initial particles).
- `exact_fp/147CylFP_ENTROPY_60371611.py`: syntax-valid pre-Stage12A snapshot
  associated with Unity job `60371611` (20,000 steps, 5,000 warm-up steps,
  1,200,000 initial particles).
- `neural_esml/147CylFP_ESML_ESML_[A-E]_60465433.py`: the complete five-member,
  syntax-valid ESML neural-closure parameter sweep associated with Unity job
  `60465433`.
- `final_exact_replica/stage16B_164_patched_rank000.py`: the exact-FP source
  archived by successful replica job `60686987`, together with its patch
  report, run metadata, and text summary.
- `SOURCE_MANIFEST.csv`: original Unity file sizes, SHA-256 hashes, roles, and
  source paths. The hashes deliberately identify the archived inputs before
  the documented mesh-plot correction below.
- `mesh_visualization.py`: shared, CPU-side plotting utility for the adaptive
  half-annulus mesh.
- `quick_mesh_check.py`: CPU-only geometry smoke test; it does not require
  CUDA, CuPy, the neural weights, or a full flow simulation.

The top-level Unity copies were later modified by the Stage12A augmentation
patch and contained an indentation error. The files published in `exact_fp/`
and `neural_esml/` are the corresponding `.stage12A_bak` copies preserved on
Unity; all pass Python syntax parsing. Their original names are restored here
so they remain directly runnable. The manifest records this provenance.

The five A--E ESML scripts are retained as a sweep because they differ in
`ESML_SENSOR0`, `ESML_BETA_C`, `ESML_BETA_G`, `ESML_LAMBDA_C_MIN`, and
`ESML_LAMBDA_G_MIN`. The repository does not guess which two sweep members
were ultimately used as manuscript replicas; that selection should be made
from the archived run summaries/checkpoints. Keeping every member avoids
silently publishing the wrong neural configuration.

The ESML sweep and the Cartesian `164CylFPAdaptive.py` replica are later
validation/audit tracks.  They are not the source of the annular Physics/ML
contours used in the cylinder section of the paper.  Those contours match the
half-annulus quadtree geometry and paired Physics/ML exports of `147CylFP.py`.

Two source/manuscript discrepancies remain visible and are intentionally not
silently changed here: the runnable Unity source uses `U_INF = 2634.1` m/s and
`R_DOM = 0.65` m, while the manuscript states 2624 m/s and a radial extent of
`1.5D = 0.4572` m.  Reproducing the archived figures therefore means retaining
the source values; changing them defines a new verification case.

Large result directories, checkpoints, scheduler output, and particle fields
are intentionally excluded. These programs require the original
MPI/CUDA/CuPy/Numba HPC environment and `model_params.npz` for neural runs.

## Mesh-plot correction

The archived solver scripts originally selected plotted cells using both
`children[:, 0] == -1` and `rho > 0`.  In a particle method, a valid active
cell may temporarily have zero particles and therefore zero density.  Omitting
those cells made the empty region look like a distorted cylinder.  Conversely,
after coarsening, a flat scan of all allocated leaf nodes can include former
children that are no longer active.

The corrected plotter walks the current quadtree from its root, plots every
reachable leaf regardless of particle occupancy, maps angular cell edges as
arcs, and draws the cylinder wall explicitly at `R_CYL`.  This changes only
visualization; it does not change the Fokker--Planck or ESML evolution.

## Lightweight verification

```bash
python -m py_compile exact_fp/*.py neural_esml/*.py final_exact_replica/*.py
python quick_mesh_check.py
```

# Cylinder Fokker--Planck and neural ESML snapshots

This directory preserves the cylinder-flow solver snapshots identified in the
Unity project while auditing the JCP manuscript cases. The source directory is:

```text
/project/pi_roohie_umass_edu/fokkerplanckDeeponet/
```

## Contents

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
- `SOURCE_MANIFEST.csv`: file sizes, SHA-256 hashes, roles, and Unity provenance.

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

Large result directories, checkpoints, scheduler output, and particle fields
are intentionally excluded. These programs require the original
MPI/CUDA/CuPy/Numba HPC environment and `model_params.npz` for neural runs.

## Lightweight verification

```bash
python -m py_compile exact_fp/*.py neural_esml/*.py final_exact_replica/*.py
```


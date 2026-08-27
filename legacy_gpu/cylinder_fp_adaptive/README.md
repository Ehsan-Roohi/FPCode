# Adaptive cubic-FP cylinder solver (v163)

This directory preserves a research snapshot of an MPI/CUDA particle solver
for rarefied argon flow over a cylinder. It uses an adaptive cubic
Fokker–Planck closure, per-cell moment reconstruction, corrected local noise,
and half-domain symmetry.

## Scope and status

- Source: `cylinder_fp_adaptive_v163.py`
- Historical console label: `v172: Local Noise Reduction + Unclamped Beta`
- Geometry: 0.3048 m cylinder in a two-dimensional half-domain
- Default grid: 1500 x 1000 cells
- Default initial particle count: 100,000,000
- Default run length: 100,000 steps
- Outputs: `surface_data.dat`, `field_data.dat`,
  `Surface_Properties.jpg`, and `Adaptation_Map.jpg`

The filename and embedded console label differ in the archived source. Both are
kept visible to avoid inventing a new version history. This is a production-size
research snapshot, not a lightweight example or a validated release benchmark.
No generated field files, binaries, or scheduler logs are included.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- Numba with a working CUDA installation
- mpi4py and an MPI runtime
- one or more CUDA-capable GPUs with enough memory for the selected grid and
  particle capacity

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Run

Review the constants near the top of the source before running. The archived
defaults are intentionally large and are unlikely to fit a workstation.

```bash
mpiexec -n 4 python cylinder_fp_adaptive_v163.py
```

Each MPI rank selects a CUDA device using `rank % num_devices`. Run from a
dedicated output directory because the script writes fixed output filenames.

## Lightweight verification

Syntax can be checked without importing CUDA dependencies:

```bash
python -m py_compile cylinder_fp_adaptive_v163.py
```

Numerical validation requires the original accelerator environment and is not
claimed by this archive.

## Provenance and licensing

This file was the only code artifact unique to the local `Fokker Planck Code`
tree when compared by SHA-256 with the duplicate `Mahdavi` tree. The legacy
Fortran collection was not copied here because its source headers identify
multiple collaborators and do not provide a distribution license.

Ehsan Roohi has confirmed ownership of this Python snapshot and authorized its
release under the MIT License. See `LICENSE` in this directory. The license is
scoped to this cylinder-solver package and does not relicense other historical
material elsewhere in the repository.

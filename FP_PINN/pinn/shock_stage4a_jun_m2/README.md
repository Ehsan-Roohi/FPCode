# Stage 4A — Jun–Zhang Mach-2 normal shock

This package reproduces the one-dimensional planar Mach-2 shock geometry and
Rankine–Hugoniot states from:

> Fei Fei, Haihong Liu, Zhaohui Liu, and Jun Zhang, “A Benchmark Study of
> Kinetic Models for Shock Waves,” *AIAA Journal* 58(6), 2596–2608 (2020),
> DOI: 10.2514/1.J059029.

The validated reference and residual in this stage use the Maxwell-molecule
BGK operator. They are not mislabeled as Cubic-FP: the AIAA paper does not
provide the complete numerical Cubic-FP coefficient solver and stabilization
parameter needed for an exact independent reproduction.

## What is new

- positive monotone neural density and temperature fields;
- exact Rankine–Hugoniot plateaus;
- hard conservation of steady mass, momentum, and energy flux;
- positive distribution correction `f = M exp(psi)`;
- steady BGK residual at fresh velocity-space points;
- 17 sparse macro locations plus three scalar diagnostic locks (phase,
  thickness slope, and asymmetry), with a complete 1600-cell validation grid;
- five velocity slices withheld from microscopic training;
- line plots only (no bar charts), with one shared legend outside the data
  region and both 300-dpi PNG and vector-PDF outputs;
- CSV tables, checkpoint, raw predictions, checksums, and an automatic
  pass/fail marker.

## One-line local run

From this directory, with the two reference NPZ files under `reference/`:

```bash
bash run_one_line.sh
```

All products are written to `outputs/stage4a_jun_m2/`. The packaging command
places a single ZIP in the repository root.

On Unity/Slurm, submit `run_stage4a.sbatch`; it creates a reusable project
virtual environment if needed and still places the final ZIP in repository root.

## Scope of the result

Stage 4A is a verification baseline and fixes the previous heat-flux mismatch
by enforcing the three physical flux invariants rather than learning heat flux
as a free output. A publication-strength Fokker–Planck claim still requires a
fully specified ES-FP/Cubic-FP implementation and at least the Mach-10 case.

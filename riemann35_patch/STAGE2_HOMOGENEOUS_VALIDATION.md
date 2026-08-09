# Riemann35 cubic-FP Stage 2: homogeneous validation

Stage 1 established that the provisional Julia source compiles, preserves the
five collision invariants, keeps a Maxwellian stationary in the OU limit, and
accepts a small realizable source step. Stage 2 is the first direct comparison
between:

- the deterministic FPCode particle update, including its peculiar-speed
  limiter and finite-step alpha correction; and
- Riemann35's continuous 35-moment source with analytical `Moments5_3D` and
  algebraic CHyQMOM quadrature for M6.

Both paths begin from the measured 35 moments of the same seeded 100,000-particle
Gaussian mixture. They run 200 steps with `dt=2.5e-4`, `tau=1`, `Pr=2/3`, and
`gamma_scale=0.05`, recording 21 aligned samples through final time `0.05`.

The job writes the following persistent files under
`results/riemann35_stage2_JOBID/`:

- `initial_moments.csv`: the common full 35-moment initial condition;
- `particle_history.csv`: FPCode particle diagnostics;
- `julia_chyqmom_m6_history.csv`: Julia closure diagnostics;
- `julia_closure_metrics.csv`: raw failure probes and projection metrics; and
- `summary.json`: scientific status, conservation errors, history-relative L2
  errors, and final M400 difference.

The first Unity attempt (`62745344`) found that the raw CHyQMOM-M6 source leaves
the M4 realizability cone even when the step is divided into 256 substeps. The
updated driver separates two questions rather than hiding that result:

1. A raw-closure probe records the first failed step and trial margins down to
   `dt/65536`.
2. A clearly labelled diagnostic trajectory applies Riemann35's shipped
   Appendix-B projection, followed by the smallest tested convex blend with a
   Gaussian having the same M0--M2 moments needed to recover a strict interior
   margin. Projection frequency and correction magnitude are recorded.

The operational job fails on non-finite data, time misalignment, failure to
recover an interior state, mass drift above `1e-12`, or energy drift above
`1e-10`. Closure accuracy remains diagnostic: the summary reports whether the
projected trajectory's final M400 difference improves over the earlier
Gaussian-tail value of 9.702%, but it also marks the raw closure as scientifically
failed whenever projection was required. A projected result is not represented
as validation of the unmodified closure.

Submit from the FPCode branch root on Unity:

```bash
sbatch --export=ALL,JULIA_MODULE=julia/1.10.5 \
  riemann35_patch/run_unity_stage2.sbatch
```

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
- `julia_closure_metrics.csv`: legacy-cap probes and adaptive-step metrics; and
- `summary.json`: scientific status, conservation errors, history-relative L2
  errors, and final M400 difference.

The first Unity attempt (`62745344`) stopped when the original validation
integrator exhausted its hard cap of 256 equal substeps. The margin probe in the
second attempt (`62745699`) showed that, at macro step 100 (`t=0.025`), trial
steps through `dt/256` produced a nonpositive directional variance
(`margin=-Inf`), while `dt/512` and smaller remained realizable. That result
ruled out a vector field pointing outside the cone for every positive step, but
did not by itself distinguish a finite cap from a runaway closure.

The persistent-adaptive third attempt (`62745949`) made that distinction. It
accepted 106 and rejected 25 microsteps, reached a minimum accepted
`h/dt=2^-23`, and then required a step below `2^-24`. At that point the source
norm was `5.18e10`, even though the minimum accepted realizability margin was
positive (`0.210`) and mass, momentum, and energy remained at roundoff. The raw
unclipped continuous source therefore has a high-moment runaway near `t=0.025`;
increasing the cap is not a scientifically adequate fix.

The current driver therefore uses no projection. It advances the raw analytical
M5 + CHyQMOM-M6 source with a persistent adaptive microstep:

1. reject and halve any trial with negative/non-finite realizability margin;
2. accept realizable trials and retain the stable microstep across macro steps;
3. attempt to double the microstep only after 16 consecutive acceptances; and
4. record accepted/rejected counts, minimum `h/dt`, source norm, and margins.

The operational job fails on non-finite data, failure to reach final time, time
misalignment, mass drift above `1e-12`, or energy drift above `1e-10`. Closure
accuracy remains diagnostic: the summary reports whether the adaptive raw
trajectory's final M400 difference improves over the earlier Gaussian-tail
value of 9.702%. No Appendix-B projection or Gaussian interiorization is used
in the reported closure trajectory. Stage 3 introduces a separate bounded
quadrature source matching FPCode's speed limiter and continuous alpha
correction; it does not relabel the failed raw trajectory as successful.

Submit from the FPCode branch root on Unity:

```bash
sbatch --export=ALL,JULIA_MODULE=julia/1.10.5 \
  riemann35_patch/run_unity_stage2.sbatch
```

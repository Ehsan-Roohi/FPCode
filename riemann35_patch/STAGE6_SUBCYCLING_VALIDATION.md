# Riemann35 cubic-FP Stage 6: realizability-aware finite-map subcycling

Stage 5 repaired the order-one CHyQMOM reconstruction jump and passed all 52
focused tests, but its full `dt=2.5e-4` candidate left the realizability cone
at macro step 1. Stage 6 tests the narrow next hypothesis: the corrected
finite map is scientifically usable, but requires smaller finite map steps.

For every macro interval Stage 6 starts from the same committed state and
tries `nsub = 1, 2, 4, ...`. A failed trial is discarded in full. The first
trial whose every microstep remains realizable is committed. This advances the
whole physical interval and is not equivalent to scaling or clipping one
collision increment.

The default ceiling is 256 substeps per macro interval. It can be changed at
submission time with `FINITE_MAX_SUBSTEPS`, but exceeding this ceiling is a
scientific failure rather than an invitation to silently run indefinitely.
The output records:

- the complete first-step powers-of-two ladder;
- accepted and discarded finite-map evaluations;
- maximum substeps per macro interval, retried intervals, and restarts;
- minimum realizability margin and microstep size;
- finite-map alpha, node-speed, quadrature-residual, and collision-increment
  diagnostics;
- mass, momentum, and energy drift.

The comparison passes only if the trajectory reaches `t=0.05`, preserves the
collision invariants, beats the 9.702% Gaussian-tail final-`M400` baseline,
keeps the `M200` history error below 2%, and keeps the stress-history error
below 5%. `M300` and heat-flux histories remain reported but are not hard gates
because their particle reference can pass near zero.

Submit from a checkout containing this stage:

```bash
sbatch --export=ALL,JULIA_MODULE=julia/1.10.5,FINITE_MAX_SUBSTEPS=256 \
  riemann35_patch/run_unity_stage6.sbatch
```

If no power-of-two trial through 256 is realizable, or if the run reaches final
time but misses the accuracy gates, the current 35-moment CHyQMOM increment is
not adequate for this case. The next stage should change the closure (for
example, a positive velocity-grid or maximum-entropy reconstruction), rather
than adding another ad hoc limiter.

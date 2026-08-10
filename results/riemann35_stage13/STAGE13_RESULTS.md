# Stage 13: finite Hermite truncation audit

A zero-tail Hermite truncation was tested as a deterministic alternative to the particle reference. It is exact for a Maxwellian, but positivity is not guaranteed away from equilibrium. Each trajectory was stopped at the first negative full 35-moment realizability margin.

| K | dt/tau | completed t/tau | first negative step | minimum margin | decision |
|---:|---:|---:|---:|---:|---|
| 6 | 0.0025 | 0.0375 | 15 | -3.778e-04 | REJECT |
| 8 | 0.0025 | 0.0375 | 15 | -1.144e-01 | REJECT |
| 8 | 0.00125 | 0.03375 | 27 | -7.667e-04 | REJECT |

The violation also occurs for K=8 after halving the time step. It is therefore not acceptable to quote this signed Hermite trajectory as a physical reference. The next deterministic reference must evolve a nonnegative velocity-space density or enforce a positivity projection with a documented convergence study.

# Stage 15: particle-count refinement

The Stage-11 rare-beam reference averages sixteen independent 100k-particle paths. Because the cubic-FP coefficients are nonlinear functions of empirical moments, averaging many small ensembles need not equal one continuum solution. Eight independent 200k-particle paths test this finite-ensemble bias at fixed total work order.

| Ensemble | M400 history vs QMC | final particle mean | final particle SEM | final QMC mean | combined z |
|---|---:|---:|---:|---:|---:|
| 16x100k | 1.01% | 23.6103 | 1.142e-01 | 24.063 | -3.92 |
| 8x200k | 1.02% | 23.3466 | 1.108e-01 | 24.063 | -6.38 |

Doubling particles per seed did not move the particle mean toward QMC: the M400 history difference remains about 1%, while the two particle ensembles are statistically compatible with one another. We therefore retain particle and QMC trajectories as a conservative reference envelope. This envelope is still much narrower than the 15--22% errors of the rejected algebraic closures.

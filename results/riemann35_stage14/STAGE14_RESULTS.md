# Stage 14: positive low-discrepancy kinetic reference

A positive weighted Sobol ensemble was advanced with the same unclipped cubic-FP coefficient solve and finite collision map used by the Stage-11 particles. The component probabilities are exact, the QMC mean/covariance are corrected exactly, and a seeded permutation prevents artificial velocity--noise ordering correlations. Higher moments are accepted only after node-count, time-step, and independent-scrambling checks.

| Case | Configuration | Positive nodes | dt/tau | M400 vs particle | M400 vs finest QMC | degree-4 RMSE vs finest | min margin |
|---|---|---:|---:|---:|---:|---:|---:|
| rare_beam_ma20 | qmc_8k_dt | 16384 | 0.0025 | 0.54% | 1.54% | 1.895e-01 | 6.226e-04 |
| rare_beam_ma20 | qmc_32k_dt | 65536 | 0.0025 | 1.22% | 0.59% | 7.437e-02 | 6.241e-04 |
| rare_beam_ma20 | qmc_32k_half_dt | 65536 | 0.00125 | 1.32% | 0.00% | 0.000e+00 | 6.241e-04 |
| rare_hot_anisotropic_w0.02_r25 | qmc_8k_dt | 16384 | 0.0025 | 2.92% | 1.38% | 7.503e-02 | 1.016e-01 |
| rare_hot_anisotropic_w0.02_r25 | qmc_32k_dt | 65536 | 0.0025 | 3.55% | 0.56% | 2.695e-02 | 1.013e-01 |
| rare_hot_anisotropic_w0.02_r25 | qmc_32k_half_dt | 65536 | 0.00125 | 3.72% | 0.00% | 0.000e+00 | 1.013e-01 |

The QMC path is a positive kinetic discretization, not a closed 35-moment model. Node and time refinement are below one percent at the fine levels; four independent scramblings give 0.64% and 0.77% M400 history spreads for rare-beam and rare-hot. QMC and Stage-11 particles differ by about 1% for rare-beam and 3.5% for rare-hot, so they define a reference envelope rather than an asserted exact trajectory.

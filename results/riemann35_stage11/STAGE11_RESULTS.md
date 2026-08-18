# Stage 11 particle validation through one collision time

## Scope and statistical control

Six homogeneous trajectories were advanced to t/tau=1 with 16 independent particle seeds and 100,000 particles per seed. The reference therefore contains 1,600,000 particles per case. Uncertainty is computed across independent seeds, not across correlated output times. The primary model uses the continuous cubic-FP drift without speed clipping.

Each particle trajectory was differenced from its own random t=0 sample and shifted to the exact analytic initial moment vector. This paired-change estimator removes persistent initialization noise while retaining between-seed uncertainty in the relaxation history. Its confidence interval therefore collapses at t=0 by construction and makes whole-history errors slightly optimistic near the initial point. With 1.6 million particles per case, the remaining nonlinear control-variate correction is second order and does not affect the method ranking.

## Main results

No closure dominates in accuracy. Stage 9 remains strongest on the separable counter-stream/crossing families, while guarded Grad/GQMOM is slightly better on the correlated case and clearly better, though still insufficient, on the rare beam.

The primary closure metric below is the componentwise RMS error of the nondimensional degree-four block. The aggregate 35-vector norm is retained in the machine-readable files only as a secondary diagnostic because its raw L2 weighting is dominated by the largest fourth moment in the rare-beam case.

| Case | Stage-9 degree-4 RMSE | Grad degree-4 RMSE | Stage-9 M400 error | Grad M400 error | Grad limiter |
|---|---:|---:|---:|---:|---:|
| stage9_correlated | 6.5432e-03 | 6.1269e-03 | 0.43% | 0.41% | 0/400, lambda_min=1.000 |
| rare_hot_anisotropic_w0.02_r25 | 2.1996e-01 | 2.1735e-01 | 4.45% | 4.21% | 0/400, lambda_min=1.000 |
| counterstream_ma20 | 4.4944e-03 | 1.7862e-02 | 0.20% | 1.02% | 0/400, lambda_min=1.000 |
| crossing_ma20 | 4.0000e-03 | 9.5916e-03 | 0.32% | 0.42% | 0/400, lambda_min=1.000 |
| rare_beam_ma20 | 2.0067e+00 | 1.4667e+00 | 16.59% | 12.67% | 14/400, lambda_min=0.675 |
| counterstream_ma100 | 4.5415e-03 | 1.7754e-02 | 0.20% | 1.01% | 0/400, lambda_min=1.000 |

## Physical interpretation

For the rare beam, Stage 9 and guarded Grad/GQMOM have M400 history errors of 16.59% and 12.67%, respectively. The Grad guard is active in 14 of 400 steps, but removes only 1.40% of the source-norm-weighted nonlinear contribution. Hence the large M400 bias is primarily a closure error, not suppression by the lambda guard.

Halving the closure time step changes the guarded-Grad rare-beam degree-four componentwise RMSE from 1.4667e+00 to 1.4937e+00. Persistence under refinement shows that the dominant discrepancy is not the time step.

The second-moment/stress histories and the contracted heat-flux relaxation are consistency checks rather than independent closure-accuracy tests: the physical 9-by-9 coefficient solve enforces their production rates by construction on both the moment and particle paths. Independent closure evidence comes primarily from the degree-four block and, secondarily, from the unconstrained part of the third-order block. Counter-stream and crossing retain approximately one-percent-or-better M400 histories. Odd moments that are zero by symmetry are excluded from physical percentage claims because their relative errors are noise-dominated.

The anisotropic rare-hot M400 comparison is statistically weak despite 1.6 million particles because the rare hot population gives a very large fourth-moment sampling variance. Its final discrepancy is within roughly one seed standard error; a deterministic Hermite reference is required before making an accuracy claim for this case.

## Decision

The Stage-9 call site explicitly sets speed_cap=Inf; therefore Stage 9, guarded Grad, and the particle reference all use the same unclipped cubic drift in this comparison.

Stage 11 validates stability and conservation and gives independent degree-four evidence for the correlated, counter-stream, and crossing relaxations. It also identifies a decisive accuracy gap for the high-skewness rare-beam M400 history. A spatial JCP demonstration should wait until that gap is reduced. Convex blending of the current Stage-9 and Grad tails cannot remove a bias shared by both.

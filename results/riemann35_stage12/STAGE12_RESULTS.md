# Stage 12 high-skew closure candidates

All candidates were tested on the Stage-11 rare-beam reference through one collision time. The two-population reconstruction is exact at t=0 for this generating mixture, but that exact initial fit does not guarantee an accurate history because cubic FP evolution immediately drives each population away from a Gaussian shape.

| Candidate | Degree-4 RMSE | M400 history error | M300 error | Limited steps | Minimum margin | Gate |
|---|---:|---:|---:|---:|---:|---:|
| algebraic_base | 2.3078e+00 | 18.99% | 0.47% | 0 | 7.125e-04 | FAIL |
| algebraic_residual | 2.6962e+00 | 21.58% | 3.78% | 87 | 9.994e-13 | FAIL |
| persistent_two_population | 2.2878e+00 | 18.87% | 8.40% | 0 | 7.126e-04 | FAIL |
| dynamic_m6 | 2.0464e+00 | 16.83% | 0.33% | 0 | 7.125e-04 | FAIL |
| dynamic_m8 | 1.8449e+00 | 15.19% | 0.30% | 0 | 7.125e-04 | FAIL |

The best Stage-12 candidate is `dynamic_m8` at 15.19%, still well above the 3% gate. No Stage-12 candidate is promoted to the production closure.

The source-level particle diagnostic shows that Stage 9 and the two-population model reproduce the exact initial M400 source, while all tested algebraic closures become overly dissipative after the initial transient. This rejects the hypothesis that an exact t=0 two-Gaussian fit alone resolves the rare-beam history.

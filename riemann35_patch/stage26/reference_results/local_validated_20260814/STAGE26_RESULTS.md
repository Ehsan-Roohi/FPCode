# Stage 26: regularized four-delta nonequilibrium audit

This homogeneous screening test follows Rodney Fox's proposed four-delta construction: unit mass, zero momentum, unit energy trace, and nonzero third-order moments. The four planar deltas are represented by common narrow 3-D Gaussians so the HyQMOM/GQMOM covariance remains positive definite.

The reference is a positive independently scrambled Full-FP QMC ensemble. It tests closure accuracy for the same cubic FP operator; it is not a substitute for later MD/DSMC validation of the collision model itself.

Overall gate: **PASS**

| Gate | Result |
|---|---:|
| all four methods completed | PASS |
| initial mass momentum energy constraints | PASS |
| initial third order moments nonzero | PASS |
| positive realizable histories | PASS |
| collision invariants | PASS |
| reference M400 scramble spread | PASS |
| reference M420 scramble spread | PASS |
| adaptive M400 history error | PASS |
| adaptive M420 history error | PASS |
| positive reference and adaptive microstate weights | PASS |
| no blocked causal activation | PASS |
| no donor persistence probe | PASS |

| Method | M400 error | M420 error | Third-norm error |
|---|---:|---:|---:|
| Stage-9 mixture | 10.53% | 18.78% | 16.41% |
| Grad/GQMOM | 4.16% | 78.13% | 1.23% |
| Causal memory | 0.48% | 1.74% | 0.75% |

| Algebraic method | Minimum weight | Max negative weights | Negative mass fraction | Minimum even tail |
|---|---:|---:|---:|---:|
| Stage-9 mixture | 5.66084e-06 | 0 | 0.000% | 0.103784 |
| Grad/GQMOM | -0.00533913 | 82 | 4.861% | -0.129465 |

The adaptive run enforces no-donor persistence. A separate safe-state probe forces a release request without an available causal donor and must retain the positive microstate.

Signed Grad/GQMOM source weights are reported as a comparison diagnostic. A negative even tail moment is physically inadmissible even when the retained 35-moment vector remains realizable.

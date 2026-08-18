# Stage 18: adaptive positive micro-solver sizing

Stage 17 proves that the missing tail is not identifiable from the instantaneous 35-moment state. This screen therefore retains a positive kinetic microstate only in troubled cells. It measures the smallest Sobol ensemble that meets the 3% rare-beam gate; it is not presented as a new algebraic closure.

| points/component | total nodes | scrambles | runtime sum | M400 vs particle | M400 vs QMC | scramble spread | min margin | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 512 | 1024 | 4 | 2.77s | 3.43% | 2.59% | 4.77% | 5.973e-04 | FAIL |
| 1024 | 2048 | 4 | 3.96s | 1.67% | 1.10% | 2.32% | 6.179e-04 | PASS |
| 2048 | 4096 | 4 | 6.21s | 1.10% | 0.99% | 2.83% | 6.208e-04 | PASS |
| 4096 | 8192 | 4 | 10.85s | 1.58% | 1.39% | 1.62% | 6.210e-04 | PASS |
| 8192 | 16384 | 4 | 20.05s | 0.45% | 1.20% | 1.40% | 6.226e-04 | PASS |

The smallest passing proof-of-concept uses 2048 positive nodes. The next engineering step is a troubled-cell activation/deactivation rule and conservative projection between this persistent microstate and HyQMOM-35. Spatial publication tests remain blocked until that coupling is implemented and benchmarked.

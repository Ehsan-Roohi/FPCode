# Stage 16: positive maximum-entropy closure

The candidate uses an adaptive two-population quadrature only as positive support, then solves the discrete entropy dual so that every retained moment through degree four is matched. M5/M6 are evaluated from the resulting positive weights. Promotion requires M400 error below 3% against both independent reference constructions.

| Candidate | status | support | M400 vs particle | M400 vs QMC | degree-4 RMSE vs QMC | limited | min margin | gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| maxent_n4 | REACHED_FINAL_TIME | 128 | 19.56% | 20.36% | 2.507e+00 | 0 | 7.125e-04 | FAIL |
| maxent_n6 | REACHED_FINAL_TIME | 432 | 19.99% | 20.78% | 2.557e+00 | 0 | 7.125e-04 | FAIL |
| maxent_n8 | REACHED_FINAL_TIME | 1024 | 21.42% | 22.20% | 2.708e+00 | 0 | 7.125e-04 | FAIL |

No candidate is promoted unless it passes the reference-envelope gate and remains positive, conservative, and realizable for the full collision time.

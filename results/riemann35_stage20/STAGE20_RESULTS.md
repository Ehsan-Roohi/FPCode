# Stage 20: causal positive tail memory with hysteresis

The Stage-19 disagreement sensor is now part of the solver rather than an offline script. A positive microstate may be born only from a known initial decomposition or a supplied causal donor. Entropy reweighting matches all 35 transported moments on that support; failure to match is explicit. Dropping an active microstate is the direct positive moment projection.

Sensor cadence is every 10 collision steps. The on/off source thresholds are 0.10124/0.05062; the tail thresholds are 0.41005/0.20503. A source alarm additionally requires standardized skewness above 1.0e-03; this excludes the symmetric counter-stream false alarm found in the lifecycle audit. Release requires 8 consecutive safe sensor evaluations and at least 20 active collision steps.

| Case | Sensor at t=0 | Micro steps | Stage-9 M400 error | Adaptive M400 error | QMC error | Scramble spread | Blocked | Chatter |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage9_correlated | off | 0.0% | 0.43% | 0.43% | -- | 0.00% | 0 | 0 |
| rare_hot_anisotropic_w0.02_r25 | off | 0.0% | 4.45% | 4.45% | 1.64% | 0.00% | 0 | 0 |
| counterstream_ma20 | off | 0.0% | 0.20% | 0.20% | -- | 0.00% | 0 | 0 |
| crossing_ma20 | off | 0.0% | 0.32% | 0.32% | -- | 0.00% | 0 | 0 |
| rare_beam_ma20 | ON | 100.0% | 16.59% | 1.66% | 1.12% | 2.33% | 0 | 0 |
| counterstream_ma100 | off | 0.0% | 0.20% | 0.20% | -- | 0.00% | 0 | 0 |

The rare-beam reference-envelope maximum is 2.33%; the 3% gate is PASS. The equal-case average micro active fraction is 16.7%.

A separate safe-causal-birth control exercised the off path and released micro to macro at step 80 (t/tau=0.200) with exactly one transition.

Retrospectively applying the skew gate to the 292 Stage-19 synthetic states changes recall from 95.1% to 94.7%, precision from 98.0% to 98.4%, false-positive rate from 17.2% to 13.8%, and active fraction from 87.3% to 86.6%. This is a disclosed target-family tradeoff, not a new universal calibration.

This is a homogeneous lifecycle gate. It validates causal birth, positive bidirectional projection, and the activation/release state machine. It does not yet validate spatial donor selection, kinetic transport across cell faces, or an independent DVM reference.

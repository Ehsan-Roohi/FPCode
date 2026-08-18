# Stage 14 independent-scrambling control

The finest positive QMC rule was repeated with 4 independent Sobol scramblings. Uncertainty below is computed across scrambling replicates, separately from the Stage-11 particle-seed uncertainty.

| Case | M400 history vs particle | QMC scramble spread | final QMC SEM | final particle SEM | combined z | min margin |
|---|---:|---:|---:|---:|---:|---:|
| rare_beam_ma20 | 1.02% | 0.64% | 1.798e-02 | 1.142e-01 | 3.92 | 6.241e-04 |
| rare_hot_anisotropic_w0.02_r25 | 3.45% | 0.77% | 1.471e-02 | 3.388e-01 | 1.00 | 1.013e-01 |

The positive QMC mean is accepted as the current kinetic reference only when the node/time refinement differences and the independent-scrambling spread are both small relative to the closure error being measured.

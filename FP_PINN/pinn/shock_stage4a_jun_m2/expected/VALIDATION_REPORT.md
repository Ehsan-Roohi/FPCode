# Stage 4A validation report

## Result

Profile gate: **PASS**.

This is a reproducible Mach-2 planar normal-shock result for the geometry and
plateau states used by Jun Zhang's AIAA benchmark.  The certified numerical
operator in this package is BGK.  It is **not** claimed to be the paper's
Cubic-FP solution because the paper does not publish the complete Cubic-FP
coefficient/regularization implementation.

## Main errors against the independent 1600-cell DVM-BGK reference

| quantity | relative L2 | active-support relative L2 |
|---|---:|---:|
| density | 0.2058% | — |
| velocity | 0.1449% | — |
| temperature | 0.1115% | — |
| heat flux | 2.2661% | 1.9058% |
| normal stress | 2.2233% | 2.1350% |

The three steady fluxes are imposed exactly; their maximum relative drifts are
1.151e-07,
1.039e-07, and
1.866e-07.

Shock-thickness error: 1.509%.
Shock-asymmetry error: 0.171%.
Held-out distribution-slice mean/max L2 error:
2.861% /
3.791%.
Fresh relative BGK residual RMS/p99:
5.971e-02 /
2.765e-01.

## Interpretation

The heat-flux improvement is physical, not a plotting adjustment: density and
temperature are learned while mass, momentum and energy flux conservation
algebraically determine velocity, stress and heat flux.  Seventeen macroscopic
locations, three explicitly reported scalar diagnostic locks (phase, maximum
density slope, and density asymmetry), and sparse microscopic anchors are used
for training; the remaining complete 1600-point profiles and five full velocity
slices are validation data.

Stage 4A is a verification baseline.  A publication-level FP claim still needs
the same run with a fully specified ES-FP or Cubic-FP operator and then the
discriminating Mach-10 case.

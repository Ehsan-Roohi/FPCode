# Stage 34 model comparison and decision

## Qualification result

The official `counterstream_ma20` state is an equal-weight, genuinely
bimodal two-Gaussian distribution (`a/sigma=20`). It is evolved
homogeneously to `t/tau_FP=1`; no spatial transport, DVM projection,
activation rule, or kinetic memory is present.

Stage-9 passes the declared retained-state qualification:

- aggregate M0--M4 history error: **0.573%**;
- worst active retained component: **M004, 0.911%**;
- minimum retained H2 margin: **4.248201e-5**;
- minimum necessary H4-PSD margin: **3.197392e-5**;
- limiter: **never used**;
- maximum invariant drift: **2.7e-15**.

The positive H4 result is a necessary condition, not proof of the full
multivariate truncated moment problem.

## Cubic-FP numerical comparison

All values below are history-relative L2 errors by total-degree block against
the time-refined positive QMC cubic-FP reference. Odd blocks are identically
zero by symmetry and are omitted.

| Path | Degree 2 | Degree 4 | Degree 6 | Degree 8 | Interpretation |
|---|---:|---:|---:|---:|---|
| Stage-9 positive finite mixture | 0.311% | 0.596% | 0.911% | 2.703% | qualified through retained M0--M4; M6 useful; M8 diagnostic |
| Grad--HyQMOM signed extension | 0.311% | 0.751% | 3.602% | 9.816% | acceptable retained state, weaker predictive tail |
| 8 x 100,000 particles | 0.276% | 0.401% | 0.769% | 2.126% | independent stochastic cross-check of the same cubic-FP law |
| QMC node change, N to 4N | 0.350% | 0.837% | 1.769% | 3.275% | velocity-node uncertainty |
| QMC time change, dt to dt/2 at 4N | 0.373% | 0.586% | 1.253% | 3.197% | time/reference uncertainty |

Thus degree 6 is converged below 3%. Degree 8 is close but remains a
diagnostic because the QMC node and time changes are slightly above 3%.

The separate `crossing_ma20` initialization exposes a decisive problem if
the signed Grad quadrature is interpreted as an M0--M8 distribution: 4.606%
of its quadrature mass is negative and its necessary H4-PSD margin is
`-8.131e-3`. The Stage-9 reconstruction is positive and restores the exact
initial M0--M8 sequence to roundoff.

## BGK and ES-BGK controls

The clocks are rate matched, not symbol matched:

`tau_BGK = tau_sigma,ES = tau_FP/2`.

At `Pr=2/3`, ES-BGK then matches the cubic-FP linear stress and heat-flux
relaxation rates. BGK and ES-BGK are alternative collision models, not
accuracy references for cubic FP. Their differences from the positive QMC
cubic-FP trajectory quantify different model physics:

| Selected moment | QMC final | Stage-9 final | BGK final | ES-BGK final | BGK history difference vs QMC | ES-BGK history difference vs QMC |
|---|---:|---:|---:|---:|---:|---:|
| M200 | 1.27206 | 1.26866 | 1.26866 | 1.26866 | 0.236% | 0.236% |
| M400 | 4.46259 | 4.42341 | 3.81197 | 3.41628 | 14.384% | 18.230% |
| M600 | 24.36340 | 24.37844 | 16.67802 | 11.15340 | 36.012% | 49.073% |
| M800 | 175.36647 | 181.47951 | 102.18795 | 43.42866 | 51.568% | 74.998% |

This symmetric case has zero heat flux, so it does not test the Prandtl
mechanism. It shows that the models predict materially different high-order
relaxation, but it cannot establish which collision law is physically more
accurate. That requires an asymmetric two-peak test and independent
Boltzmann/DSMC evidence.

## Decision

1. Use the **positive Stage-9 finite-mixture closure**, not the signed Grad
   M0--M8 extension, for continued cubic-FP research.
2. Keep **ES-BGK as the production baseline** until cubic FP demonstrates a
   material advantage against independent physical data, not merely against
   a kinetic discretization of itself.
3. Keep the causal/DVM microstate only as a diagnostic or adaptive fallback;
   it does not replace the hyperbolic HyQMOM transport solver.
4. Before another shock claim, freeze the Knudsen-number convention and run
   the same Mach-2 spatial problem with ES-BGK. The existing prototype uses
   `tau_FP=1`; with `omega=0.5`, the code-local mapping gives `Kn=1` upstream
   and `Kn=3.295` downstream, but the physical reference length was not fixed.
5. Add a predeclared asymmetric two-peak control to exercise heat flux and
   the Prandtl response, followed by Boltzmann/DSMC comparison.

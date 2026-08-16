# Stage 34: Rodney two-peak homogeneous audit

**Stage-9 qualification outcome: `HOLD`.** Stage-9 completion, retained H2 and necessary H4 PSD conditions, the declared 3% gate for every retained degree block and active component, and separate 3% QMC node/time convergence gates determine this label. A positive H4 margin is necessary but is not proof of full realizability; a negative margin is decisive. Grad–HyQMOM is a disclosed comparator; predictive M5–M8 remain diagnostics unless separately converged.

The qualification state is the official Stage-10 `counterstream_ma20` equal-weight, genuinely bimodal mixture. The former `±0.55 e_x`, `0.45 I` state is only a two-component unimodal control and is not qualification evidence. There is no transport, projection, activation, or kinetic memory. The Stage-9 finite Gaussian-mixture map and Grad–HyQMOM independently advance the retained 35 moments; QMC and particles advance positive velocity measures under the same implemented cubic-FP operator.

| Diagnostic | Value |
|---|---:|
| Stage-9 minimum retained H2 margin | 4.248201e-05 |
| Stage-9 minimum necessary H4 PSD margin | 3.197392e-05 |
| Grad minimum retained H2 margin | 4.248201e-05 |
| Grad minimum necessary H4 PSD margin | 1.418064e-06 |
| maximum signed negative mass fraction | 5.264373e-07 |
| minimum limiter fraction at samples | 1.000000e+00 |
| Stage-9 aggregate retained error (diagnostic) | 0.819% |
| Stage-9 worst active retained component | M004: 1.757% |
| Stage-9 per-degree/per-active-component gate | True |
| QMC node-refinement retained gate | False |
| QMC time-refinement retained gate | True |
| wall runtime | 62.05 s |

| Degree | Stage-9 relative L2 | Grad relative L2 | Particle relative L2 | QMC node refinement | QMC time refinement |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 2 | 0.299% | 0.299% | 0.329% | 1.183% | 0.350% |
| 3 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 4 | 0.858% | 0.909% | 0.674% | 2.331% | 0.853% |
| 5 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 6 | 1.613% | 3.813% | 1.341% | 4.688% | 1.813% |
| 7 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| 8 | 3.305% | 9.800% | 2.715% | 8.556% | 3.408% |

| Moment | Stage-9 final | Grad final | fine QMC final | particle final | Stage-9 minus QMC | Grad minus QMC |
|---|---:|---:|---:|---:|---:|---:|
| M200 | 1.2686558 | 1.2686556 | 1.268926 | 1.2799344 | -2.702180e-04 | -2.703798e-04 |
| M400 | 4.423407 | 4.4308296 | 4.4458443 | 4.4917624 | -2.243724e-02 | -1.501468e-02 |
| M600 | 24.378444 | 24.886352 | 24.546609 | 24.648946 | -1.681657e-01 | 3.397424e-01 |
| M800 | 181.47951 | 193.04194 | 181.86669 | 180.13622 | -3.871776e-01 | 1.117525e+01 |

The largest final Stage-9 scaled discrepancy among predictive M5–M8 is `M080`: 2.230402e+00. The corresponding Grad worst moment is `M800`: 1.117525e+01.

BGK and semi-analytic ES-BGK are exported only as alternative collision-model trajectories. The fair clock match is tau_sigma=tau_FP/2: tau_BGK=tau_sigma, while ES-BGK uses df/dt=(Pr/tau_sigma)(G_ES-f), nu=1-1/Pr, and requires Pr>=2/3. Disagreement with cubic FP is not labeled numerical error. This symmetric zero-heat-flux state does not test heat-flux relaxation or identify Prandtl response.

The QMC/particle references validate only the numerical closure for the implemented cubic FP collision operator. They are not hard-sphere, DSMC, or molecular-dynamics validation.
The archived crossing-Ma20 control is a caveat for the Grad tail: its necessary H4 PSD margin is negative at t=0. This does not change the Stage-9 outcome; the positive Stage-9 reconstruction satisfies that necessary condition in the control, without claiming that positivity alone proves the full multidimensional moment problem.

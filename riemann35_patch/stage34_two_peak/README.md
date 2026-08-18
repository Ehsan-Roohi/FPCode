# Stage 34: homogeneous equal-weight two-peak audit

This stage implements the collision-only test requested after the Stage-32
shock discussion. It deliberately removes transport and uses the official
Stage-10 `counterstream_ma20` definition. Its separation ratio is
`a/sigma=20`, so it is genuinely bimodal. The earlier mixture with means
`±0.55 e_x` and covariance `0.45 I` has `a/sigma=0.82`; it is preserved only
as `two_component_unimodal_control` and is not qualification evidence.

The retained 35 moments are advanced independently by the Stage-9 positive
finite-Gaussian-mixture cubic-FP map and the Appendix-C Grad--HyQMOM cubic-FP
map. Independent positive scrambled-Sobol QMC and random-particle ensembles
use the same cubic-FP coefficient solve and finite stochastic collision
update. No spatial transport, DVM projection, activation sensor, or kinetic
memory is used.

All raw moments through total degree eight (165 moments) are exported.  The
retained H2 matrix and normalized H4 matrix (requiring moments through degree
eight) are audited separately. H4 positivity is only a necessary PSD
condition, not proof of full multidimensional realizability; H4 negativity is
a decisive failure. Reference histories used for accuracy are paired-change
aligned to the analytic initial moments; realizability is always evaluated on
the unmodified positive raw reference measures.

A declared 3% history gate applies only to retained M0--M4 and must pass for
every degree block and every active/nonzero retained component. Reference
convergence is audited separately with a three-level QMC hierarchy: base
`N,dt`, node-refined `4N,dt`, and time-refined `4N,dt/2`. Both the node and
time comparisons must pass the same per-degree and per-active-component 3%
gate. M5--M8 are predictive diagnostics unless separately converged. Analytic
BGK and semi-analytic homogeneous ES-BGK trajectories are exported as
alternative collision models, not accuracy references. Their
clocks are matched by `tau_sigma=tau_FP/2`; ES-BGK uses `Pr=2/3` and
`nu=1-1/Pr=-0.5`. ES-BGK is restricted to `Pr>=2/3`. This symmetric state has
zero heat flux, so it does not test heat-flux relaxation or identify the
Prandtl response. The official Stage-10 `counterstream_ma20` and
`crossing_ma20` definitions are reused for initial-closure controls.

Quick run:

```bash
python riemann35_patch/stage34_two_peak/run_two_peak_audit.py \
  --smoke --output results/riemann35_stage34_two_peak_smoke
```

Default workstation run:

```bash
python riemann35_patch/stage34_two_peak/run_two_peak_audit.py \
  --output results/riemann35_stage34_two_peak
```

The default uses four QMC scrambles at each hierarchy level, four particle
seeds, `N=4096` and `4N=16384` points per mixture component, `dt/tau=0.005`
for the base/node levels, `dt/tau=0.0025` for the time-refined level and the
models/particles, and advances to `t/tau=1`. Use `--help` to reduce or refine
the independent reference hierarchies.

The default run is retained as a disclosed `HOLD`: its Stage-9 solution passes,
but the base-to-`4N` QMC comparison changes active component `M040` by 4.061%,
above the declared 3% reference gate. The accepted refined run is:

```bash
python riemann35_patch/stage34_two_peak/run_two_peak_audit.py \
  --qmc-base-points 8192 --qmc-refined-points 32768 \
  --qmc-scrambles 8 --particles 100000 --particle-seeds 8 --workers 4 \
  --output results/riemann35_stage34_two_peak_refined
```

The refined run passes all retained-state and reference-convergence gates.
Its Stage-9 history-relative errors by total degree are 0.311% (degree 2),
0.596% (degree 4), 0.911% (degree 6), and 2.703% (degree 8). Degree 8 remains
a diagnostic because its separate QMC node/time changes are 3.275% and
3.197%, respectively.

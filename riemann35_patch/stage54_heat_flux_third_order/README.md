# Stage 54: third-order moments in a heat-flux case

## Question

How accurately does the proposed 35-moment cubic-FP closure predict the
third-order moments when heat flux is nonzero?

This is the direct homogeneous test requested after the boundary-of-moment-
space audit.  It uses the already qualified positive Full-FP QMC update as an
independent kinetic reference for the same cubic FP operator.  It is not an
MD/DSMC validation of the physical collision model.

## Why the trace-free tensor is decisive

Let

`T_ijk = integral(c_i c_j c_k f dc)`

be the symmetric third-order central-moment tensor.  In three dimensions,

`q_i = (1/2) sum_j T_ijj`

and

`T_ijk = T^TF_ijk + (delta_ij t_k + delta_ik t_j + delta_jk t_i)/5`,

where `t_i=sum_j T_ijj=2q_i`.  The FP coefficient solve explicitly targets
the contracted heat-flux production.  Heat-flux agreement is therefore a
consistency check; the seven-dimensional trace-free part `T^TF` is the
independent third-order closure test.

Rodney's regularized four-population state is rotated out of the coordinate
plane by fixed 29-degree and 41-degree tilts.  Rotation leaves mass, momentum,
energy, and positivity unchanged, but makes all ten independent components of
`T` nonzero.  No symmetry-zero component is used in a percentage claim.

## Frozen comparison

Six jobs are run concurrently:

1. positive QMC at `8192` nodes/component and `dt/tau=0.0025`;
2. node-refined QMC at `32768` nodes/component and the same time step;
3. time-refined QMC at `32768` nodes/component and `dt/tau=0.00125`;
4. the proposed positive finite-mixture HyQMOM-35 closure;
5. the signed Grad/GQMOM comparator; and
6. the positive tail-memory extension.

Each QMC level and the positive tail-memory calculation use four independent
Sobol scramblings.  Histories extend to `t/tau=1` and are sampled every
`0.025 tau`.  The time-refined QMC mean is the accuracy reference.  Node,
time, and scramble convergence must each be below 3% before a closure claim is
accepted.  The QMC, proposed, and comparator paths all use the same unclipped
continuous cubic drift (`speed_cap=Inf`); no finite-speed limiter can improve
one side of the comparison selectively.

The selected positive extension must retain positive weights and H2
realizability, keep invariants below `2e-8`, and satisfy:

- full third-order tensor history error below 3%;
- trace-free tensor history error below 5%; and
- every component RMSE below 3% of the initial full-tensor norm.

The core proposed closure and the signed Grad/GQMOM result are both reported,
but neither is silently substituted for the selected positive extension.

## Outputs

The collector always writes:

- `stage54_heat_flux_summary.json`;
- `stage54_heat_flux_errors.csv`;
- `stage54_third_order_components.png` (all ten components);
- `stage54_heat_flux_summary.png` (contracted versus trace-free evidence);
- `STAGE54_RESULTS.md`; and
- `STAGE54_HEAT_FLUX_RESULTS_<timestamp>.zip`.

The component figure uses a single external legend and no annotations inside
the data panels, so labels cannot obscure curves.

## Local smoke test

```bash
python riemann35_patch/stage54_heat_flux_third_order/test_heat_flux_third_order.py
```

## Unity submission

Use the pinned 40-character commit printed for this stage:

```bash
FP_STAGE54_COMMIT=<commit> bash riemann35_patch/stage54_heat_flux_third_order/submit_unity_stage54.sh
```

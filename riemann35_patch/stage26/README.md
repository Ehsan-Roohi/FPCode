# Stage 26: regularized four-delta nonequilibrium audit

## Question

Can the cubic-FP closure handle the extreme homogeneous initial condition
suggested by Rodney Fox before another expensive spatial calculation is run?
The state has four planar velocity populations, unit mass, zero momentum, unit
energy trace, and nonzero third-order moments.  It exercises asymmetric
relaxation while remaining much cheaper than the Stage-25A spatial shock.

This stage answers a closure question for the existing cubic FP operator.  A
Full-FP QMC ensemble is the reference.  It is **not** MD/DSMC validation of the
physical collision model; that remains a separate external-validation task.

## Initial state

The fixed weights are `0.45, 0.25, 0.20, 0.10`.  Four distinct velocities are
centered, rotated by 17 degrees, and scaled so that

`(M200 + M020 + M002) / M000 = 1`.

Because the production code is three-dimensional and GQMOM requires a
positive-definite covariance, each mathematical delta is represented by the
same narrow isotropic Gaussian.  The frozen default assigns 3% of the energy
trace to this regularization and 97% to the four population centers.  The
reported initial constraint audit must show mass and energy errors and the
momentum norm below `1e-12`, with central third-order norm above `0.05`.

## Compared methods

1. Positive Full-FP QMC reference with four independent Sobol scramblings.
2. Stage-9 principal-axis Gaussian-mixture map.
3. Appendix-C Grad-HyQMOM/Gaussian-GQMOM map.
4. Causal positive tail memory using the unchanged Stage-19 sensor.

The adaptive calculation receives the four-population microstate only at the
known initial time.  The homogeneous problem has no inflow or active spatial
neighbor, so `causal_reactivation_available=False`: an active microstate is
not discarded after a temporarily safe sensor reading.  This is the explicit
no-donor persistence rule; blocked release events are recorded.

## Frozen screening controls and gates

- `dt/tau = 0.0025`, final time `t/tau = 1`;
- four QMC scramblings;
- 4096 Sobol points per component for Full FP;
- 1024 Sobol points per component for adaptive memory;
- sampling and sensor evaluation every ten steps;
- all four jobs run concurrently as a Slurm array.

The collector reports every metric, but the decision gates are:

- all methods complete;
- exact initial mass, momentum, energy, and nonzero-third-moment constraints;
- realizable histories and invariant errors below `2e-8`;
- Full-FP scramble spread below 3% for both `M400` and the untransported
  predictive observable `M420`;
- adaptive history errors below 3% for `M400` and `M420`; and
- no blocked causal activation.

Stage-9 and Grad/GQMOM errors are comparisons, not pass requirements.  If the
reference-spread gate alone fails, increase the reference point count before
changing any physical or closure setting.

## Outputs

Each array task writes a method JSON and compressed history archive.  The
after-any collector writes:

- `stage26_four_delta_summary.json`;
- `stage26_four_delta_errors.csv`;
- `stage26_four_delta_histories.png`;
- `STAGE26_RESULTS.md`; and
- one ZIP result bundle next to the checkout.

`submission.txt` records the pinned commit, checkout, result directory, array
job, collector job, and bundle path.

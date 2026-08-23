# Stage 57: persistent positive four-population closure

## Scope

This stage continues the Riemann35 moment-closure study for Rodney's oblique,
nonzero-heat-flux state. It does not run or train the separate FP-PINN model.
The positive QMC calculation is an internal reference for the implemented
collision operator; it is not MD or DSMC validation.

Stage 56 made the projected-tail time integration converged and reduced the
heat-flux error below 1%, but its positive two-population target manifold still
gave 11.23% full-third-tensor and 19.83% trace-free errors. Stage 57 therefore
changes the algebraic closure rather than the time integrator.

## Method change

The closure retains the four labeled positive Gaussian populations in Rodney's
regularized initial state. Each population stores one probability, a three-
component mean, and a symmetric 3x3 covariance. Together with density, this is
a compact 41-scalar latent state. No velocity particles or quadrature nodes are
retained between steps.

Each step uses positive Gauss-Hermite nodes only to evaluate the cubic
Fokker-Planck drift and maps the updated nodes back to the mean and covariance
of their original population. A constrained Newton projection then:

1. preserves mixture momentum and the full second central moment;
2. preserves the unprojected trace-free third tensor;
3. imposes the exact collision-model heat-flux relaxation
   `q_new = exp(-2 Pr dt/tau) q_old`;
4. accepts only positive population weights and SPD covariance matrices.

The heat-flux rate is derived from the collision operator. No QMC-fitted
closure parameter is used; QMC is evaluated only after the method is frozen.

## Frozen runs

Six jobs run concurrently:

1. positive QMC reference at `dt/tau=0.0003125`, 32768 nodes per initial
   Gaussian component, four independent Sobol scramblings;
2. the Stage-56 exact-Strang method at `dt/tau=0.0003125` as a control;
3. persistent four-population closure at `dt/tau=0.0025`;
4. persistent four-population closure at `dt/tau=0.00125`;
5. persistent four-population closure at `dt/tau=0.000625`;
6. persistent four-population closure at `dt/tau=0.0003125`.

All deterministic runs use five marginal quadrature nodes, so the refinement
study changes only the time step.

## Qualification gates

- all six runs complete and QMC scramble spread is below 3%;
- the three successive full-third-tensor changes contract monotonically and
  the finest change is below 1.5%;
- heat-flux, full-third-tensor, and trace-free errors are below 1%, 3%, and 5%;
- every normalized component RMSE is below 3%, and every component is within
  the larger of 20% of its reference norm or two reference SEM norms;
- collision invariants and selected H2 realizability pass;
- weights and covariances remain positive, the full heat-flux projection is
  accepted, and its residual remains below `1e-8`;
- the initial four-Gaussian moment/tail/source audit is exact to `1e-9`;
- the retained latent state contains no more than 41 scalars.

No gate is relaxed by the collector. A scientific failure still produces a
complete result bundle and a successful collector job.

## Local structural test

```bash
python riemann35_patch/stage57_persistent_four_population/test_persistent_mixture.py
```

## Unity submission

Use the exact 40-character commit printed for this stage:

```bash
MOMENT_STAGE57_COMMIT=<commit> bash riemann35_patch/stage57_persistent_four_population/submit_unity_stage57.sh
```


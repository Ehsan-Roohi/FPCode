# Stage 56: exact projected-tail time-integration gate

## Scope

This stage continues the Riemann35 moment-closure study for Rodney's oblique,
nonzero-heat-flux state. It does not run or train the separate FP-PINN model.
The positive QMC calculation is used only as an internal reference for the
implemented collision operator; it is not MD or DSMC validation.

Stage 55 reduced the heat-flux history error to about 0.46%, but its selected
35+49 projected-tail candidate changed by 6.21% after halving the time step.
The coarse agreement therefore could not qualify the closure.

## Method change

Stage 56 keeps the same 35 retained moments and 49 M5/M6 memory scalars. The
positive two-population projection target is rebuilt from the current 35
moments and discarded after use. No velocity microstate is retained.

The stiff tail relaxation with `tau_tail/tau=0.01` is integrated analytically:

```text
tail_new = target + exp(-dt/tau_tail) * (tail_old - target)
```

The qualified candidate uses symmetric Strang composition:

1. exact half tail relaxation at the old 35-moment state;
2. one guarded dynamic 35+49 collision-source step;
3. exact half tail relaxation at the new 35-moment state.

The Stage-55 post-step Lie split is retained as a control.

## Frozen runs

Six jobs run concurrently:

1. positive QMC reference at `dt/tau=0.0003125`, 32768 nodes per initial
   Gaussian component, four independent Sobol scramblings;
2. Stage-55-style Lie control at `dt/tau=0.0025`;
3. exact Strang at `dt/tau=0.0025`;
4. exact Strang at `dt/tau=0.00125`;
5. exact Strang at `dt/tau=0.000625`;
6. exact Strang at `dt/tau=0.0003125`.

All runs use five marginal quadrature nodes. Thus the time study does not mix
time-step and quadrature refinement.

## Gates

- all six runs complete;
- QMC scramble spread below 3%;
- three successive Strang changes contract monotonically;
- finest successive full-third-tensor change below 3%;
- heat-flux error below 1%;
- full-third-tensor error below 3%;
- trace-free third-tensor error below 5%;
- every normalized component RMSE below 3%;
- collision invariants, H2 realizability, positive target weights, and zero
  negative target mass.

No gate is relaxed when the result is collected. A scientific failure still
produces a complete bundle and a successful collector job.

## Local test

```bash
python riemann35_patch/stage56_tail_time_integrator/test_tail_time_integrator.py
```

## Unity submission

Use the exact 40-character commit printed for this stage:

```bash
MOMENT_STAGE56_COMMIT=<commit> bash riemann35_patch/stage56_tail_time_integrator/submit_unity_stage56.sh
```

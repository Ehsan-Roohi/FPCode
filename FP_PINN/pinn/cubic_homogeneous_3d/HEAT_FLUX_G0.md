# Heat-flux G0: analytic gate and production protocol

## Governing target

For the homogeneous cubic FP closure in this directory,

```text
Qx(0) = 0.25
dQx/dt = -(4/3) nu Qx
Qx(t) = 0.25 exp[-(4/3) nu t]
```

At the default `nu=1`, the exact endpoint is
`Qx(1)=0.065899...`, the exact decay rate is `4/3`, and the implied effective
Prandtl number relative to the stress rate `2 nu` is `2/3`.

The analytic history is the primary heat-flux validation target.  The particle
history is retained as an independent implementation cross-check.

## Frozen gates

All thresholds are relative L2 error over the saved analytic `Qx(t)` history;
equality passes.

| Level | Threshold | Meaning |
|---|---:|---|
| continuation | 5% | sufficient to continue the FP-PINN line |
| primary | 2% | accepted main quantitative result |
| publication | 1% | publication-grade heat-flux history |

A heat-flux checkpoint must also pass the conservation, marginal, exact-IC,
positivity, portable-reload, and transverse-symmetry checks.  The ordinary
`--strict-gate` exit status is tied to the 5% continuation level; 2% and 1%
are reported separately in `metrics.json`.

## Loss and quadrature

Training remains particle-free.  G0 combines:

- the pointwise FP residual;
- the FP residual projected onto `v |v|^2`;
- the common-random-number moment-rate residual
  `dQx/dt + (4/3) nu Qx`;
- mass, momentum, and energy penalties;
- an exactly reweighted core/initial/broad-tail proposal; and
- four rotating fixed antithetic panels for the heat-flux production run.

## Production run

`RUN_HEAT_FLUX_G0_UNITY.sh` submits only array task 2.  It is a new 30,000-epoch
production run, not a 500-epoch pilot and not a short continuation.  It writes
every 2,500-epoch checkpoint, independently evaluates them, selects the best
admissible checkpoint using analytic-Qx error, and packages all outputs.

Run from the repository root on Unity:

```bash
bash FP_PINN/pinn/cubic_homogeneous_3d/RUN_HEAT_FLUX_G0_UNITY.sh
```

For a plumbing-only pilot, explicitly set `FP_STAGE2_EPOCHS=500`.  Such a run
must always be labelled `pilot`; it is not evidence for any production gate.

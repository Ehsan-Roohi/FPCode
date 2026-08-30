# Stress G2 — deterministic structure-preserving cubic-FP PINN

G2 is the preregistered continuation after the accepted heat-flux G1 run
(`63737422`).  It tests whether the same PINN construction works for an
independent second-moment relaxation, rather than fitting only the third
Hermite heat-flux mode.

## Scientific case

The initial density is the positive axisymmetric Gaussian

```text
cov(f0) = diag(1.6, 0.7, 0.7),  mass = 1,  mean = 0,  trace(cov) = 3.
```

For the repository cubic Fokker--Planck closure,

```text
Delta P(t) = Pxx(t) - Pperp(t) = 0.9 exp(-2 nu t).
```

The qualification trainer does **not** import or use this history.  Training
uses only the deterministic f-weighted PDE residual and the invariant-tilt
gauge penalty.  The exact history is used after training by `evaluate_g2.py`.

The ansatz contains the traceless second Hermite mode
`cx^2 - (cy^2+cz^2)/2`, with an amplitude that is bounded and exactly zero at
`t=0`.  An exponential tilt enforces mass, momentum and energy at every time.

## Preregistered array

| task | seed | purpose | enters overall verdict |
|---|---:|---|---|
| `stress_s1` | 20260911 | independent scratch fit | yes |
| `stress_s2` | 20260912 | independent scratch fit | yes |
| `stress_s3` | 20260913 | independent scratch fit | yes |
| `no_stress_mode` | 20260911 | labelled Hermite-mode ablation | no |

## Frozen gates

All three scratch seeds must satisfy every blocking gate and their stress-L2
spread must be at most 1 percentage point.

* analytic stress-history relative L2 <= 2%; <= 1% is publication level;
* train/fine quadrature error difference <= 0.5 percentage points;
* exact-tilt mass, momentum and energy drift <= 0.5%, 0.1%, 0.5%;
* deterministic marginals versus FV <= 3%;
* FV stress history versus the analytic moment law <= 0.5%;
* fitted stress decay rate within 5% of `2 nu`;
* zero heat flux, exact transverse symmetry, positive finite density, exact
  initial condition, exact axisymmetry and portable checkpoint reload.

Raw pre-tilt drift, full-field L2 and the held-out PDE residual are reported
diagnostics.  The no-mode ablation is reported but cannot make the overall
verdict pass or fail.

If the accepted G1 summary is present, aggregation also reports the measured
effective Prandtl number

```text
Pr_eff = mean(G1 heat-flux decay rate) / mean(G2 stress decay rate),
```

whose exact value for this closure is `2/3`.

## One-line Unity launch

Run `RUN_STRESS_G2_UNITY.sh` from a pinned checkout.  It submits four GPU tasks
and a fail-closed dependent CPU aggregator.  Default resolution and optimizer
cost are the same order as G1: about 4–8 GPU-hours total.

Outputs:

```text
FP_PINN_G2_JOB<JOBID>_STRESS_<VARIANT>_COMPLETE.zip
FP_PINN_G2_JOB<JOBID>_STRESS_<VARIANT>_COMPLETE.zip.sha256
G2_SEED_SUMMARY.md
G2_SEED_SUMMARY.json
```

## Article decision after G2

* `PASS`: heat flux and stress are both publication-grade and reproducible;
  proceed to G3 parameter generalization (`nu` and nonequilibrium amplitude)
  plus a common ablation table.
* `NO_GO` with an accurate FV reference: inspect optimization once.  Do not
  hide a failed seed or loosen a frozen gate.
* A strong full paper still needs G3 generalization, timing/cost reporting and
  one spatial benchmark.  G1+G2 alone support the homogeneous-method section,
  not yet the final manuscript claim.

# Stage 28: localized kinetic pocket and measured speed gate

Stage 28 keeps the causal positive-memory rules qualified in Stage 27, but
uses a localized regularized four-delta pocket inside an equilibrium
background.  The expensive closure-disagreement sensor is sampled at an
explicit cadence.  Skipped samples hold the activation mask and release
counters fixed; they never extrapolate a safe release decision.
Inactive cells already matching their local Maxwellian to `1e-12` bypass an
otherwise redundant macro collision solve; the runner reports every shortcut.

The qualification requires all of the following in one run:

- synchronous donor provenance and positive DVM masses;
- realizability, finite-volume balance, and micro/macro synchronization;
- final and space-time `M400`/predictive `M420` errors below 3% against a
  refined positive DVM;
- peak and mean kinetic fractions below 50%; and
- measured adaptive wall time below the same-grid coarse Full-DVM wall time.

Smoke test from the repository root:

```bash
python -m riemann35_patch.stage28.run_localized_pocket --mode smoke --output results/riemann35_stage28/smoke
```

Unity submission is pinned to a 40-character Git commit:

```bash
FP_STAGE28_COMMIT=<commit> bash riemann35_patch/stage28/submit_unity_stage28.sh
```

This is a numerical qualification for the implemented cubic FP operator, not
independent MD/DSMC evidence of physical fidelity.

# Stage 29: advecting causal kinetic front

Stage 29 moves the localized regularized four-delta pocket through an
equilibrium background.  A retained-moment alarm alone reacts one cell too
late for predictive `M420`, so this stage adds a cheap causal front detector:
the incoming half-range DVM flux from an active neighbour is compared with the
known positive background carrier.  The detector reuses the frozen Stage-25
`tail_on = 0.40` threshold; no physical or hysteresis threshold is retuned.

Every front birth uses a positive upwind carrier--donor proposal followed by
the existing entropy projection.  The proposal contains only known kinetic
data and never reconstructs an unidentified tail from the 35 moments.

Qualification requires at least one new causal front birth, final and
space-time `M400`/`M420` errors below 3%, a peak kinetic fraction below 50%,
positive/conservative evolution, and measured speedup over the same-grid
coarse Full-DVM reference.

Run from the repository root:

```bash
python -m riemann35_patch.stage29.run_advecting_front --mode workstation --output results/riemann35_stage29/local
```

This is numerical validation of the implemented cubic FP operator.  It is not
independent MD/DSMC evidence of physical fidelity.

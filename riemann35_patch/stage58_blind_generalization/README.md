# Stage 58: prospective blind generalization

## Scope

This stage tests whether the Stage-57 compact positive four-population closure
generalizes beyond Rodney's single oblique Riemann35 state. It does not run or
train the separate FP-PINN model. QMC is an internal reference for the
implemented homogeneous collision operator, not MD or DSMC validation.

The closure equations and every accuracy threshold are inherited without
fitting to the Stage-58 references. The only numerical change is that the
constrained projection is allowed enough Newton iterations for translated and
scaled states; the physical constraints and heat-flux law are unchanged.

## Prospective protocol

The case registry is deterministic and SHA-256 fingerprinted before QMC is
evaluated. Every result records `qmc_used_to_define_case=false` and
`closure_parameters_refit=false`.

One model-only preflight limitation is disclosed explicitly. Planned Gaussian
regularization fractions 0.020 and 0.025 were removed before any QMC reference
was evaluated because their closure-only fine/coarse full-third history changes
were 8.3% and 7.2%. Stage 58 therefore makes no claim below the original 0.030
regularization fraction. This exclusion is a declared domain boundary, not an
accuracy result.

## Frozen cases

Five array tasks run concurrently:

1. the exact Stage-57 oblique state as an independent anchor repeat;
2. a hot, dense, translated, and newly rotated state;
3. a broad, low-density, translated, and scaled state;
4. a state with unseen weights and population centers;
5. a fully three-dimensional state with distinct anisotropic SPD covariances.

All states contain four positive Gaussian populations and all ten independent
central third-order components are nonzero. The anchor is not counted as a
blind case.

For each case, Unity runs:

- eight independent QMC scrambles with 65536 points per population at
  `dt/tau=0.0003125`;
- the frozen persistent closure at `dt/tau=0.000625`;
- the same closure at `dt/tau=0.0003125`.

The final state still stores only 41 scalars. No velocity nodes are retained
between deterministic steps.

## Qualification gates

The anchor and every one of the four blind cases must pass individually:

- QMC scramble spread below 2%;
- fine/coarse full-third history change below 1%;
- heat-flux, full-third, and trace-free history errors below 1%, 3%, and 5%;
- maximum normalized component RMSE below 3%;
- every component within the larger of 20% of its reference history norm or
  two QMC SEM norms;
- conserved mass, momentum, and energy trace;
- exact initial moment/tail/source audit;
- positive weights, SPD covariances, H2 realizability, full projection
  fraction, projection residual below `1e-8`, and at most 41 stored scalars;
- registry fingerprint match and explicit confirmation that no reference was
  used for case design or parameter fitting.

A failed scientific gate still produces the complete bundle and a successful
collector job.

## Local structural test

```bash
python riemann35_patch/stage58_blind_generalization/test_blind_generalization.py
```

## Unity submission

Use the exact 40-character commit printed for this stage:

```bash
MOMENT_STAGE58_COMMIT=<commit> bash riemann35_patch/stage58_blind_generalization/submit_unity_stage58.sh
```

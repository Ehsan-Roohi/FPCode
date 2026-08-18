# Validated Stage-26 local reference result

This directory freezes the accepted local production run of the regularized
four-delta audit at commit
`96284afa0de423f217d65149b7ad5a36e9d76a41`.

## Frozen controls

- `dt/tau = 0.0025`, `t/tau = 1.0`;
- four independent QMC scramblings;
- 8192 Sobol points per component for both Full-FP and adaptive ensembles;
- 3% common-Gaussian regularization of the four planar delta populations;
- history gates at 3% for `M400` and untransported `M420`.

## Decision

All 12 selected-method gates passed.  Full-FP scramble spreads were 1.18% for
`M400` and 2.67% for `M420`; causal-memory history errors were 0.48% and 1.74%,
respectively.  The no-donor persistence probe passed and all Full-FP/adaptive
particle weights stayed positive.

The algebraic comparisons did not meet the selected-method accuracy level.
Stage-9 errors were 10.53% (`M400`) and 18.78% (`M420`).  Grad/GQMOM errors
were 4.16% and 78.13%; its signed source quadrature reached 82 negative
weights, 4.861% negative mass, and a physically inadmissible minimum even-tail
moment of -0.129465.

This validates the causal adaptive closure against an independently scrambled
Full-FP QMC ensemble for the same cubic FP operator.  It is not MD/DSMC
validation of the physical collision operator.

## Files and integrity

- `STAGE26_RESULTS.md`: readable gate report;
- `stage26_four_delta_summary.json`: complete machine-readable result;
- `stage26_four_delta_errors.csv`: all history errors;
- `stage26_four_delta_histories.png`: comparison plot;
- `STAGE26_LOCAL_VALIDATED_20260814.zip`: complete histories, method summaries,
  logs, report, table, and plot.

ZIP SHA-256:
`519724960a8ec53fecc010853ca66eb9f15ebb4f18cedddaed97c3d9eae1f418`.

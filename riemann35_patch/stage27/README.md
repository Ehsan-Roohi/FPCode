# Stage 27: spatial causal-donor crossing test

This stage is a deterministic workstation-sized follow-on to the homogeneous
Stage-26 four-delta audit.  Two unequal positive Gaussian mixtures meet at a
one-dimensional interface.  Their third moments trigger the frozen kinetic
sensor, while the adaptive solver initially retains positive DVM memory only
in the two cells touching the interface.

The test is intentionally not another normal-shock campaign.  Its purpose is
to qualify the spatial lifecycle before spending cluster time:

- every kinetic birth must cite a boundary inflow or a neighbour that was
  already active at the beginning of the step;
- a newly born cell cannot donate again during the same step;
- positive coarse and refined Scharfetter--Gummel DVM solutions provide the
  independently discretized reference pair;
- positivity, realizability, finite-volume balance, micro/macro synchronization,
  retained `M400`, and predictive `M420` are audited;
- pre-donor `M420` error is reported, not hidden: an inactive cell does not yet
  possess the unidentified tail.

Run the workstation qualification from the repository root:

```bash
python -m riemann35_patch.stage27.run_crossing_donor --mode workstation --output results/riemann35_stage27/local
```

This establishes numerical behavior for the implemented cubic FP operator.  It
does not replace a later independent MD/DSMC physical validation.

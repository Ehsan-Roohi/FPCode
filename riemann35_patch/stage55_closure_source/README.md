# Stage 55: locate the third-order closure-source error

Stage 54 showed that the positive HyQMOM-35 finite-mixture path has about
25% history error in the third-order tensor, while a persistent kinetic tail
is accurate but never leaves micro mode.  Stage 55 asks the narrower causal
question: which unavailable high-order information corrupts the instantaneous
third-order FP source?

The audit follows the same oblique heat-flux state at `t/tau = 0, 0.1, 0.25,
0.5, 0.75, 1`.  A positive Full-FP Sobol ensemble supplies the evolving state.
At every audit time it evaluates the analytic generator, so the comparison is
not contaminated by a finite-difference time derivative.

Six frozen paths are compared:

1. exact positive-node source at the base QMC resolution;
2. exact positive-node source at the refined QMC resolution;
3. exact coefficients with only the projection tail replaced by the retained
   Gaussian-tail closure;
4. retained/Gaussian-tail coefficients with an otherwise exact node source;
5. retained/Gaussian-tail coefficients and Gaussian-tail projection together;
6. a positive compact Gaussian-mixture quadrature reconstructed from the 35
   moments, used for both coefficients and projection.

This separates coefficient-system `M5` error from source-projection `M5`
error.  The compact path is a diagnostic candidate, not a predeclared
success.  The collector reports all ten third-order components, the heat-flux
contraction, the trace-free tensor, the three contracted fifth moments, and
the FP coefficients.  Reference node convergence must be below 3% before a
causal conclusion is accepted.

The scientific outcome is classified as `DIAGNOSIS_PASS` when the reference
is converged and one isolated substitution explains at least 70% of the full
Gaussian-source error.  `COMPACT_PASS` additionally requires a positive
compact representation with third-source error below 3%, trace-free-source
error below 5%, retained-moment residual below `2e-8`, and at least a 100-fold
reduction in support relative to refined QMC.  A diagnosis can therefore pass
without claiming that the compact closure has already been solved.

Local smoke test:

```bash
python riemann35_patch/stage55_closure_source/test_stage55_source.py
```

Unity submission:

```bash
FP_STAGE55_COMMIT=<40-character-commit> bash riemann35_patch/stage55_closure_source/submit_unity_stage55.sh
```

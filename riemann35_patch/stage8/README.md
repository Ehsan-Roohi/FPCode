# Stage 8: structural audit before spatial deployment

Stage 8 is an adversarial audit, not a success gate. It records three issues
that the short-time in-family benchmark cannot resolve:

1. boundedness of the reconstructed sixth moment as skewness approaches zero
   from either side at fixed positive fourth cumulant;
2. realizability and reconstruction viability through `t/tau = 3`;
3. prototype per-cell source cost relative to a single-Gaussian tail.
4. a genuinely nonseparable two-Gaussian state with vector means and two
   different full covariance tensors, for which a marginal tensor product is
   not an exact multivariate reconstruction.

Run locally with

```bash
python3 riemann35_patch/stage8/run_structural_audit.py \
  --dt 2.5e-3 \
  --t-final 3.0 \
  --timing-repeats 100 \
  --output results/riemann35_stage8/structural_audit.json
```

The unequal-variance location--scale branch is expected to pass the
zero-skewness continuity test. Spatial deployment remains blocked if any
long-time trajectory leaves the realizability cone or the reconstruction
fails. A deterministic velocity-space reference and head-to-head
Gaussian-GQMOM/Grad-HyQMOM comparisons remain separate required gates.

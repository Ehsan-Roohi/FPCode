# Stage 20: causal positive tail-memory lifecycle

This stage turns the Stage-19 offline sensor into a homogeneous adaptive
macro--micro collision lifecycle.

The macro path is the Stage-9 realizability-aware Gaussian-mixture finite map.
The micro path is a persistent positive weighted QMC velocity state. A
microstate may be created only from a known physical initial/inflow
decomposition or supplied by a causal donor. `project_positive_microstate`
uses positive entropy reweighting on that support to match all 35 transported
moments. It fails explicitly if the target is outside the support's discrete
convex hull. The reverse projection is the direct positive quadrature moment
map.

The source-disagreement alarm is gated by standardized skewness to reject the
transient symmetric counter-stream false alarm found during this audit. The
tail-disagreement alarm remains independent. Separate on/off thresholds,
minimum active time, and consecutive safe evaluations provide hysteresis.

Run from the repository root:

```bash
python riemann35_patch/stage20/run_hysteretic_tail_memory_audit.py \
  --stage11 results/riemann35_stage11 \
  --stage14 results/riemann35_stage14 \
  --output results/riemann35_stage20 \
  --workers 4
```

Acceptance gates are:

1. rare-beam `M400` error and scramble spread below 3% against both Stage-11
   particles and Stage-14 positive QMC;
2. no blocked causal activation on the six Stage-11 histories;
3. no representation chatter;
4. positive realizable histories;
5. conservative positive projection; and
6. one clean micro-to-macro transition in a safe causal-birth control.

This stage does not implement spatial microstate transport or an independent
DVM/spectral reference.

# FPCode scientific state

This file is the durable handoff record for future work.  Update it whenever a
stage changes the selected method or the next scientific decision.

## Current accepted state: Stage 27

The accepted method remains causal adaptive macro--micro evolution with
positive kinetic memory.  An instantaneous algebraic tail cannot be treated as
a universal closure for the 35 retained moments because degree-five and
degree-six information is not identifiable from moments through degree four.

Stage 27 moved the Stage-26 method into a lightweight one-dimensional spatial
crossing-population test.  It also fixed an activation-order defect: a
microstate born during a step may not donate again until the next step.  Every
birth now records its inflow/neighbor provenance and neighbor donors must have
been active at the beginning of the step.

The validated workstation run is archived under
`stage27/reference_results/local_validated_20260814`.

Accepted quantitative result:

- all numerical and causal contracts passed;
- the kinetic wavefront reached all 12 cells at step 3, with eight earlier
  requests safely blocked until a donor arrived;
- final adaptive errors against the refined positive DVM were 0.1573% for
  `M400` and 0.2937% for `M420`;
- post-activation space-time errors were 0.1275% for `M400` and 0.2751% for
  `M420`;
- coarse/refined final `M420` difference was 0.3219%;
- maximum finite-volume balance residual was `1.06e-13`, maximum micro/macro
  synchronization residual was `4.46e-12`, and every DVM mass stayed positive;
- the algebraic macro comparison missed the final refined DVM by 5.22% in
  `M400` and 16.52% in `M420`.

The accuracy result is not a performance result.  The mean active fraction was
92.86%, every cell was kinetic after step 3, and the unoptimized adaptive path
took 7.525 times the coarse Full-DVM wall time because its sensor/projection
overhead remained active.  Do not claim a speedup from Stage 27.

## Previous accepted state: Stage 26

Stage 26 tested Rodney Fox's homogeneous regularized four-delta construction:
unit mass, zero momentum, unit energy trace, nonzero third moments, and four
unequal planar populations.  The validated local run is archived under
`stage26/reference_results/local_validated_20260814`.

Accepted quantitative result:

- all 12 selected-method gates passed;
- Full-FP QMC spread: 1.18% (`M400`), 2.67% (`M420`);
- causal-memory error: 0.48% (`M400`), 1.74% (`M420`);
- exact initial invariants and maximum dynamic invariant errors below
  `1.5e-11`;
- positive Full-FP/adaptive particle weights;
- forced no-donor persistence probe passed;
- Stage-9 comparison: 10.53% (`M400`), 18.78% (`M420`);
- Grad/GQMOM comparison: 4.16% (`M400`), 78.13% (`M420`), with signed negative
  mass and a negative even-tail prediction.

## Binding method constraints

- Microstate birth must be causal: known initialization, kinetic inflow, or an
  active spatial donor.  It must not be reconstructed after an alarm from the
  same 35 moments.
- Particle/DVM weights must stay positive; retained moments must stay
  realizable; mass, momentum, and energy must be conserved to the declared
  tolerance.
- `M400` and the untransported predictive observable `M420` remain primary
  quantitative gates, with independently evolved references and a 3% target.
- Full-FP/QMC agreement validates closure behavior for the implemented cubic
  FP operator, not the operator's physical fidelity.  Independent MD/DSMC
  validation remains a later task.
- Do not repeat the Stage-26 Unity run unless independent reproducibility is
  explicitly needed; the accepted local result and full archive are frozen in
  Git.

## Next scientific step

Optimize the spatial lifecycle before returning to an expensive normal shock:
avoid recomputing costly release sensors in every active cell at every step,
then test a localized non-equilibrium pocket whose kinetic region need not fill
the whole domain.  The next gate must preserve Stage-27 causality and accuracy
while demonstrating an active fraction below 50% and an actual wall-time gain
against the same coarse Full-DVM reference.  Independent MD/DSMC validation
remains necessary later for physical fidelity.

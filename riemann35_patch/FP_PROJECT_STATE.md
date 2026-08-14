# FPCode scientific state

This file is the durable handoff record for future work.  Update it whenever a
stage changes the selected method or the next scientific decision.

## Current accepted state: Stage 28 (local qualification; Unity reproduction pending)

Stage 28 closes the performance defect exposed by Stage 27 on a localized
regularized four-delta pocket in an equilibrium spatial background.  The
spatial lifecycle now accepts an explicit sensor cadence.  Skipped samples
hold activation and release state fixed, and the default cadence remains one,
so earlier stages are unchanged.  An additional opt-in shortcut bypasses the
macro collision map only when the transported 35 moments match their local
Maxwellian to a declared relative tolerance (`1e-12` in Stage 28).

The full 48-cell, 24-step local qualification is archived under
`stage28/reference_results/local_validated_20260814`.

Accepted local quantitative result:

- all numerical, localization, and measured-performance gates passed;
- mean, peak, and final kinetic fractions were all 16.67%;
- final adaptive errors against the refined positive DVM were 0.2092% for
  `M400` and 0.4974% for predictive `M420`;
- space-time errors were 0.1294% (`M400`) and 0.4689% (`M420`);
- the adaptive path evaluated the sensor 144 times instead of the 2,304
  Stage-27-style evaluations and used 380 audited Maxwellian fixed-point
  shortcuts;
- adaptive wall time was 0.8757 times the same-grid coarse Full-DVM wall time,
  a measured 1.142x speedup;
- maximum finite-volume balance residual was `7.41e-14`, maximum micro/macro
  synchronization residual was `3.84e-15`, and all DVM masses stayed positive.

The same pinned configuration must still be reproduced on Unity before
calling this a cluster-validated performance result.  No independent MD/DSMC
physical validation has yet been performed.

## Previous accepted state: Stage 27

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

First reproduce the pinned Stage-28 qualification on Unity and compare its
measured adaptive/coarse-DVM timing ratio with the frozen local result.  If the
cluster run passes, the next scientific gate is a moving/advecting localized
nonequilibrium region that forces at least one new causal neighbor birth while
retaining the sub-50% kinetic fraction and measured speedup.  Independent
MD/DSMC validation remains necessary later for physical fidelity.

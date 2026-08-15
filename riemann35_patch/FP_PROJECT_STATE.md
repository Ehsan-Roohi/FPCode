# FPCode scientific state

This file is the durable handoff record for future work.  Update it whenever a
stage changes the selected method or the next scientific decision.

## Current accepted state: Stage 30 (local qualification)

Stage 30 closes the complete moving-front lifecycle gate.  It doubles the
Stage-29 horizon to 48 steps and observes both positive causal kinetic birth
ahead of the advecting four-delta pocket and retirement of kinetic memory
behind it.  Release retains the frozen Stage-25 off thresholds, eight verified
safe observations, and twenty-step minimum dwell.

The lifecycle now permits separate activation and release sensor cadences.
Stage 30 audits inactive-cell activation every eight steps and active-cell
release every four steps.  An opt-in causal-candidate filter skips the
expensive activation closure in inactive cells without kinetic inflow or a
neighbour active at the beginning of the step; those cells cannot legally be
born.  Skips are counted explicitly, and all earlier stages retain their dense
default audit.

The full 48-cell, 48-step workstation qualification is archived under
`stage30/reference_results/local_validated_20260814`.

Accepted local quantitative result:

- all numerical, causal, lifecycle, localization, accuracy, and measured-
  performance gates passed;
- ten causal kinetic-front births and four releases occurred; cell 17 was
  released behind the leading cell at step 29 and stayed inactive for the
  required four-step audit window;
- mean kinetic fraction was 19.13%, and peak/final fractions were 25.00%;
- final adaptive errors against the refined positive DVM were 0.4487% for
  `M400` and 0.9273% for predictive `M420`;
- space-time errors were 0.2109% (`M400`) and 0.7310% (`M420`);
- 210 impossible no-donor activation sensors were skipped and only 135
  expensive closure sensors were evaluated;
- adaptive wall time was 0.8609 times the same-grid coarse Full-DVM wall time,
  a measured 1.162x speedup;
- maximum finite-volume balance residual was `1.73e-12`, maximum micro/macro
  synchronization residual was `5.89e-11`, and all DVM masses stayed positive.

This is local numerical validation of the implemented cubic FP operator.  A
Unity reproduction remains useful for portable cluster timing but is not a
scientific blocker.  No independent MD/DSMC physical validation has yet been
performed.

## Previous accepted state: Stage 29

Stage 29 closes the moving-front gate proposed after Stage 28.  A localized
regularized four-delta pocket with unit positive x velocity is advected through
an equilibrium background.  Because the retained-moment sensor reacts one
cell too late for predictive `M420`, the adaptive lifecycle now has an opt-in
causal front detector: it measures the incoming half-range discrepancy between
an already-active DVM donor and a known positive background carrier.  The
detector uses the frozen Stage-25 `tail_on = 0.40` threshold.

Every front birth forms a positive directional carrier--donor proposal and
then applies the existing entropy projection to the transported 35 moments.
It uses only already-known kinetic information; no unidentified tail is
reconstructed from the retained moments.  Omitting the new options preserves
the Stage-27/28 lifecycle exactly.

The full 48-cell, 24-step workstation qualification is archived under
`stage29/reference_results/local_validated_20260814`.

Accepted local quantitative result:

- all numerical, causal, localization, accuracy, and measured-performance
  gates passed;
- four kinetic-front births occurred, three at the right-moving leading front;
- mean kinetic fraction was 16.83%, and peak/final fractions were 20.83%;
- final adaptive errors against the refined positive DVM were 0.2300% for
  `M400` and 0.4280% for predictive `M420`;
- space-time errors were 0.1099% (`M400`) and 0.6405% (`M420`);
- adaptive wall time was 0.8010 times the same-grid coarse Full-DVM wall time,
  a measured 1.248x speedup;
- maximum finite-volume balance residual was `3.45e-15`, maximum micro/macro
  synchronization residual was `1.84e-13`, and all DVM masses stayed positive.

This is local numerical validation of the implemented cubic FP operator.  A
Unity reproduction is useful for portable cluster timing but is not a blocker
for the scientific sequence.  No independent MD/DSMC physical validation has
yet been performed.

## Previous accepted state: Stage 28

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

The archived timing is a local workstation result, not a cluster benchmark.

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

Apply the now-frozen complete lifecycle to a held-out one-dimensional normal
shock case that was not used to tune its thresholds.  Compare density,
velocity, temperature, stress, heat-flux, `M400`, and predictive `M420`
profiles against positive coarse/refined Full-DVM references while preserving
causality, positivity, conservation, localized support, and measured speedup.
A Unity reproduction remains an optional portability/timing check.
Independent DSMC validation remains necessary later for physical fidelity.

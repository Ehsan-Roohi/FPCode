# FPCode scientific state

This file is the durable handoff record for future work.  Update it whenever a
stage changes the selected method or the next scientific decision.

## Current accepted state: Stage 26

The accepted method is causal adaptive macro--micro evolution with positive
kinetic memory.  An instantaneous algebraic tail cannot be treated as a
universal closure for the 35 retained moments because degree-five and
degree-six information is not identifiable from moments through degree four.

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

Move from homogeneous relaxation to a lightweight spatial causal-donor test
before returning to an expensive normal shock.  The next experiment should
use known kinetic inflow/donor states, exercise transport across an interface,
compare positive Full-DVM/QMC and adaptive histories, and audit causal birth,
positivity, realizability, conservation, and `M400`/`M420` accuracy.

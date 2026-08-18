# Stage 10 general-state audit

## Scope

Stage 10 implements the Appendix-C Grad-HyQMOM reconstruction with
Gaussian-GQMOM univariate measures and compares it with the corrected Stage-9
principal-axis tensor Gaussian mixture.  The audit contains
292 realizable states: the five Stage-9 cases, anisotropic
rare-hot populations, near-delta counter-stream/crossing/rare-beam states up
to Ma=100, and 256 seeded random two- to four-component multivariate Gaussian
mixtures.

The physical cubic coefficient called beta in the code (Lambda in the report)
is explicit:

    Lambda = -nu ||Pi||_F^2 / (tr P)^(7/2) <= 0.

Thus Lambda vanishes for isotropic stress.  The isotropic rare-hot exact-OU
test validates the OU path and conservation plumbing; it is not evidence for
accuracy when the nonlinear cubic correction is active.

The two marginal cubics are kept distinct:

    equal variance:       2 v^3 + kappa_4 v - kappa_3^2 = 0,
    equal-weight scale:   2 w^3 + kappa_4 w - kappa_3^2/3 = 0.

## Main numerical findings

* Appendix-C Grad-HyQMOM reconstructed all 292 of
  292 states.  Its median and 90th-percentile active cubic-FP
  source errors were 36.372%
  and 291.048%.
* The Stage-9 tensor closure reconstructed 283 of
  292 states.  Its corresponding median and 90th-percentile
  source errors were 78.545%
  and 1274.596%.
These aggregate medians are **not a global accuracy ranking**.  The empirical
CDFs cross and the 292-state set is deliberately heterogeneous.  The exact
tail used by this audit is the tail of the generating Gaussian mixture, so all
accuracy statements are conditional on that known family.  By physical
family:

* counter-stream median source error is
  0.000%
  for Stage 9 and
  4.555%
  for Grad/GQMOM;
* crossing median source error is
  0.000%
  versus
  2.113%;
* rare-beam median source error is
  0.000%
  versus
  26.462%;
* anisotropic rare-hot median source error is
  3.914%
  versus
  6.723%; and
* the aggregate reversal is driven mainly by the 256
  rotated random mixtures, where Grad/GQMOM has the smaller median error.

Thus Stage 9 remains an in-family accuracy reference for separable, axis-aligned
mixtures; guarded Grad/GQMOM is the continuous, cheaper, more robust baseline.
Neither method dominates the other in accuracy.

* The unguarded Stage-9 finite map completed 283 of
  292 initial steps and produced 7
  negative-margin states.
* Even the exact generating-mixture source followed by forward Euler left the
  cone in 24 of 292
  states (8.2%); on the boundary band the
  count was 23 of
  85.  A guard therefore controls both closure
  error and finite-step time-discretization error.
* Exact OU splitting plus a scalar realizability limiter completed
  292 of 292 initial steps with zero
  realizability failures.  The limiter activated in 101
  states; its minimum value was 8.047e-04.
* Both the Stage-9 finite map and guarded Grad/GQMOM reached t=tau on all six
  selected trajectories, including counter-stream and
  crossing states at Ma=20, a counter-stream at Ma=100, and the rare beam that
  defeated the *unguarded* Grad source.  In the Ma=20 rare-beam trajectory the
  guard was active in 14 of
  400 steps and the minimum lambda was
  0.675.  The per-step lambda history is stored
  in the JSON rather than asserted only in prose.
* The Stage-9 branch switch is genuinely discontinuous.  Across kappa_4=0 at
  s_3=1, an input perturbation of 2e-8 produces an M6 jump of 4.667.  The
  Gaussian-GQMOM jump is 2.25e-7 for the same perturbation.
* Prototype median evaluation time is roughly 0.032 s for Grad-HyQMOM versus
  0.315 s for the Stage-9 tensor reconstruction and 0.018 s for the single
  Gaussian closure.  The Appendix-C path is therefore about ten times faster
  than the Stage-9 tensor fit in this Python prototype.

## Interpretation

The guarded Appendix-C route is continuous across the former marginal branch
seam, cheaper than the Stage-9 tensor prototype, and realizable for the entire
292-state initial-step audit.  Those properties make it the appropriate robust
baseline for the next particle validation and for coupling to unchanged BFL
free transport.  They do not make it uniformly more accurate than Stage 9.

Only 19 states are in the deep-interior bin, so its small median
error must be reported with that sample count.  Extremely near-boundary random
mixtures can require strong limiting; in 70 of 85 boundary states the guard is
active and the global minimum lambda is 8.047e-04.
There the nonlinear correction can be almost extinguished.  The Grad source
quadrature is signed in 96.2% of states, with
median negative mass
16.1%; this is acceptable as a linear
source quadrature, not as a positive VDF reconstruction, and it explains why a
moment-cone guard is needed.

The next evidence gate is therefore a 16-seed particle comparison on [0,tau]
for **both** maps and all six selected trajectories.  Near the univariate
Hankel boundary, the next algorithmic repair should be a smooth degeneration
to QMOM as b2 approaches zero, not the Appendix-D transport wave-speed cap.
A deterministic Hermite reference remains the late-time accuracy gate.

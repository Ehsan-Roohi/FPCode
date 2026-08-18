# Stage 32 local development result (HOLD)

This directory freezes the selected 48-cell, 48-step Mach-2 development run
of the direction-aware causal precursor.  It is a scientifically useful
partial success, not a passed qualification and not a blind cross-case test.

- decision: `DEVELOPMENT_HOLD`;
- eight causal/front births, including two downstream `left_neighbor` births;
- both downstream births were predicted weighted-only events while the legacy
  mass signal remained below the frozen `0.40` threshold;
- three releases and zero four-step chatter events;
- mean/peak/final kinetic fractions: 16.37% / 20.83% / 20.83%;
- local speedup versus coarse Full-DVM: 1.503x;
- full-profile stress / heat-flux errors: 2.55% / 4.41%;
- shock-core stress / heat-flux / predictive-`M420` errors:
  2.06% / 2.32% / 0.73%;
- full-profile predictive-`M420` error: 0.44%;
- positivity, conservation, realizability, refinement, directionality,
  localization, lifecycle, and measured-performance contracts passed.

The sole physical-profile failure was full-domain heat flux.  Its remaining
error is dominated by weak macro far-field tails outside the shock core.  The
direction-aware rule is therefore not frozen, Stage 30 remains the last fully
passed qualification, and Mach 2.5 remains untouched.

# Stage 32 candidate A (HOLD)

Candidate A applied the mass, stress, heat-flux, and raw-`M420` half-range
signals instantaneously in both directions while preserving the frozen `0.40`
threshold.  Its full 48-cell, 48-step Mach-2 development run is retained as a
negative result.

- decision: `DEVELOPMENT_HOLD`;
- 14 causal births, all from `right_neighbor`, and zero downstream births;
- 10 weighted-only births, five releases, and one four-step chatter event;
- mean/peak/final kinetic fractions: 19.47% / 29.17% / 29.17%;
- local speedup versus coarse Full-DVM: 1.387x;
- stress / heat-flux errors: 5.83% / 9.28%;
- shock-core predictive `M420` error: 3.95%;
- positivity, conservation, realizability, refinement, localization, and
  measured-performance contracts passed.

The raw-`M420` weight reacted strongly to low-mass upstream tails but did not
identify the persistent downstream layer.  This candidate is not the frozen
Stage-32 rule.

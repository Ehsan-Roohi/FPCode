# Stage 29 frozen local result

This directory archives the accepted 48-cell, 24-step workstation
qualification produced on 2026-08-14.

- `STAGE29_ADVECTING_FRONT_RESULT.md` is the compact human-readable report.
- `stage29_advecting_front_summary.json` contains configuration, gates,
  diagnostics, timings, and birth provenance.
- `stage29_advecting_front_histories.npz` contains the numerical histories.
- `stage29_advecting_front.png` is the physical-space and moment-profile
  diagnostic figure.

The run passed all causal, numerical, localization, accuracy, and measured
performance gates.  It is local numerical evidence for the implemented cubic
FP operator, not independent physical validation or a portable cluster
benchmark.

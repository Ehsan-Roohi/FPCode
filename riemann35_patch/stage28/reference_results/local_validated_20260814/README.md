# Validated local Stage-28 reference

This directory freezes the full `unity` preset run performed locally on
2026-08-14 before cluster submission.  The run used 48 spatial cells, 24
steps, coarse `17x15x11` and refined `19x17x13` positive DVM grids, an
eight-step sensor cadence, and the opt-in `1e-12` Maxwellian fixed-point
shortcut.

Decision: `QUALIFICATION_PASS`.

The JSON file is the machine-readable source of truth.  The NPZ file contains
the complete retained-moment, predictive-`M420`, and active-mask histories;
the PNG is a physical profile/space-time visualization.  This workstation
result qualifies the deterministic configuration but does not substitute for
the separately submitted Unity reproduction or independent MD/DSMC physical
validation.

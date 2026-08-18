# Stage 31 held-out local result (HOLD)

This directory freezes the 48-cell, 48-step local workstation run of the
held-out Mach-2 normal shock.  It is a scientifically useful failed gate, not
a passed qualification.

- `STAGE31_HELDOUT_SHOCK_RESULT.md` is the concise result.
- `stage31_heldout_shock_summary.json` contains the configuration, all gates,
  metrics, birth provenance, and timing.
- `stage31_heldout_shock_profiles.csv` contains the final refined-DVM,
  coarse-DVM, and adaptive physical profiles.
- `stage31_heldout_shock_profiles.npz` stores float32 final moment/profile
  fields and the complete boolean active-mask history; the solver and gate
  calculations used float64 before this archival down-cast.
- `stage31_heldout_shock_profiles.png` is the physical line-profile and
  kinetic-support figure.

The adaptive method remained causal, positive, conservative, localized, and
1.370x faster than coarse Full-DVM locally.  The gate remained on hold because
normal-stress, heat-flux, and shock-core predictive `M420` errors exceeded 3%.

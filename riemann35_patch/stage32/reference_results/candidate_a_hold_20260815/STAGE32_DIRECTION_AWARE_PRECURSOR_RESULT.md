# Stage 32 direction-aware Mach-2 precursor result

- Decision: **DEVELOPMENT_HOLD**
- Case: Mach 2 development case; Stage 31 remains WORKSTATION_HOLD
- Steps / final time: 48 / 0.374661 tau
- Causal births / front births / releases: 14 / 14 / 5
- Downstream / weighted-only front births: 0 / 10
- Mean/peak/final kinetic fraction: 19.473% / 29.167% / 29.167%
- Full-profile adaptive errors rho/u/T: 0.4040% / 0.3849% / 0.2342%
- Full-profile adaptive errors stress/heat flux: 5.8321% / 9.2753%
- Full-profile adaptive errors M400/M420: 0.4670% / 1.9082%
- Space-time adaptive M400/M420 errors: 0.3209% / 1.0589%
- Adaptive/coarse-DVM wall-time ratio: 0.721x (1.387x speedup)
- Maximum balance / micro-macro sync residual: 8.980e-13 / 1.780e-10

This is a numerical development/cross-case test of the implemented cubic
Fokker-Planck model. It is not independent DSMC or experimental
validation of the collision operator's physical fidelity.

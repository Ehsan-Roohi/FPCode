# Stage 32 direction-aware Mach-2 precursor result

- Decision: **DEVELOPMENT_HOLD**
- Case: Mach 2 development case; Stage 31 remains WORKSTATION_HOLD
- Steps / final time: 48 / 0.374661 tau
- Causal births / front births / releases: 8 / 8 / 3
- Downstream / weighted-only front births: 2 / 2
- Mean/peak/final kinetic fraction: 16.369% / 20.833% / 20.833%
- Full-profile adaptive errors rho/u/T: 0.3626% / 0.4323% / 0.2137%
- Full-profile adaptive errors stress/heat flux: 2.5493% / 4.4057%
- Full-profile adaptive errors M400/M420: 0.5538% / 0.4384%
- Space-time adaptive M400/M420 errors: 0.3515% / 0.3083%
- Adaptive/coarse-DVM wall-time ratio: 0.665x (1.503x speedup)
- Maximum balance / micro-macro sync residual: 1.671e-14 / 4.602e-13

This is a numerical development/cross-case test of the implemented cubic
Fokker-Planck model. It is not independent DSMC or experimental
validation of the collision operator's physical fidelity.

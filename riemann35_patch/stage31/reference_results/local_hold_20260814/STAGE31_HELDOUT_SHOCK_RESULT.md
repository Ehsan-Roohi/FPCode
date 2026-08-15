# Stage 31 held-out Mach-2 normal-shock result

- Decision: **WORKSTATION_HOLD**
- Held-out case: Mach 2 (Stage 25A used Mach 3)
- Steps / final time: 48 / 0.374661 tau
- Causal births / front births / releases: 6 / 6 / 3
- Mean/peak/final kinetic fraction: 13.776% / 16.667% / 16.667%
- Full-profile adaptive errors rho/u/T: 0.4657% / 0.4223% / 0.2512%
- Full-profile adaptive errors stress/heat flux: 5.9689% / 10.0667%
- Full-profile adaptive errors M400/M420: 0.5193% / 2.1163%
- Space-time adaptive M400/M420 errors: 0.3483% / 1.1323%
- Adaptive/coarse-DVM wall-time ratio: 0.730x (1.370x speedup)
- Maximum balance / micro-macro sync residual: 8.370e-14 / 4.305e-12

This is a held-out numerical cross-case test of the implemented cubic
Fokker-Planck model. It is not independent DSMC or experimental
validation of the collision operator's physical fidelity.

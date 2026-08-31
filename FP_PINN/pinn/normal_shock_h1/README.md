# H1: spatial normal shock

Gate 0 constructs a positive Mott--Smith two-Maxwellian manifold connecting
exact monatomic Rankine--Hugoniot states. Every mixture point conserves mass,
momentum, and energy flux while retaining nonequilibrium stress and heat flux.

This is a physics/initialization gate, **not** a claimed FP solution. Gate 1
solves the steady nonlinear local-moment Dougherty--FP equation on a structured
axisymmetric velocity grid.  Its positive neural correction uses transformed
coordinates and vanishes at both exact Rankine--Hugoniot boundaries.  Gate 1 is
a discrete feasibility/residual gate; external DVM/DSMC validation is Gate 2.

```bash
bash FP_PINN/pinn/normal_shock_h1/RUN_H1_GATE0.sh

# submits one Mach-5 GPU qualification job on Unity
bash FP_PINN/pinn/normal_shock_h1/RUN_H1_GATE1_UNITY.sh

# H1R pilot: Simpson moments and a structural three-flux exponential tilt
bash FP_PINN/pinn/normal_shock_h1/RUN_H1R_PILOT_UNITY.sh
```

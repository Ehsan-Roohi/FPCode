# H1: spatial normal shock

Gate 0 constructs a positive Mott--Smith two-Maxwellian manifold connecting
exact monatomic Rankine--Hugoniot states. Every mixture point conserves mass,
momentum, and energy flux while retaining nonequilibrium stress and heat flux.

This is a physics/initialization gate, **not** a claimed cubic-FP solution. The
next gate adds the spatial cubic-FP residual and domain-decomposed PINN.

```bash
bash FP_PINN/pinn/normal_shock_h1/RUN_H1_GATE0.sh
```

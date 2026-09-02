# H2: reference-qualified kinetic PINN for a normal shock

H1R3 is retired as a scientific qualification path. Its nonlinear exponential
tilt imposed three fluxes to machine precision, so the flux gate was not an
independent test of the Fokker--Planck equation.

H2 follows the positive macro--micro and moment-observability construction in
Roohi, *Tail observability and fourth-order closure recovery in
physics-informed neural networks for BGK normal shocks* (2026):

1. Reproduce the stationary Mach-2 BGK case on the paper's upstream-mean-free-
   path coordinate with an independent conservative DVM reference.
2. Use `f = M exp(psi)` with a bounded exponent and explicit heat-flux and
   normal-stress Hermite modes. No post-network flux tilt is permitted.
3. Train with sparse macro locks and sparse joint `qx`/`sigma_xx` anchors.
   Dense reference profiles and distribution slices remain held out.
4. Qualify on disjoint held-out points using `rho`, `u`, `T`, `qx`, and
   `sigma_xx`, plus the kinetic residual and physical flux drift.
5. Replace BGK by the Dougherty--Fokker--Planck collision operator without
   changing the representation, quadrature, reference split, or gates.
6. Advance from Mach 2 to Mach 5 only after both BGK reproduction and FP Mach-2
   qualification pass.

The reference contract requires columns `x,rho,u,temperature,qx,sigma_xx` and
a JSON sidecar (for example `reference.csv.json`) containing at least
`{"solver":"conservative DVM","mach":2.0,"neural":false}`. Neural output is
explicitly rejected as a reference.

Gate 0:

```bash
FP_H2_REFERENCE=/absolute/path/to/reference.csv FP_H2_MACH=2 \
  bash FP_PINN/pinn/normal_shock_h1/RUN_H2_REFERENCE_GATE.sh
```

This gate audits data provenance, grid size, disjoint train/validation indices,
positivity, and equilibrium plateaus. It makes no PINN accuracy claim.

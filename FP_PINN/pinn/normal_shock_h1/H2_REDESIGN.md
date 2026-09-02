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

## Registered full-state reference

The paper's Mach-2 BGK full-state archive is accepted only at SHA-256
`8959d23bfe7643d0010bedd65516c6985103b50e3f15c1cc862893180c770a02`.
The loader never unpickles its legacy `states` member. It maps the numerical
arrays `x_mfp,rho,ux,T,qx,sig` and independently reintegrates `f(v)` with `v,w`.

Validation partitions are preregistered before PINN training. The full held-out
domain is reported, the shock-core metric is fixed at `|x/lambda1| <= 30`, and
the outer tails are reported separately. Both physical endpoints remain in the
audit. This prevents a boundary artifact from being hidden by an after-the-fact
crop while keeping it distinct from the shock-core accuracy claim.

Gate 0:

```bash
FP_H2_REFERENCE=/absolute/path/to/standing_M2_fullstate.npz FP_H2_MACH=2 \
  bash FP_PINN/pinn/normal_shock_h1/RUN_H2_REFERENCE_GATE.sh
```

This gate audits SHA-256 provenance, direct moment consistency, deterministic
disjoint train/validation indices, positivity, endpoint equilibrium, and the
outer-tail artifact. It makes no PINN accuracy claim.

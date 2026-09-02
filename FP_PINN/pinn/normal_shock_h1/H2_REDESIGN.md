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

## Gate 1: stationary BGK equation

`train_h2_bgk.py` is the first actual H2 solve. It maps the 80-mean-free-path
domain to the neural coordinate and fixes `Kn_eff=1/80`. The model uses four
128-wide SiLU layers and the positive bounded `M exp(psi)` representation with
the heat-flux and normal-stress modes stated in the paper. The collision term
is a local discrete Maxwellian built from the moments implied by the neural
distribution. No full-state distribution values and no dense profiles enter
the loss.

Training uses 32 macro locks, 16 joint heat-flux/stress anchors, the steady
velocity-weighted BGK residual, and the three collision-invariant fluxes. All
other 1552 interior stations are held out. The preregistered shock-core gates
are 2% for each macro field and 20% for each nonequilibrium moment; flux drift
is limited to 1%, the normalised residual to 0.2, and boundary error to 0.5%.
These thresholds are written in `h2_bgk.py` before the first run.

```bash
FP_H2_REFERENCE=/absolute/path/to/standing_M2_fullstate.npz FP_H2_MACH=2 \
  bash FP_PINN/pinn/normal_shock_h1/RUN_H2_BGK_UNITY.sh
```

The archive includes the blind DVM comparison for all five fields, the exact
train/validation indices, optimisation history, metrics, checksums, weights,
and a six-panel physics figure. Plot legends sit above their axes and outside
the data curves.

### H2R anti-aliasing correction

The first Gate-1 run exposed a decisive distinction between training-grid and
dense-grid conservation: flux RMS at the fixed collocation stations was below
1%, while dense held-out flux profiles varied by 37--55%. Sparse-field errors
showed the same pattern. This is collocation aliasing, not insufficient epoch
count, and the scientific gates are not relaxed.

H2R removes the finite-difference loophole. It evaluates `df/dx` by TensorFlow
forward-mode automatic differentiation, resamples 257 interior collocation
stations every epoch, and audits the final residual independently on 641 fixed
stations. Flux locks are applied to the resampled stations, so a narrow
between-node excursion cannot remain systematically invisible. Checkpoint
selection still uses training objectives only; all dense DVM profiles remain
held out until the final audit.

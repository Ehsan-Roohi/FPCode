# Riemann35.jl + cubic Fokker--Planck integration map

This note records the Stage-20 coupling target for
[`comp-physics/Riemann35.jl`](https://github.com/comp-physics/Riemann35.jl),
the current realizability-preserving, high-order, GPU/multi-GPU 35-moment code.
The primary path is an analytical cubic-FP collision model with adaptive
positive tail memory; a neural closure is not required.

## Confirmed interfaces

- The canonical raw-moment order in `Riemann35.MomentIndices.IJK` is identical
  to `hyqmom_fp.HYQMOM_35_INDICES`.
- The current CPU collision entry point is
  `src/numerics/collision35.jl::collision35`.
- High-order CPU stages use the device-compatible single-source helper
  `src/numerics/recon_dev.jl::bgk_relax_tup`.
- The retained state contains all raw moments through total degree four.
- `Moments5_3D(M)` supplies the 21 total-degree-five HyQMOM flux moments.
- `chyqmom_nodes_3d(M)` supplies a non-negative conditional velocity
  quadrature from which arbitrary higher raw moments can be evaluated.

## Closure required by cubic FP

For a retained moment `M_alpha`, the weak FP source contains

```text
sum_i alpha_i <v^(alpha-e_i) a_i(c)>
  + D sum_i alpha_i(alpha_i-1) M_(alpha-2e_i),
```

where the FPCode drift has a cubic term in the peculiar velocity.  Therefore:

- equations through degree two need moments through degree four;
- degree-three equations need degree-five moments; and
- degree-four equations need degree-six moments.

The existing M5 transport closure is necessary but not sufficient. The
original Riemann35-compatible baseline used this split tail:

1. total degree `<= 4`: the retained state exactly;
2. total degree `5`: `Moments5_3D`, preserving the solver's own HyQMOM closure;
3. total degree `6`: moments evaluated from `chyqmom_nodes_3d`.

The CHyQMOM node inversion intentionally truncates a documented subset of
high-order cross constraints.  It should therefore be used only to supply M6
in this first experiment, not to overwrite the retained M4 state or the
analytical M5 closure.  Sensitivity to the M6 construction must be reported.

Stages 11--17 establish that this and the other tested instantaneous closures
cannot be universally exact. Positive distributions with the same retained 35
moments admit materially different M6 tails and even opposite fourth-moment
source signs. The split CHyQMOM tail remains a useful inexpensive macro prior,
but it is not the selected high-error correction.

The finite-step validation exposed the same requirement in a stronger form:
directly mapping the quadrature overwrites six retained cross moments and
creates an order-one jump as `dt -> 0`. The residual-cancelled finite map uses
`M_new = M + (Q_mapped - Q_unmapped)`, preserving the exact retained state
while using CHyQMOM only for the collision increment.

The first residual-cancelled full step still left the realizability cone in
Stage 5. Stage 6 therefore retries each complete macro interval using
powers-of-two finite-map subcycling, always restarting a failed trial from the
last committed state. This cleanly distinguishes a timestep admissibility
problem from a structural closure error: success must include both cone
preservation and improvement over the particle-reference accuracy baseline.

Stage 20 supplies the replacement lifecycle. A source-disagreement/high-skew
sensor selects between the inexpensive Stage-9 macro map and a persistent
positive QMC microstate. Macro-to-micro conversion is allowed only when a
causal support is supplied from an initial condition, kinetic inflow, or
active neighbor. Positive entropy reweighting matches all 35 moments on that
support. Micro-to-macro conversion is the direct positive quadrature moment
map. Hysteresis controls release and prevents representation chatter.

## Implementation sequence

1. Port `fp_collision_source35(M, tau; Pr=2/3)` and the Stage-9 macro finite map
   beside `collision35`; keep source-invariant and realizability tests.
2. Add a collision selector such as `collision_model = :bgk | :esbgk |
   :cubic_fp_macro | :cubic_fp_adaptive`; leave the default path byte-identical.
3. Port the Stage-20 sensor, hysteresis state, and direct micro-to-macro
   projection. Match the Python six-trajectory audit before spatial work.
4. Add one-dimensional positive microstate transport. Define face fluxes and
   causal donor selection before permitting macro-to-micro birth in an
   interior cell; an algebraic reconstruction from the 35 moments is invalid.
5. Validate a shock/relaxation problem against an independently discretized
   positive DVM or spectral reference and report missed alarms, false alarms,
   active-cell fraction, and wall time.
6. Run crossing jets and compare with DSMC/SPARTA before proposing MFC coupling.

## Stage-25A implementation status

The first spatial contract is now implemented in
`hyqmom_fp/spatial_shock.py` and
`riemann35_patch/stage25a/run_normal_shock.py`. It includes:

- exact Mach-3 monatomic Rankine-Hugoniot end states;
- conservative positive 1D upwind transport of a three-dimensional DVM;
- a 35-moment positive kinetic-flux macro baseline;
- shared macro/micro face fluxes and causal neighbour/inflow birth rules;
- the frozen Stage-24B sensor thresholds without spatial retuning;
- full-DVM, macro, and adaptive histories with conservation, positivity,
  realizability, active-cell, transition, and timing diagnostics;
- scale-invariant marginal tolerances plus transactional progress,
  failure-checkpoint, and exact-configuration resume support; and
- a predeclared qualification protocol in
  `riemann35_patch/stage25a/STAGE25A_NORMAL_SHOCK_PROTOCOL.md`.

The retained local smoke run is development evidence only. It exercises the
new contracts on a coarse grid; the frozen physical/velocity/time/domain
refinement and like-for-like economy gates remain required before any spatial
accuracy claim.

## GPU boundary

`chyqmom_nodes_3d` currently allocates vectors/matrices and is documented as a
CPU-side addition, so it cannot simply be called from Riemann35's GPU collision
kernel.  The scientifically useful order is:

1. validate CPU macro--micro transport and the independent reference;
2. derive allocation-free sensor and positive microstate kernels;
3. then add the GPU adaptive implementation and CPU/GPU parity tests.

This avoids optimizing an unvalidated closure and keeps the current GPU BGK
path untouched.

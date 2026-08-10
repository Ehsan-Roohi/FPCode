# Riemann35.jl + cubic Fokker--Planck integration map

This note updates the Stage-0 coupling target from the older `HyQMOM.jl`
solver to [`comp-physics/Riemann35.jl`](https://github.com/comp-physics/Riemann35.jl),
the current realizability-preserving, high-order, GPU/multi-GPU 35-moment code.
The primary path is an analytical cubic-FP collision model; a neural closure is
not required for the first integration.

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

The existing M5 transport closure is necessary but not sufficient.  The first
Riemann35-compatible CPU prototype should use this split tail:

1. total degree `<= 4`: the retained state exactly;
2. total degree `5`: `Moments5_3D`, preserving the solver's own HyQMOM closure;
3. total degree `6`: moments evaluated from `chyqmom_nodes_3d`.

The CHyQMOM node inversion intentionally truncates a documented subset of
high-order cross constraints.  It should therefore be used only to supply M6
in this first experiment, not to overwrite the retained M4 state or the
analytical M5 closure.  Sensitivity to the M6 construction must be reported.

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

## Implementation sequence

1. Add a CPU function `fp_collision_source35(M, tau; Pr=2/3)` beside
   `collision35` and test mass, all three momenta, and total-energy source
   invariants to roundoff.
2. Add a collision selector such as `collision_model = :bgk | :esbgk |
   :cubic_fp`; leave the default path byte-identical.
3. Integrate the nonlinear source with a positivity/realizability-aware
   substep.  Do not reuse the exact BGK exponential because the cubic source is
   nonlinear in both the state and its closure.
4. Run homogeneous two-stream relaxation against the NumPy particle validator
   in this branch, then a single-cell Julia/particle comparison.
5. Run the existing crossing-jets case with BGK and cubic FP and monitor the
   moment-cone projection count as part of the result, not only density plots.
6. Compare the spatial result with DSMC/SPARTA before proposing MFC coupling.

## GPU boundary

`chyqmom_nodes_3d` currently allocates vectors/matrices and is documented as a
CPU-side addition, so it cannot simply be called from Riemann35's GPU collision
kernel.  The scientifically useful order is:

1. validate the CPU model and M6 closure;
2. derive an allocation-free scalar M6 closure or a device quadrature kernel;
3. then add the GPU single-source implementation and parity tests.

This avoids optimizing an unvalidated closure and keeps the current GPU BGK
path untouched.

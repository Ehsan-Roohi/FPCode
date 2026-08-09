# HyQMOM–Fokker–Planck coupling: Stage 0

This directory contains the first executable bridge between:

- the public [`comp-physics/HyQMOM.jl`](https://github.com/comp-physics/HyQMOM.jl)
  fourth-order, 35-moment kinetic solver; and
- FPCode's cubic Fokker–Planck collision model.

It is a homogeneous collision prototype, not yet a replacement for
`HyQMOM.jl/src/numerics/collision35.jl` and not yet an MFC integration.

## What is implemented

1. The exact 35-moment array ordering used by HyQMOM.jl.
2. Conversion of those raw moments to density, velocity, temperature, stress,
   and heat flux using FPCode conventions.
3. Projection of the continuous-time FP velocity-space generator onto every
   retained moment through fourth order.
4. The linear Ornstein–Uhlenbeck baseline and FPCode's `C`, `Gamma`, and
   `Beta` cubic drift corrections.
5. Exact collision-invariant projection for mass, momentum, and total energy.
6. A reproducible homogeneous relaxation comparison against a 35-moment BGK
   source.

The projected drift is

```text
a(c) = -c/tau + C c + Gamma (|c|^2 - 5 theta)
       + Beta (|c|^2 c - 2 q/rho),
```

with isotropic velocity diffusion `theta/tau`. The coefficient map follows
`FP_PINN/legacy_source/147CylFP.py` in nondimensional units (`mass = kB = 1`).
The finite-step particle limiter `m2_lim` and stochastic `alpha` correction are
not copied into the continuous moment generator; conservation is imposed
directly on the projected source.

## Important closure result

The experiment makes a key mathematical requirement explicit: closing the
HyQMOM free-transport flux at fourth order requires fifth-order moments, but
projecting the **cubic** FP collision drift onto the fourth-order equations
also requires **sixth-order** moments. Thus, the existing HyQMOM M5 flux closure
alone is not sufficient for the complete hybrid collision model.

For this first runnable milestone, retained moments through M4 are used exactly
and only the missing M5/M6 tail is reconstructed by a local Gaussian with the
same mean and full covariance. The closure is isolated behind
`GaussianTailClosure`; it can be replaced by a Grad-HyQMOM, quadrature,
maximum-entropy, neural, or tabulated closure without changing the projection
code.

## Run

From the repository root:

```bash
python -m unittest discover -s tests -p 'test_hyqmom_fp.py' -v

python examples/run_hyqmom_fp_relaxation.py \
  --steps 200 \
  --dt 2.5e-4 \
  --tau 1.0 \
  --output results/hyqmom_fp_relaxation.csv
```

Only NumPy is required for this Stage-0 prototype.

## Next integration milestones

1. Validate the projected source directly against FPCode particle ensembles
   in a spatially homogeneous relaxation problem.
2. Replace the Gaussian M5/M6 tail with a structure-preserving high-order
   reconstruction and quantify sensitivity to that choice.
3. Port the validated source to Julia behind a collision-model selector in
   `HyQMOM.jl`.
4. Reproduce the crossing-jet example with BGK and projected FP collisions.
5. Compare against DSMC/SPARTA over Mach and Knudsen sweeps before proposing
   integration into MFC.

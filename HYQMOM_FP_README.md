# HyQMOM–Fokker–Planck coupling: Stage 20

This directory contains the first executable bridge between:

- the public [`comp-physics/Riemann35.jl`](https://github.com/comp-physics/Riemann35.jl)
  realizability-preserving fourth-order, 35-moment kinetic solver; and
- FPCode's cubic Fokker–Planck collision model.

It is now a homogeneous adaptive macro--micro collision prototype, not yet a replacement for
`Riemann35.jl/src/numerics/collision35.jl` and not yet an MFC integration.
[`RIEMANN35_FP_INTEGRATION.md`](RIEMANN35_FP_INTEGRATION.md) maps the verified
CPU/GPU interfaces and the proposed non-neural integration sequence.

## What is implemented

1. The exact 35-moment array ordering used by Riemann35.jl.
2. Conversion of those raw moments to density, velocity, temperature, stress,
   and heat flux using FPCode conventions.
3. Projection of the continuous-time FP velocity-space generator onto every
   retained moment through fourth order.
4. The linear Ornstein–Uhlenbeck baseline and the physical 9-by-9 solve for
   FPCode's `C`, `Gamma`, and `Beta` cubic-drift corrections.
5. Exact collision-invariant projection for mass, momentum, and total energy.
6. A reproducible homogeneous relaxation comparison against a 35-moment BGK
   source.
7. A deterministic NumPy particle reference that mirrors FPCode's analytical
   coefficient map, peculiar-speed limiter, exact OU factor, stochastic
   update, and finite-step energy correction.
8. Family-wide Gaussian-mixture, Grad/GQMOM, particle, QMC, Hermite, and
   maximum-entropy closure/reference audits.
9. A constructive positive truncated-moment witness showing that the same 35
   moments can have different M5/M6 tails and opposite fourth-moment source
   signs.
10. A causal adaptive tail-memory lifecycle with a positive QMC microstate,
    entropy projection matching all 35 transported moments, a target-specific
    disagreement/high-skew sensor, and activation/release hysteresis.

The projected drift is

```text
a(c) = -c/tau + C c + Gamma (|c|^2 - 3 theta)
       + Beta (|c|^2 c - 2 q/rho),
```

with isotropic velocity diffusion `theta/tau`. The coefficient map follows
`FP_PINN/legacy_source/147CylFP.py` in nondimensional units (`mass = kB = 1`).
The finite-step particle limiter `m2_lim` and stochastic `alpha` correction are
not copied into the continuous moment generator; conservation is imposed
directly on the projected source.

## Closure result and selected method

The audits make a key mathematical requirement explicit: closing the
HyQMOM free-transport flux at fourth order requires fifth-order moments, but
projecting the **cubic** FP collision drift onto the fourth-order equations
also requires **sixth-order** moments. Thus, the existing HyQMOM M5 flux closure
alone is not sufficient for the complete hybrid collision model.

Stages 11--16 show that none of the tested instantaneous algebraic closures
meets the 3% rare-beam `M400` history gate. Stage 17 then constructs positive
distributions with identical moments through M4 but an 8.83% M600 range and a
self-consistent `dM400/dt` range from -127.5 to +24.2. Thus, an exact universal
35-to-M5/M6 map does not exist without an additional modeling assumption.

The selected method retains HyQMOM-35 as the macro model in well-identified
cells and carries a positive kinetic microstate only when the online sensor
indicates unresolved tail memory. A microstate must be causal: known at an
initial/inflow boundary or inherited from a transported kinetic neighbor. It
is never reconstructed after an alarm from the same 35 numbers. Entropy
reweighting matches all 35 moments on the causal support; failure to match the
target convex hull is explicit.

## Run

From the repository root:

```bash
python -m unittest discover -s tests -p 'test_hyqmom_fp.py' -v

python examples/run_hyqmom_fp_relaxation.py \
  --steps 200 \
  --dt 2.5e-4 \
  --tau 1.0 \
  --output results/hyqmom_fp_relaxation.csv

python examples/validate_hyqmom_fp_particles.py \
  --particles 100000 \
  --steps 200 \
  --dt 2.5e-4 \
  --tau 1.0 \
  --sample-every 10 \
  --output results/hyqmom_fp_particle_validation.csv

python riemann35_patch/stage20/run_hysteretic_tail_memory_audit.py \
  --stage11 results/riemann35_stage11 \
  --stage14 results/riemann35_stage14 \
  --output results/riemann35_stage20 \
  --workers 4
```

The Stage-20 audit requires NumPy, SciPy, and Matplotlib.

## Stage-20 result

On the six Stage-11 trajectories, only `rare_beam_ma20` activates the
microstate. Its `M400` history error falls from 16.59% for Stage 9 to 1.66%
against particles and 1.12% against the positive QMC reference; four-scramble
spread is 2.33%. The equal-case micro active fraction is 16.7%, no causal
activation is blocked, all histories remain realizable, and there is no mode
chatter. A separate safe causal-birth control executes exactly one
micro-to-macro release at `t/tau=0.2`.

These numbers pass the homogeneous 3% gate. They do not replace an independent
positive DVM/spectral reference or a spatial validation.

## Remaining integration milestones

1. Build an independently discretized positive DVM/spectral reference that
   does not share the particle/QMC finite collision map.
2. Add one-dimensional shock/relaxation transport of the positive microstate,
   including face fluxes and causal neighbor/inflow donor selection.
3. Port the validated adaptive CPU source to Julia behind a collision-model selector in
   `Riemann35.jl`, leaving the default BGK/ES-BGK path unchanged.
4. Reproduce the crossing-jet/shock examples with BGK, algebraic FP, and
   adaptive tail-memory FP collisions, reporting active-cell fraction and cost.
5. Compare against DSMC/SPARTA or positive DVM data over Mach and Knudsen sweeps before proposing
   integration into MFC.

# Stage 10: general realizability and Grad-HyQMOM audit

This stage implements the Appendix-C Grad-HyQMOM approximation from
Bryngelson, Fox & Laurent (JCP 566, 2026, 115242), with Gaussian-GQMOM
univariate moments and the existing HyQMOM fifth-order closure.  It compares
that source closure with the corrected Stage-9 tensor mixture and a single
Gaussian tail on a deterministic ensemble of realizable 3-D Gaussian-mixture
states.

The ensemble contains the five Stage-9 regression states, anisotropic rare-hot
populations, near-delta counter-streams/crossing jets through Ma=100, rare
beams, and seeded random two- to four-component multivariate mixtures.  The
audit reports M5/M6 error, cubic-FP source-vector error, source direction,
one-step and selected long-time realizability, signed Grad mass, branch seam
jumps, and prototype cost.

The production candidate uses an exact Ornstein--Uhlenbeck substep followed by
the Grad--HyQMOM nonlinear cubic correction.  A scalar line search retains the
largest correction that keeps the degree-two moment matrix realizable.  The
reported limiter fraction is one when this guard is inactive.

Run from the repository root:

```bash
python riemann35_patch/stage10/run_general_realizability_audit.py \
  --random-states 256 --workers 4 --dt 0.0025 --long-time 1.0 \
  --output results/riemann35_stage10
```

The Stage-9 map and Grad-HyQMOM are not assumed equivalent.  Stage 10 first
separates high-order closure accuracy from integration robustness; it does not
claim a proof of realizability for arbitrary moment vectors.

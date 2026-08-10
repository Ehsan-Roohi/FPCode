# Stage 11: dual-map particle validation through one collision time

Stage 11 compares the Stage-9 finite Gaussian-mixture map and the Stage-10
exact-OU/guarded Grad--HyQMOM/Gaussian--GQMOM map against sixteen independent
particle seeds on six homogeneous non-equilibrium trajectories through
`t/tau = 1`.

The primary reference uses the continuous cubic Fokker--Planck drift without
the legacy particle-speed clip.  Stage 9 is run with an infinite speed cap for
the same reason.  Each seed history is differenced from its own random initial
sample and shifted to the exact analytic initial moments before ensemble
averaging.  The reported uncertainty is therefore the standard error across
independent *trajectory changes*, not 41 time samples treated as independent.

Run the full validation with

```bash
python riemann35_patch/stage11/run_particle_validation.py \
  --particles 100000 --seeds 16 --workers 8 \
  --dt 0.0025 --final-time 1.0 --sample-every 10 \
  --output results/riemann35_stage11
```

The JSON records the exact seed list, numerical model, case-wise errors,
closure diagnostics, and limiter statistics.  Per-step CSV files contain the
complete Grad limiter history and Stage-9 realizability/reconstruction history.
The compressed NPZ file preserves both raw and paired-change particle moment
histories for all seeds.

No aggregate statement that one closure is more accurate is made.  Stage 9 is
expected to be exact or nearly exact for some axis-separable initial families;
Grad/GQMOM is evaluated as the continuous and robust baseline.

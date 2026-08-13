# Heat-flux v2 training

This variant improves the third-moment history without using particle data in
the training loss.  It adds a weak Fokker--Planck residual projected onto
`v |v|^2`, a broad exactly reweighted tail proposal, optional fixed velocity
quadrature, float64 high-order moments/closure, and differentiable propagation
through the 9x9 closure solve.

## Unity pilot

Run from the FPCode repository root.  The command updates the heat-flux branch
and submits only array task 2.  It fine-tunes the 30,000-epoch heat-flux weights
from Stage-2 job 62716918 with a smaller restarted Adam learning rate.

```bash
FP_TEST=/project/pi_roohie_umass_edu/github_sync/FPCode-pinn-test; cd "$FP_TEST" && git fetch origin && (git switch fp-pinn-heatflux-v2 2>/dev/null || git switch -c fp-pinn-heatflux-v2 --track origin/agent/fp-pinn-heatflux-v2) && git pull --ff-only origin agent/fp-pinn-heatflux-v2 && OLD_WEIGHTS="$FP_TEST/FP_PINN/pinn/cubic_homogeneous_3d/outputs/stage2-62716918/heat_flux/stage2_final.weights.h5" && test -f "$OLD_WEIGHTS" && FP_STAGE2_EPOCHS=5000 FP_STAGE2_LEARNING_RATE=1.0e-4 FP_STAGE2_LR_DECAY_STEPS=2500 FP_N_TIME_BATCH=12 FP_N_VELOCITY_PER_TIME=4096 FP_HEAT_FLUX_WEIGHT=10 FP_TAIL_FRACTION=0.20 FP_TAIL_VARIANCE=4.0 FP_FIXED_VELOCITY_QUADRATURE=1 FP_REFERENCE_PARTICLES=500000 FP_REFERENCE_DT=0.0025 FP_REFERENCE_SAVE_EVERY=20 FP_EVALUATION_SAMPLES=131072 FP_RESUME_WEIGHTS="$OLD_WEIGHTS" sbatch --array=2 FP_PINN/pinn/cubic_homogeneous_3d/slurm/run_stage2_array.sbatch
```

The log must include finite `pde=` and `qweak=` values.  The publication target
is `heat_flux_history_relative_l2 < 0.05`; the old `0.20` gate is only a smoke
threshold.  If differentiating the closure is unstable on a particular GPU,
repeat the pilot with `FP_STOP_GRADIENT_CLOSURE=1` as a diagnostic fallback.

## Symmetry-aware continuation after job 62845211

Job `62845211_2` completed all 5,000 epochs and removed the broad-tail Hessian
failure.  Its active heat-flux component had relative L2 error about 16.4%, but
the unrestricted network developed a nonphysical transverse heat flux.  This
continuation therefore preserves the legacy H5 layer shapes while restricting
the heat-flux correction to the exact axisymmetric feature set
`(t,cx,cy^2+cz^2)`.  Training and validation quadrature also use antithetic
four-point transverse sign orbits.  Each sampled `(cx,cy,cz)` is accompanied
by all sign reflections in `cy` and `cz`, cancelling transverse odd moments
and the `Pyz` cross moment at the quadrature level.

Validation now reports both:

- `heat_flux_active_axis_relative_l2` for the physical `Qx` relaxation; and
- `max_transverse_heat_flux_relative` as an independent symmetry gate.

The full-vector `heat_flux_history_relative_l2` remains in `metrics.json` for
backward comparison, but particle noise in nominally zero `Qy,Qz` no longer
defines the active relaxation gate.

Run the continuation from the repository root:

```bash
FP_TEST=/project/pi_roohie_umass_edu/github_sync/FPCode-pinn-test; cd "$FP_TEST" && git fetch origin && (git switch fp-pinn-heatflux-v2 2>/dev/null || git switch -c fp-pinn-heatflux-v2 --track origin/agent/fp-pinn-heatflux-v2) && git pull --ff-only origin agent/fp-pinn-heatflux-v2 && mkdir -p slurm_logs && OLD_WEIGHTS="$FP_TEST/FP_PINN/pinn/cubic_homogeneous_3d/outputs/stage2-62845211/heat_flux/stage2_final.weights.h5" && test -f "$OLD_WEIGHTS" && FP_STAGE2_EPOCHS=7500 FP_STAGE2_LEARNING_RATE=3.0e-5 FP_STAGE2_LR_DECAY_STEPS=2500 FP_N_TIME_BATCH=12 FP_N_VELOCITY_PER_TIME=4096 FP_HEAT_FLUX_WEIGHT=25 FP_TAIL_FRACTION=0.20 FP_TAIL_VARIANCE=4.0 FP_FIXED_VELOCITY_QUADRATURE=1 FP_REFERENCE_PARTICLES=500000 FP_REFERENCE_DT=0.0025 FP_REFERENCE_SAVE_EVERY=20 FP_EVALUATION_SAMPLES=131072 FP_RESUME_WEIGHTS="$OLD_WEIGHTS" sbatch --array=2 --nodes=1 --ntasks=1 --gres=gpu:1 --exclude=gypsum-gpu001,gypsum-gpu011,gypsum-gpu012,gypsum-gpu013,gypsum-gpu015 --output="$FP_TEST/slurm_logs/fp-pinn-stage2-%A_%a.out" --error="$FP_TEST/slurm_logs/fp-pinn-stage2-%A_%a.err" FP_PINN/pinn/cubic_homogeneous_3d/slurm/run_stage2_array.sbatch
```

Every completed task automatically creates one self-contained archive in the
repository root, for example:

```text
FP_PINN_STAGE2_JOB62899999_HEAT_FLUX_COMPLETE.zip
```

The archive contains all case outputs, the particle reference, portable
weights, CSV histories, validation figures, metrics, Slurm logs when available,
and `run_metadata.json` with the exact Git commit and job identifiers.

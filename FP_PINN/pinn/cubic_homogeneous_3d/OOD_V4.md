# Stage-2 V4: out-of-baseline robustness suite

V4 tests whether the self-consistent homogeneous cubic Fokker–Planck PINN
continues to work away from the single baseline heat-flux initial condition.
It is an algorithmic robustness suite: every case is trained independently
from its exact positive initial condition.  It is not presented as zero-shot
parameter generalization.

## Nine controlled cases

| Task | Case | Change from the baseline |
|---:|---|---|
| 0 | `heat_flux` | legacy `Qx(0)=0.25`, `nu=1` control |
| 1 | `ood_hf_q0125` | half-amplitude heat flux |
| 2 | `ood_hf_q0400` | stronger heat flux |
| 3 | `ood_hf_shape_w020` | same moments, different positive-mixture shape |
| 4 | `ood_hf_nu050` | slower collision rate |
| 5 | `ood_hf_nu200` | faster collision rate |
| 6 | `ood_stress_mild` | mild anisotropic stress |
| 7 | `ood_stress_strong` | strong anisotropic stress |
| 8 | `ood_coupled_axisym` | simultaneous stress and heat flux |

All initial states have unit mass, zero momentum, and trace-three covariance.
The heat-flux states use a strictly positive two-Gaussian mixture whose mean,
variance, and third moment are imposed analytically.

Each task creates an independent particle reference, trains the strong-PDE
PINN without particle data, validates against that reference, and performs a
fresh residual audit.  Heat-flux cases additionally receive the deterministic
exact-operator moment projection from V3.

## One-command Unity submission

Run from the repository root:

```bash
bash FP_PINN/pinn/cubic_homogeneous_3d/slurm/submit_stage2_ood_v4.sh
```

The wrapper submits the nine-task GPU array with at most three simultaneous
tasks and an `afterany` CPU collector.  The collector runs even if a case
fails, records incomplete cases, makes the aggregate CSV/JSON/PNG/PDF, and
atomically writes this single archive directly in the repository root:

```text
FP_PINN_STAGE2_V4_OOD_JOB<ARRAY_JOB_ID>_COMPLETE.zip
```

The most useful results are at the ZIP top level.  Per-case outputs, source
snapshot, Slurm logs, checksums, portable weights, and particle references are
also included.

Production defaults are 500,000 reference particles for heat-flux cases,
250,000 for stress-only cases, 15,000 PINN epochs, four rotating velocity
panels, order-48 deterministic evaluation, and 500 heat-flux projection
steps.  For a plumbing-only smoke test, override them at submission time, for
example:

```bash
FP_V4_PARTICLES=50000 FP_V4_EPOCHS=100 FP_V4_PROJECTION_STEPS=10 \
  bash FP_PINN/pinn/cubic_homogeneous_3d/slurm/submit_stage2_ood_v4.sh
```

Do not interpret a `COMPLETED` Slurm state alone as scientific success.  Use
the aggregate gate table and inspect the validation curves before advancing
to the spatially inhomogeneous problem.

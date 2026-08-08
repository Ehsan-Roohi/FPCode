# Stage 2 — homogeneous 3-D cubic Fokker–Planck PINN

This directory is the next controlled step after the one-dimensional
Ornstein–Uhlenbeck benchmark.  It solves the spatially homogeneous velocity
problem

\[
\partial_t f=-\nabla_{\boldsymbol c}\!\cdot(\boldsymbol a f)
              +\nu\theta\,\Delta_{\boldsymbol c}f,
\]

with

\[
a_i=-\nu c_i+C_{ij}c_j+\Gamma_i(c^2-\langle c^2\rangle)
    +\lambda(c_i c^2-Q_i),\qquad
Q_i=\langle c_i c^2\rangle=2q_i/\rho .
\]

The six independent entries of the symmetric matrix `C` and the three entries
of `Gamma` are recomputed from the canonical 9×9 moment system.  The analytic
coefficient

\[
\lambda=-\nu\,\|\Pi^{\mathrm{dev}}\|^2/\langle c^2\rangle^{7/2}
\]

is explicitly included in the cubic drift.  This is important: calculating
`lambda` for the right-hand side while omitting it from the actual velocity
drift does not implement the same cubic Fokker–Planck model.

## The controlled triad

All cases have unit mass, zero momentum, and `theta=1` (`<c²>=3`).  They differ
only in the non-equilibrium mode.

| Array task | Case | Initial state | Primary observable |
|---:|---|---|---|
| 0 | `equilibrium` | Maxwellian | invariance of Maxwellian |
| 1 | `stress` | Gaussian covariance `(1.6, 0.9, 0.5)` | deviatoric-stress relaxation |
| 2 | `heat_flux` | positive skew Gaussian mixture | `Qx` / heat-flux relaxation |

The heat-flux initial condition has the exact moments `<cx>=0`, `<cx²>=1`, and
`<cx³>=Qx=0.25`; it is a positive distribution, not a signed Grad correction.

## What is physics-informed here?

`train_stage2.py` represents `log(f)` and uses an ansatz that enforces the
initial condition exactly and positivity by construction.  At every optimizer
step it:

1. importance-samples velocity space on the whole `R³` domain;
2. obtains moments of the current neural density;
3. solves the self-consistent 9×9 closure and computes `lambda`;
4. differentiates the four-input network `(t,cx,cy,cz)` to form the FP PDE
   residual; and
5. penalizes mass, momentum, and energy drift.

Particle histories are never used in the training loss.  The independent
particle solver is used only after training for stress/heat-flux histories,
one-dimensional marginals, and numerical gates.

## Files

- `cubic_operator.py`: canonical moments, 9×9 closure, cubic drift, and three
  positive initial distributions.
- `reference_particle.py`: OU-exact/cubic-explicit particle reference with
  exact discrete projection of momentum and energy.
- `train_stage2.py`: self-consistent TensorFlow PINN and validation/figure
  generation.
- `slurm/run_stage2_array.sbatch`: Unity `gpu` partition array for all three
  cases.
- `summarize_stage2.py`: one-line numerical summary of the triad.
- `tests/`: analytic, indexing, closure, restart, and TensorFlow smoke tests.

## Unity: update and run

From the existing worktree, update the Stage-2 branch (replace the branch name
only if the repository reports a different one):

```bash
FP_TEST="/project/pi_roohie_umass_edu/github_sync/FPCode-pinn-test"
cd "$FP_TEST"
git fetch origin
git switch fp-pinn-cubic-stage2 2>/dev/null || \
  git switch -c fp-pinn-cubic-stage2 --track origin/codex/fp-pinn-cubic-stage2
git pull --ff-only origin codex/fp-pinn-cubic-stage2
```

Submit a short end-to-end audit first.  It runs all three array tasks, including
the independent reference, TensorFlow derivatives, H5 reload, validation, and
plots:

```bash
cd "$FP_TEST"
FP_STAGE2_EPOCHS=500 \
FP_REFERENCE_PARTICLES=50000 \
FP_EVALUATION_SAMPLES=16384 \
sbatch FP_PINN/pinn/cubic_homogeneous_3d/slurm/run_stage2_array.sbatch
```

After the short audit produces finite metrics and all plumbing is correct,
submit the paper-scale run:

```bash
cd "$FP_TEST"
FP_STAGE2_EPOCHS=30000 \
FP_REFERENCE_PARTICLES=250000 \
FP_EVALUATION_SAMPLES=65536 \
sbatch FP_PINN/pinn/cubic_homogeneous_3d/slurm/run_stage2_array.sbatch
```

The script uses the ordinary `gpu` partition, not `gpu-preempt`, and activates
the known Unity environment directly:

```text
/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu
```

It also adds the pip-installed NVIDIA library directories to
`LD_LIBRARY_PATH` and aborts before training if TensorFlow cannot see a GPU.

## Monitor

The submission prints an array job ID, for example `62700000`.  Use:

```bash
squeue -j 62700000

tail -f fp-pinn-stage2-62700000_0.out   # equilibrium
tail -f fp-pinn-stage2-62700000_1.out   # stress
tail -f fp-pinn-stage2-62700000_2.out   # heat flux
```

Finished states and memory use:

```bash
sacct -j 62700000 \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList
```

## Inspect and package results

If the array ID is `62700000`, the common result root is:

```bash
RUN_ROOT="$FP_TEST/FP_PINN/pinn/cubic_homogeneous_3d/outputs/stage2-62700000"

python "$FP_TEST/FP_PINN/pinn/cubic_homogeneous_3d/summarize_stage2.py" \
  "$RUN_ROOT"

find "$RUN_ROOT" -maxdepth 2 -type f \
  \( -name 'metrics.json' -o -name 'stage2_validation.pdf' \
     -o -name 'stage2_validation.png' -o -name 'moments_by_time.csv' \
     -o -name 'stage2_final.weights.h5' \) -print
```

Create a small upload archive without copying the large particle `.npz` files:

```bash
cd "$FP_TEST/FP_PINN/pinn/cubic_homogeneous_3d"
zip -r "FP_PINN_STAGE2_RESULTS_62700000.zip" "outputs/stage2-62700000" \
  -i '*/metrics.json' '*/config.json' '*/loss_history.csv' \
     '*/moments_by_time.csv' '*/stage2_validation.pdf' \
     '*/stage2_validation.png' '*/stage2_final.weights.h5' \
     '*/reference_metrics.json' '*/reference_history.csv'
```

## Gates and interpretation

Each `metrics.json` contains the raw quantities and Boolean checks.  The first
run is diagnostic; by default a numerical gate failure is recorded but does
not make Slurm label the job as a software failure.  Set `FP_STRICT_GATE=1`
only after tuning if a failed physics gate should return exit status 2.

The principal gates are:

- marginal-distribution relative L2 error;
- maximum mass, momentum, and energy error;
- stress-history error for `stress`;
- heat-flux-history error for `heat_flux`;
- equilibrium invariance for `equilibrium`;
- exact initial condition, nonnegative density, and exact portable H5 reload.

Do not advance to spatial Couette flow merely because Slurm says `COMPLETED`.
Advance when all three `metrics.json` files are finite, the curves are
physically monotone/credible, and the recorded gates pass or any threshold
change has a documented scientific justification.


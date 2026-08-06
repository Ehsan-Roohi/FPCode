# Stage 0: positive OU Fokker–Planck PINN

This is the first, deliberately small verification problem for the FP-PINN
work. It solves

```text
f_t = (v f)_v + f_vv
```

for a symmetric bimodal initial distribution on `v in [-6,6]`. The associated
SDE is `dV = -V dt + sqrt(2) dW`. An analytic solution is available, so a bad
sign, a missing factor of two, or an incorrect second derivative is exposed
before the project moves to the cubic 3-D velocity operator.

The neural density is positive by construction and satisfies the initial
condition exactly. The loss contains the strong FP residual, zero-flux velocity
boundary conditions, unit mass, and the analytic second-moment evolution.

## CPU reference test

From the repository root:

```bash
cd FP_PINN/pinn/ou_1d
python -m unittest discover -s tests -v
```

This test uses only NumPy and does not train a network.

## First Unity GPU run

Check whether the existing `dsmc-gpu` environment contains TensorFlow:

```bash
module load conda/latest
conda activate dsmc-gpu
python -c 'import tensorflow as tf; print(tf.__version__)'
```

If it is available, submit from the `FPCode` repository root:

```bash
sbatch FP_PINN/pinn/ou_1d/slurm/run_quick.sbatch
```

If a separate environment is preferred, create it once and pass its absolute
path when submitting:

```bash
module load conda/latest
conda env create \
  --prefix /project/pi_roohie_umass_edu/envs/fp-pinn \
  --file FP_PINN/pinn/ou_1d/environment.yml

FP_CONDA_ENV=/project/pi_roohie_umass_edu/envs/fp-pinn \
  sbatch FP_PINN/pinn/ou_1d/slurm/run_quick.sbatch
```

Monitor the run with:

```bash
squeue -u "$USER"
tail -F fp-pinn-ou-JOBID.out
```

Results are written under `FP_PINN/pinn/ou_1d/outputs/quick-JOBID/`. The most
important files are `metrics.json`, `ou_pinn_validation.png`,
`solution_tfinal.csv`, and `loss_history.csv`.

For a longer run after the quick job is healthy:

```bash
sbatch FP_PINN/pinn/ou_1d/slurm/run_quick.sbatch \
  --epochs 5000 --n-interior 8192 --n-velocity-quad 513
```

## Interpretation

This is a sign/factor/autodiff/positivity test, not yet the final cubic-FP
solver. The next stage will retain the positive distribution representation,
replace the OU drift by the cubic drift `A_i(C, Gamma, q, rho, epsilon)`, and
couple a differentiable `9 x 9` moment closure to the residual. That step needs
the curated legacy Couette source to be present on GitHub so its exact moment
ordering and coefficient conventions can be reused without guessing.



## Stage-1 convergence gate

The quick job is only an installation and sign-convention smoke test. A
scientifically useful next step is the Stage-1 gate:

- the analytic density is not used in the training loss;
- the hard ansatz imposes positivity and the initial density exactly;
- the loss combines the strong FP residual with a support-weighted
  log-density residual;
- mass conservation and the OU first- and second-moment ODEs are enforced;
- an exponentially decaying learning rate and gradient clipping stabilize the
  30,000-epoch optimization;
- the exact solution is used only after training for independent validation.

Submit it from the repository root:

```bash
sbatch FP_PINN/pinn/ou_1d/slurm/run_stage1.sbatch
```

The default job uses one RTX 2080 Ti, 30,000 epochs, 8,192 interior points per
epoch, 513 velocity quadrature points, and a (96\times5) tanh network. Results
are written under `FP_PINN/pinn/ou_1d/outputs/stage1-JOBID/`.

The terminal log prints either `STAGE1_GATE PASS` or `STAGE1_GATE FAIL`.
Passing requires global and final-time relative L2 errors below 5%, maximum
mass and first-moment errors below 1%, maximum second-moment error below 3%,
an exact initial condition to (10^{-6}), and nonnegative density. The gate
thresholds and every individual check are also stored in `metrics.json`.
Failure is retained as a completed computational job so all diagnostics remain
available; it means the method must be improved before moving to the cubic
3-D operator.

Useful files are `stage1_validation.png`, `metrics.json`,
`metrics_by_time.csv`, `loss_history.csv`, `solution_grid.npz`, and the
TensorFlow checkpoint.

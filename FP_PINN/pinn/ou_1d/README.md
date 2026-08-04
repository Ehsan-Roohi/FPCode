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


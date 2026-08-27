#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
METHODS=(qmc_base qmc_refined exact_coeff_gaussian_projection gaussian_coeff_exact_projection gaussian_both compact_positive)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
cd "$FP_STAGE55_RUN"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$FP_STAGE55_PYTHON" riemann35_patch/stage55_closure_source/run_source_method.py --method "$METHOD" --output "$FP_STAGE55_RESULT_ROOT"

#!/usr/bin/env bash
#SBATCH --job-name=fp-g0-gh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="${FP_REPO_ROOT:-$SUBMIT_DIR}"
STAGE2_DIR="${FP_STAGE2_DIR:-$REPO_ROOT/FP_PINN/pinn/cubic_homogeneous_3d}"

ORDERS=(12 16 20 24)
ORDER="${ORDERS[$SLURM_ARRAY_TASK_ID]}"

export FP_CASE_NAME="heat_flux"
export FP_OUTPUT_ROOT="outputs/g0-gh-audit-${SLURM_ARRAY_JOB_ID}/order-${ORDER}"
export FP_PACKAGE_TAG="HEAT_FLUX_GH${ORDER}_AUDIT"
export FP_SEED="${FP_SEED:-20260810}"
export FP_EVALUATE_ONLY="1"
export FP_CHECKPOINT_SWEEP="0"
export FP_STRICT_GATE="0"
export FP_EVALUATION_GAUSS_HERMITE_ORDER="$ORDER"
export FP_REFERENCE_PARTICLES="${FP_REFERENCE_PARTICLES:-500000}"
export FP_REFERENCE_DT="${FP_REFERENCE_DT:-0.0025}"
export FP_REFERENCE_SAVE_EVERY="${FP_REFERENCE_SAVE_EVERY:-20}"
export FP_MARGINAL_QUADRATURE_ORDER="${FP_MARGINAL_QUADRATURE_ORDER:-18}"
export FP_AXISYMMETRIC_HEAT_FLUX="1"
export FP_ANTITHETIC_HEAT_FLUX_QUADRATURE="1"
export FP_PACKAGE_RESULTS="1"

echo "Deterministic Gauss-Hermite audit order: $ORDER"
echo "Resume archive: ${FP_RESUME_ARCHIVE:-<not set>}"
echo "No optimizer steps will be taken."

exec bash "$STAGE2_DIR/slurm/run_stage2_array.sbatch"

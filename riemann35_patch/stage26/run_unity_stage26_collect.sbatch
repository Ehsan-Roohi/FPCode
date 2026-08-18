#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

: "${FP_STAGE26_RUN:?FP_STAGE26_RUN is required}"
: "${FP_STAGE26_RESULT_ROOT:?FP_STAGE26_RESULT_ROOT is required}"
: "${FP_STAGE26_BUNDLE:?FP_STAGE26_BUNDLE is required}"

PYTHON_BIN="${FP_STAGE26_PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-stage26-${SLURM_JOB_ID:-local}"

cd "$FP_STAGE26_RUN"
"$PYTHON_BIN" riemann35_patch/stage26/collect_four_delta.py \
    --root "$FP_STAGE26_RESULT_ROOT" \
    --bundle "$FP_STAGE26_BUNDLE"
echo "[stage26] collector completed=$(date -Is) bundle=$FP_STAGE26_BUNDLE"

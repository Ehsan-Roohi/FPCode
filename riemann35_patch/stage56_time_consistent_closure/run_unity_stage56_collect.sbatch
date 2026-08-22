#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

: "${FP_STAGE56_RUN:?FP_STAGE56_RUN is required}"
: "${FP_STAGE56_RESULT_ROOT:?FP_STAGE56_RESULT_ROOT is required}"
: "${FP_STAGE56_REFERENCE_ROOT:?FP_STAGE56_REFERENCE_ROOT is required}"
: "${FP_STAGE56_BUNDLE:?FP_STAGE56_BUNDLE is required}"

PYTHON_BIN="${FP_STAGE56_PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-stage56-${SLURM_JOB_ID:-local}"

cd "$FP_STAGE56_RUN"
"$PYTHON_BIN" -m riemann35_patch.stage56_time_consistent_closure.collect_time_consistent \
    --root "$FP_STAGE56_RESULT_ROOT" \
    --reference-root "$FP_STAGE56_REFERENCE_ROOT" \
    --bundle "$FP_STAGE56_BUNDLE"
echo "[stage56] collector completed=$(date -Is) bundle=$FP_STAGE56_BUNDLE"

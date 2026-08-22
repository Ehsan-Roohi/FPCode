#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

: "${FP_STAGE55_RUN:?FP_STAGE55_RUN is required}"
: "${FP_STAGE55_RESULT_ROOT:?FP_STAGE55_RESULT_ROOT is required}"
: "${FP_STAGE55_BUNDLE:?FP_STAGE55_BUNDLE is required}"

PYTHON_BIN="${FP_STAGE55_PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-stage55-${SLURM_JOB_ID:-local}"

cd "$FP_STAGE55_RUN"
"$PYTHON_BIN" -m riemann35_patch.stage55_closure_source_audit.collect_closure_audit \
    --root "$FP_STAGE55_RESULT_ROOT" \
    --bundle "$FP_STAGE55_BUNDLE"
echo "[stage55] collector completed=$(date -Is) bundle=$FP_STAGE55_BUNDLE"

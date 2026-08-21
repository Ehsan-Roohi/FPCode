#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

: "${FP_STAGE54_RUN:?FP_STAGE54_RUN is required}"
: "${FP_STAGE54_RESULT_ROOT:?FP_STAGE54_RESULT_ROOT is required}"
: "${FP_STAGE54_BUNDLE:?FP_STAGE54_BUNDLE is required}"

PYTHON_BIN="${FP_STAGE54_PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-stage54-${SLURM_JOB_ID:-local}"

cd "$FP_STAGE54_RUN"
"$PYTHON_BIN" -m riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux \
    --root "$FP_STAGE54_RESULT_ROOT" \
    --bundle "$FP_STAGE54_BUNDLE"
echo "[stage54] collector completed=$(date -Is) bundle=$FP_STAGE54_BUNDLE"

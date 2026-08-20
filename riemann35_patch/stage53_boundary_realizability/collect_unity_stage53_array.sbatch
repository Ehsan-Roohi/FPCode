#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

: "${FP_STAGE53_RUN:?FP_STAGE53_RUN is required}"
: "${FP_STAGE53_RESULT_ROOT:?FP_STAGE53_RESULT_ROOT is required}"
: "${FP_STAGE53_BUNDLE:?FP_STAGE53_BUNDLE is required}"

PYTHON_BIN="${FP_STAGE53_PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-stage53-collector-${SLURM_JOB_ID:-local}"

cd "$FP_STAGE53_RUN"
"$PYTHON_BIN" -m riemann35_patch.stage53_boundary_realizability.collect_stage53_array \
    --mode unity \
    --result-root "$FP_STAGE53_RESULT_ROOT"
cd "$FP_STAGE53_RESULT_ROOT"
zip -q -r "$FP_STAGE53_BUNDLE" .
echo "[stage53-collect] completed=$(date -Is) bundle=$FP_STAGE53_BUNDLE"

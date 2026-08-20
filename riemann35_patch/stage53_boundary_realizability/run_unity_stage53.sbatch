#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

: "${FP_STAGE53_RUN:?FP_STAGE53_RUN is required}"
: "${FP_STAGE53_RESULT_ROOT:?FP_STAGE53_RESULT_ROOT is required}"
: "${FP_STAGE53_BUNDLE:?FP_STAGE53_BUNDLE is required}"

PYTHON_BIN="${FP_STAGE53_PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-stage53-${SLURM_JOB_ID:-local}"

cd "$FP_STAGE53_RUN"
echo "[stage53] host=$(hostname) commit=$(git rev-parse HEAD) started=$(date -Is)"
"$PYTHON_BIN" -c 'import matplotlib, numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__, "matplotlib", matplotlib.__version__)'
"$PYTHON_BIN" -m riemann35_patch.stage53_boundary_realizability.run_boundary_realizability \
    --mode unity \
    --output "$FP_STAGE53_RESULT_ROOT"

cd "$FP_STAGE53_RESULT_ROOT"
zip -q -r "$FP_STAGE53_BUNDLE" .
echo "[stage53] completed=$(date -Is) bundle=$FP_STAGE53_BUNDLE"

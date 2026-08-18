#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail

: "${FP_STAGE28_RUN:?FP_STAGE28_RUN is required}"
: "${FP_STAGE28_RESULT_ROOT:?FP_STAGE28_RESULT_ROOT is required}"
: "${FP_STAGE28_BUNDLE:?FP_STAGE28_BUNDLE is required}"

PYTHON_BIN="${FP_STAGE28_PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-stage28-${SLURM_JOB_ID:-local}"

cd "$FP_STAGE28_RUN"
echo "[stage28] host=$(hostname) commit=$(git rev-parse HEAD) started=$(date -Is)"
"$PYTHON_BIN" -c 'import matplotlib, numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__, "matplotlib", matplotlib.__version__)'
"$PYTHON_BIN" -m riemann35_patch.stage28.run_localized_pocket \
    --mode unity \
    --output "$FP_STAGE28_RESULT_ROOT"

cd "$FP_STAGE28_RESULT_ROOT"
zip -q -r "$FP_STAGE28_BUNDLE" .
echo "[stage28] completed=$(date -Is) bundle=$FP_STAGE28_BUNDLE"

#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

: "${FP_STAGE53_RUN:?FP_STAGE53_RUN is required}"
: "${FP_STAGE53_RESULT_ROOT:?FP_STAGE53_RESULT_ROOT is required}"
: "${SLURM_ARRAY_TASK_ID:?Stage 53 must be submitted as an array}"

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 4 )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID must be in 0..4" >&2
    exit 2
fi

PYTHON_BIN="${FP_STAGE53_PYTHON:-python}"
TASK_ROOT="$FP_STAGE53_RESULT_ROOT/epsilon_$SLURM_ARRAY_TASK_ID"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-stage53-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID}"

mkdir -p "$TASK_ROOT"
cd "$FP_STAGE53_RUN"
echo "[stage53-array] task=$SLURM_ARRAY_TASK_ID host=$(hostname) commit=$(git rev-parse HEAD) started=$(date -Is)"
"$PYTHON_BIN" -c 'import matplotlib, numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__, "matplotlib", matplotlib.__version__)'
"$PYTHON_BIN" -m riemann35_patch.stage53_boundary_realizability.run_boundary_realizability \
    --mode unity \
    --epsilon-index "$SLURM_ARRAY_TASK_ID" \
    --output "$TASK_ROOT"
touch "$TASK_ROOT/COMPLETED"
echo "[stage53-array] task=$SLURM_ARRAY_TASK_ID completed=$(date -Is)"

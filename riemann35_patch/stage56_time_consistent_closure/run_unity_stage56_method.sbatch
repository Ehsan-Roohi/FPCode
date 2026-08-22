#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

: "${FP_STAGE56_RUN:?FP_STAGE56_RUN is required}"
: "${FP_STAGE56_RESULT_ROOT:?FP_STAGE56_RESULT_ROOT is required}"

PYTHON_BIN="${FP_STAGE56_PYTHON:-python}"

case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
    0) METHOD="q4_dt2500";  DT="0.0025";    QUADRATURE_NODES="4" ;;
    1) METHOD="q5_dt2500";  DT="0.0025";    QUADRATURE_NODES="5" ;;
    2) METHOD="q5_dt1250";  DT="0.00125";   QUADRATURE_NODES="5" ;;
    3) METHOD="q5_dt0625";  DT="0.000625";  QUADRATURE_NODES="5" ;;
    4) METHOD="q5_dt03125"; DT="0.0003125"; QUADRATURE_NODES="5" ;;
    5) METHOD="q6_dt0625";  DT="0.000625";  QUADRATURE_NODES="6" ;;
    *) echo "ERROR: unknown Stage-56 task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$FP_STAGE56_RUN"
echo "[stage56] host=$(hostname) method=$METHOD commit=$(git rev-parse HEAD) started=$(date -Is)"
"$PYTHON_BIN" -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
"$PYTHON_BIN" -m riemann35_patch.stage56_time_consistent_closure.run_time_consistent_method \
    --method "$METHOD" \
    --output "$FP_STAGE56_RESULT_ROOT" \
    --dt "$DT" \
    --quadrature-nodes "$QUADRATURE_NODES" \
    --final-time 1.0 \
    --sample-interval 0.025 \
    --tail-relaxation-time 0.01
echo "[stage56] method=$METHOD completed=$(date -Is)"

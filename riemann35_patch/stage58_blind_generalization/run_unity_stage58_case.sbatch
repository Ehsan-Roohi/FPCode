#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

: "${MOMENT_STAGE58_RUN:?MOMENT_STAGE58_RUN is required}"
: "${MOMENT_STAGE58_RESULT_ROOT:?MOMENT_STAGE58_RESULT_ROOT is required}"

PYTHON_BIN="${MOMENT_STAGE58_PYTHON:-python}"
POINTS="${MOMENT_STAGE58_POINTS_PER_COMPONENT:-65536}"
REPLICATES="${MOMENT_STAGE58_REPLICATES:-8}"

case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
    0) CASE_NAME="stage57_anchor" ;;
    1) CASE_NAME="hot_dense_shifted" ;;
    2) CASE_NAME="broad_shifted" ;;
    3) CASE_NAME="alternate_weights" ;;
    4) CASE_NAME="anisotropic_3d" ;;
    *)
        echo "ERROR: unknown Stage-58 array task ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$MOMENT_STAGE58_RUN"
echo "[stage58] host=$(hostname) case=$CASE_NAME commit=$(git rev-parse HEAD) started=$(date -Is)"
"$PYTHON_BIN" -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
"$PYTHON_BIN" -m riemann35_patch.stage58_blind_generalization.run_generalization_case \
    --case "$CASE_NAME" \
    --output "$MOMENT_STAGE58_RESULT_ROOT" \
    --qmc-dt 0.0003125 \
    --coarse-dt 0.000625 \
    --fine-dt 0.0003125 \
    --final-time 1.0 \
    --sample-interval 0.025 \
    --points-per-component "$POINTS" \
    --replicates "$REPLICATES" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --quadrature-nodes 5
echo "[stage58] case=$CASE_NAME completed=$(date -Is)"

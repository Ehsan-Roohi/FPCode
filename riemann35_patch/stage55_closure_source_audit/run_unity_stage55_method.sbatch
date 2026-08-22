#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

: "${FP_STAGE55_RUN:?FP_STAGE55_RUN is required}"
: "${FP_STAGE55_RESULT_ROOT:?FP_STAGE55_RESULT_ROOT is required}"

PYTHON_BIN="${FP_STAGE55_PYTHON:-python}"
FINE_POINTS="${FP_STAGE55_FINE_POINTS:-32768}"
REPLICATES="${FP_STAGE55_REPLICATES:-4}"

case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
    0)
        METHOD="qmc_reference"
        DT="0.00125"
        POINTS="$FINE_POINTS"
        LOCAL_REPLICATES="$REPLICATES"
        QUADRATURE_NODES="4"
        ;;
    1)
        METHOD="gaussian_hyqmom35"
        DT="0.00125"
        POINTS="8"
        LOCAL_REPLICATES="1"
        QUADRATURE_NODES="4"
        ;;
    2)
        METHOD="dynamic_unprojected"
        DT="0.0025"
        POINTS="8"
        LOCAL_REPLICATES="1"
        QUADRATURE_NODES="4"
        ;;
    3)
        METHOD="projected_tail_base"
        DT="0.0025"
        POINTS="8"
        LOCAL_REPLICATES="1"
        QUADRATURE_NODES="4"
        ;;
    4)
        METHOD="projected_tail_node_refined"
        DT="0.0025"
        POINTS="8"
        LOCAL_REPLICATES="1"
        QUADRATURE_NODES="5"
        ;;
    5)
        METHOD="projected_tail_time_refined"
        DT="0.00125"
        POINTS="8"
        LOCAL_REPLICATES="1"
        QUADRATURE_NODES="5"
        ;;
    *)
        echo "ERROR: unknown Stage-55 array task ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$FP_STAGE55_RUN"
echo "[stage55] host=$(hostname) method=$METHOD commit=$(git rev-parse HEAD) started=$(date -Is)"
"$PYTHON_BIN" -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
"$PYTHON_BIN" -m riemann35_patch.stage55_closure_source_audit.run_closure_method \
    --method "$METHOD" \
    --output "$FP_STAGE55_RESULT_ROOT" \
    --dt "$DT" \
    --final-time 1.0 \
    --sample-interval 0.025 \
    --points-per-component "$POINTS" \
    --replicates "$LOCAL_REPLICATES" \
    --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --tail-relaxation-time 0.01 \
    --high-order-quadrature-nodes "$QUADRATURE_NODES"
echo "[stage55] method=$METHOD completed=$(date -Is)"

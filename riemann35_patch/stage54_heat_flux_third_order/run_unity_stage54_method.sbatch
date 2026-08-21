#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

: "${FP_STAGE54_RUN:?FP_STAGE54_RUN is required}"
: "${FP_STAGE54_RESULT_ROOT:?FP_STAGE54_RESULT_ROOT is required}"

PYTHON_BIN="${FP_STAGE54_PYTHON:-python}"
BASE_POINTS="${FP_STAGE54_BASE_POINTS:-8192}"
FINE_POINTS="${FP_STAGE54_FINE_POINTS:-32768}"
REPLICATES="${FP_STAGE54_REPLICATES:-4}"

case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
    0)
        METHOD="qmc_base"
        DT="0.0025"
        POINTS="$BASE_POINTS"
        LOCAL_REPLICATES="$REPLICATES"
        ;;
    1)
        METHOD="qmc_node_refined"
        DT="0.0025"
        POINTS="$FINE_POINTS"
        LOCAL_REPLICATES="$REPLICATES"
        ;;
    2)
        METHOD="qmc_time_refined"
        DT="0.00125"
        POINTS="$FINE_POINTS"
        LOCAL_REPLICATES="$REPLICATES"
        ;;
    3)
        METHOD="proposed_hyqmom35"
        DT="0.00125"
        POINTS="8"
        LOCAL_REPLICATES="1"
        ;;
    4)
        METHOD="grad_comparator"
        DT="0.00125"
        POINTS="8"
        LOCAL_REPLICATES="1"
        ;;
    5)
        METHOD="positive_tail_memory"
        DT="0.00125"
        POINTS="$BASE_POINTS"
        LOCAL_REPLICATES="$REPLICATES"
        ;;
    *)
        echo "ERROR: unknown Stage-54 array task ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$FP_STAGE54_RUN"
echo "[stage54] host=$(hostname) method=$METHOD commit=$(git rev-parse HEAD) started=$(date -Is)"
"$PYTHON_BIN" -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
"$PYTHON_BIN" -m riemann35_patch.stage54_heat_flux_third_order.run_heat_flux_method \
    --method "$METHOD" \
    --output "$FP_STAGE54_RESULT_ROOT" \
    --dt "$DT" \
    --final-time 1.0 \
    --sample-interval 0.025 \
    --sensor-interval 0.025 \
    --points-per-component "$POINTS" \
    --replicates "$LOCAL_REPLICATES" \
    --workers "${SLURM_CPUS_PER_TASK:-4}"
echo "[stage54] method=$METHOD completed=$(date -Is)"

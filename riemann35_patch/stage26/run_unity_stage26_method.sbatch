#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

: "${FP_STAGE26_RUN:?FP_STAGE26_RUN is required}"
: "${FP_STAGE26_RESULT_ROOT:?FP_STAGE26_RESULT_ROOT is required}"

PYTHON_BIN="${FP_STAGE26_PYTHON:-python}"
DT="${FP_STAGE26_DT:-0.0025}"
FINAL_TIME="${FP_STAGE26_FINAL_TIME:-1.0}"
SAMPLE_EVERY="${FP_STAGE26_SAMPLE_EVERY:-10}"
REFERENCE_POINTS="${FP_STAGE26_REFERENCE_POINTS:-4096}"
ADAPTIVE_POINTS="${FP_STAGE26_ADAPTIVE_POINTS:-1024}"
REPLICATES="${FP_STAGE26_REPLICATES:-4}"
SENSOR_EVERY="${FP_STAGE26_SENSOR_EVERY:-10}"

case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
    0)
        METHOD="full_fp_qmc"
        POINTS="$REFERENCE_POINTS"
        LOCAL_REPLICATES="$REPLICATES"
        ;;
    1)
        METHOD="stage9_mixture"
        POINTS="8"
        LOCAL_REPLICATES="1"
        ;;
    2)
        METHOD="grad_gqmom"
        POINTS="8"
        LOCAL_REPLICATES="1"
        ;;
    3)
        METHOD="adaptive_memory"
        POINTS="$ADAPTIVE_POINTS"
        LOCAL_REPLICATES="$REPLICATES"
        ;;
    *)
        echo "ERROR: unknown Stage-26 array task ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$FP_STAGE26_RUN"
echo "[stage26] host=$(hostname) method=$METHOD commit=$(git rev-parse HEAD) started=$(date -Is)"
"$PYTHON_BIN" -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
"$PYTHON_BIN" riemann35_patch/stage26/run_four_delta_method.py \
    --method "$METHOD" \
    --output "$FP_STAGE26_RESULT_ROOT" \
    --dt "$DT" \
    --final-time "$FINAL_TIME" \
    --sample-every "$SAMPLE_EVERY" \
    --points-per-component "$POINTS" \
    --replicates "$LOCAL_REPLICATES" \
    --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --sensor-every "$SENSOR_EVERY"
echo "[stage26] method=$METHOD completed=$(date -Is)"

#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

: "${MOMENT_STAGE56_RUN:?MOMENT_STAGE56_RUN is required}"
: "${MOMENT_STAGE56_RESULT_ROOT:?MOMENT_STAGE56_RESULT_ROOT is required}"

PYTHON_BIN="${MOMENT_STAGE56_PYTHON:-python}"
FINE_POINTS="${MOMENT_STAGE56_FINE_POINTS:-32768}"
REPLICATES="${MOMENT_STAGE56_REPLICATES:-4}"

case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
    0)
        METHOD="qmc_reference"
        DT="0.0003125"
        POINTS="$FINE_POINTS"
        LOCAL_REPLICATES="$REPLICATES"
        ;;
    1)
        METHOD="legacy_lie_h0"
        DT="0.0025"
        POINTS="8"
        LOCAL_REPLICATES="1"
        ;;
    2)
        METHOD="strang_h0"
        DT="0.0025"
        POINTS="8"
        LOCAL_REPLICATES="1"
        ;;
    3)
        METHOD="strang_h1"
        DT="0.00125"
        POINTS="8"
        LOCAL_REPLICATES="1"
        ;;
    4)
        METHOD="strang_h2"
        DT="0.000625"
        POINTS="8"
        LOCAL_REPLICATES="1"
        ;;
    5)
        METHOD="strang_h3"
        DT="0.0003125"
        POINTS="8"
        LOCAL_REPLICATES="1"
        ;;
    *)
        echo "ERROR: unknown Stage-56 array task ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$MOMENT_STAGE56_RUN"
echo "[stage56] host=$(hostname) method=$METHOD commit=$(git rev-parse HEAD) started=$(date -Is)"
"$PYTHON_BIN" -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'
"$PYTHON_BIN" -m riemann35_patch.stage56_tail_time_integrator.run_time_method \
    --method "$METHOD" \
    --output "$MOMENT_STAGE56_RESULT_ROOT" \
    --dt "$DT" \
    --final-time 1.0 \
    --sample-interval 0.025 \
    --points-per-component "$POINTS" \
    --replicates "$LOCAL_REPLICATES" \
    --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --tail-relaxation-time 0.01 \
    --quadrature-nodes 5
echo "[stage56] method=$METHOD completed=$(date -Is)"

#!/bin/bash
set -euo pipefail

: "${STAGE25A_LITE_REPO:?STAGE25A_LITE_REPO is required}"
: "${STAGE25A_LITE_ROOT:?STAGE25A_LITE_ROOT is required}"
: "${STAGE25A_LITE_PYTHON:?STAGE25A_LITE_PYTHON is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

METHODS=(macro adaptive full_dvm)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="$STAGE25A_LITE_ROOT/$METHOD"

cd "$STAGE25A_LITE_REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "[stage25a-lite] method=$METHOD host=$(hostname) started=$(date -Is)"
"$STAGE25A_LITE_PYTHON" riemann35_patch/stage25a/run_normal_shock_lite.py run \
  --method "$METHOD" \
  --steps 600 \
  --checkpoint-every 50 \
  --progress-every 10 \
  --output "$OUTPUT"
echo "[stage25a-lite] method=$METHOD completed=$(date -Is)"

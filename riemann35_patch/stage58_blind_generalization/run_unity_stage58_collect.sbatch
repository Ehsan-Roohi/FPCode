#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

: "${MOMENT_STAGE58_RUN:?MOMENT_STAGE58_RUN is required}"
: "${MOMENT_STAGE58_RESULT_ROOT:?MOMENT_STAGE58_RESULT_ROOT is required}"
: "${MOMENT_STAGE58_BUNDLE:?MOMENT_STAGE58_BUNDLE is required}"

PYTHON_BIN="${MOMENT_STAGE58_PYTHON:-python}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "$MOMENT_STAGE58_RUN"
"$PYTHON_BIN" -m riemann35_patch.stage58_blind_generalization.collect_generalization_gate \
    --root "$MOMENT_STAGE58_RESULT_ROOT" \
    --bundle "$MOMENT_STAGE58_BUNDLE"

cd "$(dirname "$MOMENT_STAGE58_BUNDLE")"
sha256sum -c "$(basename "$MOMENT_STAGE58_BUNDLE").sha256.txt"
ls -lh "$MOMENT_STAGE58_BUNDLE" "$MOMENT_STAGE58_BUNDLE.sha256.txt"

#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00

set -euo pipefail

: "${MOMENT_STAGE57_RUN:?MOMENT_STAGE57_RUN is required}"
: "${MOMENT_STAGE57_RESULT_ROOT:?MOMENT_STAGE57_RESULT_ROOT is required}"
: "${MOMENT_STAGE57_BUNDLE:?MOMENT_STAGE57_BUNDLE is required}"

PYTHON_BIN="${MOMENT_STAGE57_PYTHON:-python}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "$MOMENT_STAGE57_RUN"
"$PYTHON_BIN" -m riemann35_patch.stage57_persistent_four_population.collect_persistent_gate \
    --root "$MOMENT_STAGE57_RESULT_ROOT" \
    --bundle "$MOMENT_STAGE57_BUNDLE"

cd "$(dirname "$MOMENT_STAGE57_BUNDLE")"
sha256sum -c "$(basename "$MOMENT_STAGE57_BUNDLE").sha256.txt"
ls -lh "$MOMENT_STAGE57_BUNDLE" "$MOMENT_STAGE57_BUNDLE.sha256.txt"


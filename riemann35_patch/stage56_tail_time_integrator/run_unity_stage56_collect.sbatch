#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00

set -euo pipefail

: "${MOMENT_STAGE56_RUN:?MOMENT_STAGE56_RUN is required}"
: "${MOMENT_STAGE56_RESULT_ROOT:?MOMENT_STAGE56_RESULT_ROOT is required}"
: "${MOMENT_STAGE56_BUNDLE:?MOMENT_STAGE56_BUNDLE is required}"

PYTHON_BIN="${MOMENT_STAGE56_PYTHON:-python}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "$MOMENT_STAGE56_RUN"
"$PYTHON_BIN" -m riemann35_patch.stage56_tail_time_integrator.collect_time_gate \
    --root "$MOMENT_STAGE56_RESULT_ROOT" \
    --bundle "$MOMENT_STAGE56_BUNDLE"

cd "$(dirname "$MOMENT_STAGE56_BUNDLE")"
sha256sum -c "$(basename "$MOMENT_STAGE56_BUNDLE").sha256.txt"
ls -lh "$MOMENT_STAGE56_BUNDLE" "$MOMENT_STAGE56_BUNDLE.sha256.txt"

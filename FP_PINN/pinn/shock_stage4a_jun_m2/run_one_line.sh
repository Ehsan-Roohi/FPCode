#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${STAGE4A_PYTHON:-python3}"
REF_PROFILE="${STAGE4A_REFERENCE:-$ROOT/reference/standing_M2_x40_hmom_dvm_densemicro_H100.npz}"
REF_MICRO="${STAGE4A_MICROANCHORS:-$ROOT/reference/standing_M2_x40_hmom_dvm_densemicro_H100_microanchors.npz}"
OUTPUT_DIR="${STAGE4A_OUTPUT:-$ROOT/outputs/stage4a_jun_m2}"

if [[ ! -f "$REF_PROFILE" || ! -f "$REF_MICRO" ]]; then
    echo "Missing reference NPZ files under $ROOT/reference" >&2
    exit 2
fi

"$PYTHON_BIN" shock_stage4a.py \
  --reference "$REF_PROFILE" \
  --microanchors "$REF_MICRO" \
  --output "$OUTPUT_DIR" \
  "${@}"

ARCHIVE_ROOT="$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null || cd "$ROOT/.." && pwd)"
ARCHIVE="$ARCHIVE_ROOT/FP_PINN_STAGE4A_JUN_ZHANG_M2_SHOCK_COMPLETE.zip"
rm -f "$ARCHIVE"
(
  cd "$ROOT"
  zip -qr "$ARCHIVE" README.md requirements.txt shock_stage4a.py run_one_line.sh run_stage4a.sbatch reference outputs
)
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
echo "COMPLETE: $ARCHIVE"

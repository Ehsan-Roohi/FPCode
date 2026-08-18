#!/bin/bash
set -euo pipefail

: "${STAGE25A_LITE_REPO:?STAGE25A_LITE_REPO is required}"
: "${STAGE25A_LITE_ROOT:?STAGE25A_LITE_ROOT is required}"
: "${STAGE25A_LITE_PYTHON:?STAGE25A_LITE_PYTHON is required}"
: "${STAGE25A_LITE_BUNDLE:?STAGE25A_LITE_BUNDLE is required}"

cd "$STAGE25A_LITE_REPO"
export PYTHONUNBUFFERED=1
"$STAGE25A_LITE_PYTHON" riemann35_patch/stage25a/run_normal_shock_lite.py collect \
  --input-root "$STAGE25A_LITE_ROOT" \
  --bundle "$STAGE25A_LITE_BUNDLE"

echo "RESULT_BUNDLE=$STAGE25A_LITE_BUNDLE"

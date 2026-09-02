#!/usr/bin/env bash
set -euo pipefail
ROOT="${FP_H1_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
HERE="$ROOT/FP_PINN/pinn/normal_shock_h1"
REF="${FP_H2_REFERENCE:-}"
MACH="${FP_H2_MACH:-2}"
[[ -n "$REF" ]] || { echo "REFERENCE_REQUIRED: export FP_H2_REFERENCE to an independent DVM/DGFS/DSMC CSV or NPZ" >&2; exit 2; }
[[ -f "$REF" ]] || { echo "REFERENCE_REQUIRED: file not found: $REF" >&2; exit 2; }
cd "$HERE"
python -m unittest -v test_shock_physics.py test_fp_operator.py test_h2_reference.py
OUT="$HERE/outputs/h2_reference_m${MACH//./p}_$(date -u +%Y%m%dT%H%M%SZ)"
python audit_h2_reference.py --reference "$REF" --mach "$MACH" --output "$OUT"
echo "H2_REFERENCE_OUTPUT=$OUT"

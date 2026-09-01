#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
S72=${STAGE72_DIR:-/project/pi_roohie_umass_edu/github_sync/FPCode-stage72-density-jacobian/results/riemann35_stage72/density_jacobian_fix_20260831-224405}
[[ -f "$S72/stage71_generalization_summary.json" ]] || { echo "Stage72 directory missing: $S72" >&2; exit 2; }
cd "$ROOT"
python riemann35_patch/stage73_heat_flux_conditioning/analyze_heat_flux_conditioning.py --stage72-dir "$S72"
echo "STAGE73_RESULT=$S72/STAGE73_RESULTS.md"
echo "STAGE73_SUMMARY=$S72/stage73_heat_flux_conditioning_summary.json"

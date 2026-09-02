#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
python -m py_compile riemann35_patch/stage76_spatial_kinetic_dvm/run_stage76.py riemann35_patch/stage76_spatial_kinetic_dvm/test_preflight.py
python riemann35_patch/stage76_spatial_kinetic_dvm/test_preflight.py
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/riemann35_stage76/spatial_kinetic_dvm_${STAMP}"
mkdir -p "$OUT/logs"
BUNDLE="$OUT/STAGE76_SPATIAL_KINETIC_DVM_RESULTS_${STAMP}.zip"
JOB=$(sbatch --parsable -p cpu -N 1 -n 1 -c 2 -t 12:00:00 --mem=16G -J m76-spatial-dvm -o "$OUT/logs/m76_%j.out" -e "$OUT/logs/m76_%j.err" --export=ALL,ROOT="$ROOT",OUT="$OUT",BUNDLE="$BUNDLE" "$ROOT/riemann35_patch/stage76_spatial_kinetic_dvm/run_stage76_job.sh")
echo "STAGE76_JOB=$JOB"
echo "RESULT_DIR=$OUT"
echo "BUNDLE=$BUNDLE"

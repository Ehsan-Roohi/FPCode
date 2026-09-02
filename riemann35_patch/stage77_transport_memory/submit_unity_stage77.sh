#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
python -m py_compile riemann35_patch/stage77_transport_memory/run_stage77.py riemann35_patch/stage77_transport_memory/test_preflight.py
python riemann35_patch/stage77_transport_memory/test_preflight.py
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/riemann35_stage77/transport_memory_${STAMP}"
mkdir -p "$OUT/logs"
BUNDLE="$OUT/STAGE77_TRANSPORT_MEMORY_RESULTS_${STAMP}.zip"
JOB=$(sbatch --parsable -p cpu -N 1 -n 1 -c 2 -t 01:00:00 --mem=8G -J m77-transport -o "$OUT/logs/m77_%j.out" -e "$OUT/logs/m77_%j.err" --export=ALL,ROOT="$ROOT",OUT="$OUT",BUNDLE="$BUNDLE" "$ROOT/riemann35_patch/stage77_transport_memory/run_stage77_job.sh")
echo "STAGE77_JOB=$JOB"
echo "RESULT_DIR=$OUT"
echo "BUNDLE=$BUNDLE"

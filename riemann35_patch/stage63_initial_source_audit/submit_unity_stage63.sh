#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/riemann35_stage63/initial_source_${STAMP}"
mkdir -p "$OUT/logs"
CASES=(stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d)
JOBS=()
for i in "${!CASES[@]}"; do
  c=${CASES[$i]}
  j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:20:00 --mem=2G \
    -J "m63-$i" -o "$OUT/logs/m63-${i}_%j.out" -e "$OUT/logs/m63-${i}_%j.err" \
    --wrap="cd '$ROOT' && python riemann35_patch/stage63_initial_source_audit/run_initial_source_case.py --case '$c' --output '$OUT'")
  JOBS+=("$j")
done
DEP=$(IFS=:; echo "${JOBS[*]}")
COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:10:00 --mem=1G \
  -J m63-collect --dependency=afterok:$DEP \
  -o "$OUT/logs/m63-collect_%j.out" -e "$OUT/logs/m63-collect_%j.err" \
  --wrap="cd '$ROOT' && python riemann35_patch/stage63_initial_source_audit/collect_initial_source.py '$OUT' && cd '$OUT' && zip -q -r STAGE63_INITIAL_SOURCE_RESULTS_${STAMP}.zip STAGE63_RESULTS.md stage63_*_summary.json logs && sha256sum STAGE63_INITIAL_SOURCE_RESULTS_${STAMP}.zip > STAGE63_INITIAL_SOURCE_RESULTS_${STAMP}.zip.sha256.txt")
echo "STAGE63_JOBS=${JOBS[*]}"
echo "COLLECT_JOB=$COL"
echo "RESULT_DIR=$OUT"

#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/riemann35_stage60/fourth_order_memory_$STAMP"
mkdir -p "$OUT" "$OUT/logs"
CASES=(stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d)
JOBS=()
for i in "${!CASES[@]}"; do
  c=${CASES[$i]}
  j=$(sbatch --parsable -J "m60-$i" -p cpu -N 1 -n 1 -c 1 -t 01:00:00 -o "$OUT/logs/m60-$i-%j.out" -e "$OUT/logs/m60-$i-%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage60_fourth_order_memory/run_fourth_order_case.py --case '$c' --output '$OUT'")
  JOBS+=("$j")
done
DEP=$(IFS=:; echo "${JOBS[*]}")
COL=$(sbatch --parsable -J m60-collect -p cpu -N 1 -n 1 -c 1 -t 00:10:00 --dependency=afterok:$DEP -o "$OUT/logs/m60-collect-%j.out" -e "$OUT/logs/m60-collect-%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage60_fourth_order_memory/collect_fourth_order.py '$OUT' && cd '$OUT' && zip -q -r STAGE60_FOURTH_ORDER_MEMORY_RESULTS_${STAMP}.zip STAGE60_RESULTS.md stage60_*.json stage60_*.csv logs && sha256sum STAGE60_FOURTH_ORDER_MEMORY_RESULTS_${STAMP}.zip > STAGE60_FOURTH_ORDER_MEMORY_RESULTS_${STAMP}.zip.sha256.txt")
echo "STAGE60_JOBS=${JOBS[*]}"; echo "COLLECT_JOB=$COL"; echo "RESULT_DIR=$OUT"

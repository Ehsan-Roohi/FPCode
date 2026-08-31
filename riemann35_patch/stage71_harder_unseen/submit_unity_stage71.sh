#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
python riemann35_patch/stage71_harder_unseen/test_registry.py
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/riemann35_stage71/harder_unseen_${STAMP}"
mkdir -p "$OUT/logs"
CASES=(rare_beam_3d dense_hot_extreme dilute_broad strong_anisotropy balanced_cross_3d)
JOBS=()
for i in "${!CASES[@]}"; do
  c=${CASES[$i]}
  j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 8 -t 06:00:00 --mem=20G -J "m71-$i" -o "$OUT/logs/m71-${i}_%j.out" -e "$OUT/logs/m71-${i}_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage71_harder_unseen/run_hard_case.py --case '$c' --output '$OUT' --points-per-component 131072 --replicates 8 --workers 8")
  JOBS+=("$j")
done
DEP=$(IFS=:; echo "${JOBS[*]}")
BUNDLE="$OUT/STAGE71_HARDER_UNSEEN_RESULTS_${STAMP}.zip"
COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:15:00 --mem=4G -J m71-collect --dependency=afterok:$DEP -o "$OUT/logs/m71-collect_%j.out" -e "$OUT/logs/m71-collect_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage71_harder_unseen/collect_hard_gate.py --root '$OUT' --bundle '$BUNDLE'")
echo "STAGE71_JOBS=${JOBS[*]}"
echo "COLLECT_JOB=$COL"
echo "RESULT_DIR=$OUT"
echo "BUNDLE=$BUNDLE"

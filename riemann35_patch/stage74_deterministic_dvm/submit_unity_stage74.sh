#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
python riemann35_patch/stage74_deterministic_dvm/test_preflight.py
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/riemann35_stage74/deterministic_dvm_${STAMP}"
mkdir -p "$OUT/logs"
CASES=(rare_beam_3d dense_hot_extreme dilute_broad)
JOBS=()
for i in "${!CASES[@]}"; do
  c=${CASES[$i]}
  j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 08:00:00 --mem=16G -J "m74-$i" -o "$OUT/logs/m74-${i}_%j.out" -e "$OUT/logs/m74-${i}_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage74_deterministic_dvm/run_dvm_case.py --case '$c' --output '$OUT'")
  JOBS+=("$j")
done
DEP=$(IFS=:; echo "${JOBS[*]}")
BUNDLE="$OUT/STAGE74_DETERMINISTIC_DVM_RESULTS_${STAMP}.zip"
COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:15:00 --mem=4G -J m74-collect --dependency=afterok:$DEP -o "$OUT/logs/m74-collect_%j.out" -e "$OUT/logs/m74-collect_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage74_deterministic_dvm/collect_dvm_gate.py --root '$OUT' --bundle '$BUNDLE'")
echo "STAGE74_JOBS=${JOBS[*]}"
echo "COLLECT_JOB=$COL"
echo "RESULT_DIR=$OUT"
echo "BUNDLE=$BUNDLE"

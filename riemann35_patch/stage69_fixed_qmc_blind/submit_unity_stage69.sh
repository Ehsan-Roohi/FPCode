#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$ROOT/results/riemann35_stage69/fixed_qmc_blind_${STAMP}"; mkdir -p "$OUT/logs"
CASES=(stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d); JOBS=()
for i in "${!CASES[@]}"; do c=${CASES[$i]}; j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 8 -t 02:30:00 --mem=12G -J "m69-$i" -o "$OUT/logs/m69-${i}_%j.out" -e "$OUT/logs/m69-${i}_%j.err" --wrap="cd '$ROOT' && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python riemann35_patch/stage69_fixed_qmc_blind/run_fixed_case.py --case '$c' --output '$OUT' --workers 8"); JOBS+=("$j"); done
DEP=$(IFS=:; echo "${JOBS[*]}"); BUNDLE="$OUT/STAGE69_FIXED_QMC_BLIND_RESULTS_${STAMP}.zip"
COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:15:00 --mem=2G -J m69-collect --dependency=afterok:$DEP -o "$OUT/logs/m69-collect_%j.out" -e "$OUT/logs/m69-collect_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage58_blind_generalization/collect_generalization_gate.py --root '$OUT' --bundle '$BUNDLE'; rc=\$?; if [ -f '$BUNDLE' ]; then sha256sum '$BUNDLE' > '$BUNDLE.sha256.txt'; fi; exit \$rc")
echo "STAGE69_JOBS=${JOBS[*]}"; echo "COLLECT_JOB=$COL"; echo "RESULT_DIR=$OUT"; echo "BUNDLE=$BUNDLE"

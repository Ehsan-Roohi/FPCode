#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$ROOT/results/riemann35_stage66/canonical_covariance_${STAMP}"; mkdir -p "$OUT/logs"
CASES=(stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d); JOBS=()
for i in "${!CASES[@]}"; do c=${CASES[$i]}; j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:30:00 --mem=2G -J "m66-$i" -o "$OUT/logs/m66-${i}_%j.out" -e "$OUT/logs/m66-${i}_%j.err" --wrap="cd '$ROOT' && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python riemann35_patch/stage66_canonical_covariance/run_canonical_case.py --case '$c' --output '$OUT'"); JOBS+=("$j"); done
DEP=$(IFS=:; echo "${JOBS[*]}"); COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:05:00 --mem=1G -J m66-collect --dependency=afterok:$DEP -o "$OUT/logs/m66-collect_%j.out" -e "$OUT/logs/m66-collect_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage66_canonical_covariance/collect_canonical.py '$OUT'")
echo "STAGE66_JOBS=${JOBS[*]}"; echo "COLLECT_JOB=$COL"; echo "RESULT_DIR=$OUT"

#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$ROOT/results/riemann35_stage67/rotation_covariance_${STAMP}"; mkdir -p "$OUT/logs"
CASES=(stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d); JOBS=()
for i in "${!CASES[@]}"; do c=${CASES[$i]}; j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:45:00 --mem=2G -J "m67-$i" -o "$OUT/logs/m67-${i}_%j.out" -e "$OUT/logs/m67-${i}_%j.err" --wrap="cd '$ROOT' && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python riemann35_patch/stage67_rotation_covariance/run_rotation_case.py --case '$c' --output '$OUT'"); JOBS+=("$j"); done
DEP=$(IFS=:; echo "${JOBS[*]}"); COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:05:00 --mem=1G -J m67-collect --dependency=afterok:$DEP -o "$OUT/logs/m67-collect_%j.out" -e "$OUT/logs/m67-collect_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage67_rotation_covariance/collect_rotation.py '$OUT'")
echo "STAGE67_JOBS=${JOBS[*]}"; echo "COLLECT_JOB=$COL"; echo "RESULT_DIR=$OUT"

#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/riemann35_stage62/latent_equivalence_${STAMP}"; mkdir -p "$OUT/logs"
CASES=(stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d)
JOBS=()
for i in "${!CASES[@]}"; do c=${CASES[$i]}; j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 2 -t 01:00:00 --mem=4G -J "m62-$i" -o "$OUT/logs/m62-${i}_%j.out" -e "$OUT/logs/m62-${i}_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage62_latent_equivalence/run_latent_equivalence_case.py --case '$c' --output '$OUT'"); JOBS+=("$j"); done
DEP=$(IFS=:; echo "${JOBS[*]}")
COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:10:00 --mem=1G -J m62-collect --dependency=afterok:$DEP -o "$OUT/logs/m62-collect_%j.out" -e "$OUT/logs/m62-collect_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage62_latent_equivalence/collect_latent_equivalence.py '$OUT'")
echo "STAGE62_JOBS=${JOBS[*]}"; echo "COLLECT_JOB=$COL"; echo "RESULT_DIR=$OUT"

#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
S58=${STAGE58_DIR:-/project/pi_roohie_umass_edu/github_sync/MomentCode-stage58-blind-20260827-011710/results/riemann35_stage58/blind_generalization_20260827-011710}
[[ -f "$S58/stage58_stage57_anchor.npz" ]] || { echo "Stage58 directory missing: $S58" >&2; exit 2; }
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$ROOT/results/riemann35_stage68/qmc_density_continuity_${STAMP}"; mkdir -p "$OUT/logs"
CASES=(stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d); JOBS=()
for i in "${!CASES[@]}"; do c=${CASES[$i]}; j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:10:00 --mem=1G -J "m68-$i" -o "$OUT/logs/m68-${i}_%j.out" -e "$OUT/logs/m68-${i}_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage68_qmc_density_continuity/run_density_continuity_case.py --case '$c' --stage58-dir '$S58' --output '$OUT'"); JOBS+=("$j"); done
DEP=$(IFS=:; echo "${JOBS[*]}"); COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:05:00 --mem=1G -J m68-collect --dependency=afterok:$DEP -o "$OUT/logs/m68-collect_%j.out" -e "$OUT/logs/m68-collect_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage68_qmc_density_continuity/collect_density_continuity.py '$OUT'")
echo "STAGE58_DIR=$S58"; echo "STAGE68_JOBS=${JOBS[*]}"; echo "COLLECT_JOB=$COL"; echo "RESULT_DIR=$OUT"

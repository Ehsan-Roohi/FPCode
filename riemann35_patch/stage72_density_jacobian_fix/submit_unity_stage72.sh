#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BASE=${STAGE71_DIR:-/project/pi_roohie_umass_edu/github_sync/FPCode-stage71-hard-unseen/results/riemann35_stage71/harder_unseen_20260831-130621}
[[ -f "$BASE/stage71_generalization_summary.json" ]] || { echo "Stage71 directory missing: $BASE" >&2; exit 2; }
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$ROOT/results/riemann35_stage72/density_jacobian_fix_${STAMP}"; mkdir -p "$OUT/logs"
CASES=(rare_beam_3d dense_hot_extreme dilute_broad strong_anisotropy balanced_cross_3d); JOBS=()
for i in "${!CASES[@]}"; do c=${CASES[$i]}; j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 01:00:00 --mem=4G -J "m72-$i" -o "$OUT/logs/m72-${i}_%j.out" -e "$OUT/logs/m72-${i}_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage72_density_jacobian_fix/run_fixed_case.py --case '$c' --stage71-dir '$BASE' --output '$OUT'"); JOBS+=("$j"); done
DEP=$(IFS=:; echo "${JOBS[*]}"); BUNDLE="$OUT/STAGE72_DENSITY_JACOBIAN_FIX_RESULTS_${STAMP}.zip"
COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:15:00 --mem=4G -J m72-collect --dependency=afterok:$DEP -o "$OUT/logs/m72-collect_%j.out" -e "$OUT/logs/m72-collect_%j.err" --wrap="cd '$ROOT' && TMP=/tmp/stage72-gate-\$SLURM_JOB_ID.zip && python riemann35_patch/stage71_harder_unseen/collect_hard_gate.py --root '$OUT' --bundle \"\$TMP\" && rm -f \"\$TMP\" \"\$TMP.sha256.txt\" && python riemann35_patch/stage72_density_jacobian_fix/collect_fix.py --old-dir '$BASE' --new-dir '$OUT' && python riemann35_patch/stage71_harder_unseen/collect_hard_gate.py --root '$OUT' --bundle '$BUNDLE'")
echo "STAGE71_DIR=$BASE"; echo "STAGE72_JOBS=${JOBS[*]}"; echo "COLLECT_JOB=$COL"; echo "RESULT_DIR=$OUT"; echo "BUNDLE=$BUNDLE"

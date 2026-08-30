#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SEARCH=${STAGE58_SEARCH_ROOT:-/project/pi_roohie_umass_edu/github_sync}
if [[ -n "${STAGE58_DIR:-}" ]]; then S58="$STAGE58_DIR"; else
  S58=$(python - "$SEARCH" <<'PY'
import os,sys
root=sys.argv[1]; found=[]
for dp,_,fs in os.walk(root):
    if 'stage58_stage57_anchor.npz' in fs:
        p=os.path.join(dp,'stage58_stage57_anchor.npz'); found.append((os.path.getmtime(p),dp))
print(max(found)[1] if found else '')
PY
)
fi
[[ -n "${S58:-}" ]] || { echo 'Stage58 result directory not found' >&2; exit 2; }
CASES=(stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d)
for c in "${CASES[@]}"; do [[ -f "$S58/stage58_${c}.npz" ]] || { echo "missing $S58/stage58_${c}.npz" >&2; exit 3; }; done
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$ROOT/results/riemann35_stage65/density_aware_${STAMP}"; mkdir -p "$OUT/logs"; JOBS=()
for i in "${!CASES[@]}"; do c=${CASES[$i]}; j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:10:00 --mem=2G -J "m65-$i" -o "$OUT/logs/m65-${i}_%j.out" -e "$OUT/logs/m65-${i}_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage65_density_aware_reference/run_density_repair_case.py --case '$c' --stage58-dir '$S58' --output '$OUT'"); JOBS+=("$j"); done
DEP=$(IFS=:; echo "${JOBS[*]}")
COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:05:00 --mem=1G -J m65-collect --dependency=afterok:$DEP -o "$OUT/logs/m65-collect_%j.out" -e "$OUT/logs/m65-collect_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage65_density_aware_reference/collect_density_repair.py '$OUT'")
echo "STAGE58_DIR=$S58"; echo "STAGE65_JOBS=${JOBS[*]}"; echo "COLLECT_JOB=$COL"; echo "RESULT_DIR=$OUT"

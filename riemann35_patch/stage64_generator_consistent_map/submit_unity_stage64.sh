#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SEARCH=${STAGE58_SEARCH_ROOT:-/project/pi_roohie_umass_edu/github_sync}
if [[ -n "${STAGE58_DIR:-}" ]]; then
  S58="$STAGE58_DIR"
else
  # Avoid `... | head -n1` under pipefail: upstream commands can receive
  # SIGPIPE and terminate this submit script before any sbatch is issued.
  S58=$(find "$SEARCH" -type f -name 'stage58_stage57_anchor.npz' -printf '%T@ %h\n' 2>/dev/null | sort -nr | sed -n '1{s/^[^ ]* //;p;}')
fi
[[ -n "${S58:-}" ]] || { echo 'Stage58 result directory not found' >&2; exit 2; }
for c in stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d; do
  [[ -f "$S58/stage58_${c}.npz" ]] || { echo "missing $S58/stage58_${c}.npz" >&2; exit 3; }
done
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/riemann35_stage64/generator_consistent_${STAMP}"
mkdir -p "$OUT/logs"
CASES=(stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d)
JOBS=()
for i in "${!CASES[@]}"; do
  c=${CASES[$i]}
  j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 2 -t 02:00:00 --mem=4G -J "m64-$i" -o "$OUT/logs/m64-${i}_%j.out" -e "$OUT/logs/m64-${i}_%j.err" --wrap="cd '$ROOT' && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python riemann35_patch/stage64_generator_consistent_map/run_generator_consistent_case.py --case '$c' --stage58-dir '$S58' --output '$OUT'")
  JOBS+=("$j")
done
DEP=$(IFS=:; echo "${JOBS[*]}")
COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:10:00 --mem=1G -J m64-collect --dependency=afterok:$DEP -o "$OUT/logs/m64-collect_%j.out" -e "$OUT/logs/m64-collect_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage64_generator_consistent_map/collect_generator_consistent.py '$OUT'")
echo "STAGE58_DIR=$S58"
echo "STAGE64_JOBS=${JOBS[*]}"
echo "COLLECT_JOB=$COL"
echo "RESULT_DIR=$OUT"

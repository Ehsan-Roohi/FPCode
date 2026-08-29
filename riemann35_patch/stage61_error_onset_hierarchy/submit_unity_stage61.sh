#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

# Resolve the frozen Stage-58 result directory.  Prefer an explicit override,
# then search the common Unity github_sync worktrees.  This avoids assuming
# the historical Stage-58 checkout still has one exact directory name.
if [[ -n "${STAGE58_DIR:-}" ]]; then
  S58="$STAGE58_DIR"
else
  SEARCH_ROOT=${STAGE58_SEARCH_ROOT:-/project/pi_roohie_umass_edu/github_sync}
  S58=$(find "$SEARCH_ROOT" \
      -path '*/results/riemann35_stage58/*/stage58_stage57_anchor.npz' \
      -type f -printf '%T@ %h\n' 2>/dev/null \
      | sort -nr | head -n1 | cut -d' ' -f2- || true)
fi

[[ -n "${S58:-}" && -f "$S58/stage58_stage57_anchor.npz" ]] || {
  echo "Stage58 result directory not found under /project/pi_roohie_umass_edu/github_sync." >&2
  echo "Set STAGE58_DIR=/path/to/the/directory/containing/stage58_stage57_anchor.npz" >&2
  exit 2
}

# Require all five frozen cases before submitting anything.
CASES=(stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d)
for c in "${CASES[@]}"; do
  [[ -f "$S58/stage58_${c}.npz" ]] || {
    echo "Missing frozen Stage58 input: $S58/stage58_${c}.npz" >&2
    exit 3
  }
done

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/riemann35_stage61/error_onset_${STAMP}"
mkdir -p "$OUT/logs"
JOBS=()
for i in "${!CASES[@]}"; do
  c=${CASES[$i]}
  j=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:20:00 --mem=2G \
      -J "m61-$i" \
      -o "$OUT/logs/m61-${i}_%j.out" \
      -e "$OUT/logs/m61-${i}_%j.err" \
      --wrap="cd '$ROOT' && python riemann35_patch/stage61_error_onset_hierarchy/run_error_onset_case.py --case '$c' --stage58-dir '$S58' --output '$OUT'")
  JOBS+=("$j")
done
DEP=$(IFS=:; echo "${JOBS[*]}")
COL=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:10:00 --mem=1G \
    -J m61-collect --dependency=afterok:$DEP \
    -o "$OUT/logs/m61-collect_%j.out" \
    -e "$OUT/logs/m61-collect_%j.err" \
    --wrap="cd '$ROOT' && python riemann35_patch/stage61_error_onset_hierarchy/collect_error_onset.py '$OUT'")
echo "STAGE58_DIR=$S58"
echo "STAGE61_JOBS=${JOBS[*]}"
echo "COLLECT_JOB=$COL"
echo "RESULT_DIR=$OUT"

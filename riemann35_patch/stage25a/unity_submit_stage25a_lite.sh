#!/bin/bash
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
PYTHON="${STAGE25A_LITE_PYTHON:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python}"
STAMP="$(date +%Y%m%d-%H%M%S)"
ROOT="$REPO/results/riemann35_stage25a/lite_$STAMP"
BUNDLE="$REPO/STAGE25A_LITE_RESULTS_$STAMP.zip"
LOGS="$ROOT/logs"

test -x "$PYTHON" || { echo "ERROR: Python is not executable: $PYTHON" >&2; exit 2; }
mkdir -p "$LOGS"

EXPORTS="ALL,STAGE25A_LITE_REPO=$REPO,STAGE25A_LITE_ROOT=$ROOT,STAGE25A_LITE_PYTHON=$PYTHON,STAGE25A_LITE_BUNDLE=$BUNDLE"
ARRAY_JOB="$(sbatch --parsable \
  --job-name=fp-s25l \
  --partition=cpu \
  --array=0-2%3 \
  --cpus-per-task=4 \
  --mem=8G \
  --time=1-00:00:00 \
  --chdir="$REPO" \
  --output="$LOGS/method-%A_%a.out" \
  --error="$LOGS/method-%A_%a.err" \
  --export="$EXPORTS" \
  "$REPO/riemann35_patch/stage25a/unity_run_stage25a_lite_task.sh")"
ARRAY_JOB="${ARRAY_JOB%%;*}"

COLLECT_JOB="$(sbatch --parsable \
  --job-name=fp-s25l-pack \
  --partition=cpu \
  --dependency="afterok:$ARRAY_JOB" \
  --cpus-per-task=1 \
  --mem=4G \
  --time=01:00:00 \
  --chdir="$REPO" \
  --output="$LOGS/collect-%j.out" \
  --error="$LOGS/collect-%j.err" \
  --export="$EXPORTS" \
  "$REPO/riemann35_patch/stage25a/unity_collect_stage25a_lite.sh")"
COLLECT_JOB="${COLLECT_JOB%%;*}"

{
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  printf 'array_job=%s\n' "$ARRAY_JOB"
  printf 'collector_job=%s\n' "$COLLECT_JOB"
  printf 'result_root=%s\n' "$ROOT"
  printf 'result_bundle=%s\n' "$BUNDLE"
} > "$ROOT/submission.txt"

printf 'Submitted Stage25A-lite methods JOB=%s and collector JOB=%s\n' "$ARRAY_JOB" "$COLLECT_JOB"
printf 'Final ZIP will be: %s\n' "$BUNDLE"
squeue -j "$ARRAY_JOB,$COLLECT_JOB"

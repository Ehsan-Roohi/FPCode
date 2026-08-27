#!/bin/bash
set -euo pipefail
REPOSITORY_URL="https://github.com/Ehsan-Roohi/FPCode.git"
COMMIT="${FP_STAGE55_COMMIT:-}"
SYNC_ROOT="${FP_STAGE55_SYNC_ROOT:-/project/pi_roohie_umass_edu/github_sync}"
PYTHON_BIN="${FP_STAGE55_PYTHON:-python}"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: FP_STAGE55_COMMIT must be a pinned 40-character SHA" >&2; exit 2; }
for command_name in git sbatch "$PYTHON_BIN"; do command -v "$command_name" >/dev/null || { echo "ERROR: missing $command_name" >&2; exit 2; }; done
[[ -d "$SYNC_ROOT" ]] || { echo "ERROR: Unity sync root does not exist: $SYNC_ROOT" >&2; exit 2; }
STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN="$SYNC_ROOT/FPCode-stage55-source-$STAMP"
RESULT_ROOT="$RUN/results/riemann35_stage55/source_$STAMP"
BUNDLE="$RUN/STAGE55_CLOSURE_SOURCE_RESULTS_$STAMP.zip"
git clone --filter=blob:none --no-checkout "$REPOSITORY_URL" "$RUN"
git -C "$RUN" fetch --no-tags origin "$COMMIT"
git -C "$RUN" checkout --detach "$COMMIT"
[[ "$(git -C "$RUN" rev-parse HEAD)" == "$COMMIT" ]] || exit 2
mkdir -p "$RESULT_ROOT/logs"
EXPORTS="ALL,FP_STAGE55_RUN=$RUN,FP_STAGE55_RESULT_ROOT=$RESULT_ROOT,FP_STAGE55_BUNDLE=$BUNDLE,FP_STAGE55_PYTHON=$PYTHON_BIN"
ARRAY_RAW="$(sbatch --parsable --array=0-5%6 --job-name=fp-s55-src --output="$RESULT_ROOT/logs/method-%A_%a.out" --error="$RESULT_ROOT/logs/method-%A_%a.err" --export="$EXPORTS" "$RUN/riemann35_patch/stage55_closure_source/run_unity_stage55_method.sbatch")"
ARRAY_JOB="${ARRAY_RAW%%;*}"
COLLECT_RAW="$(sbatch --parsable --dependency="afterany:$ARRAY_JOB" --job-name=fp-s55-pack --output="$RESULT_ROOT/logs/collector-%j.out" --error="$RESULT_ROOT/logs/collector-%j.err" --export="$EXPORTS" "$RUN/riemann35_patch/stage55_closure_source/run_unity_stage55_collect.sbatch")"
COLLECT_JOB="${COLLECT_RAW%%;*}"
printf 'run=%q\ncommit=%q\nresult_root=%q\nresult_bundle=%q\narray_job_id=%q\ncollector_job_id=%q\n' "$RUN" "$COMMIT" "$RESULT_ROOT" "$BUNDLE" "$ARRAY_JOB" "$COLLECT_JOB" > "$RESULT_ROOT/submission.txt"
echo "RUN=$RUN"; echo "COMMIT=$COMMIT"; echo "RESULT_ROOT=$RESULT_ROOT"; echo "ARRAY_JOB=$ARRAY_JOB"; echo "COLLECT_JOB=$COLLECT_JOB"; echo "BUNDLE=$BUNDLE"; echo "Monitor: squeue -j $ARRAY_JOB,$COLLECT_JOB"

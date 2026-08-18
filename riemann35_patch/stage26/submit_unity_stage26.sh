#!/bin/bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/FPCode.git"
COMMIT="${FP_STAGE26_COMMIT:-}"
SYNC_ROOT="${FP_STAGE26_SYNC_ROOT:-/project/pi_roohie_umass_edu/github_sync}"
PYTHON_BIN="${FP_STAGE26_PYTHON:-python}"

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: FP_STAGE26_COMMIT must be the pinned 40-character commit SHA" >&2
    exit 2
fi
for command_name in git sbatch "$PYTHON_BIN"; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $command_name" >&2
        exit 2
    fi
done
if [[ ! -d "$SYNC_ROOT" ]]; then
    echo "ERROR: Unity sync root does not exist: $SYNC_ROOT" >&2
    exit 2
fi

STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN="$SYNC_ROOT/FPCode-stage26-fourdelta-$STAMP"
RESULT_ROOT="$RUN/results/riemann35_stage26/four_delta_$STAMP"
BUNDLE="$RUN/STAGE26_FOUR_DELTA_RESULTS_$STAMP.zip"

git clone --filter=blob:none --no-checkout "$REPOSITORY_URL" "$RUN"
git -C "$RUN" fetch --no-tags origin "$COMMIT"
git -C "$RUN" checkout --detach "$COMMIT"
if [[ "$(git -C "$RUN" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo "ERROR: checkout did not resolve to the pinned commit" >&2
    exit 2
fi
mkdir -p "$RESULT_ROOT/logs"

EXPORTS="ALL,FP_STAGE26_RUN=$RUN,FP_STAGE26_RESULT_ROOT=$RESULT_ROOT,FP_STAGE26_BUNDLE=$BUNDLE,FP_STAGE26_PYTHON=$PYTHON_BIN"
ARRAY_RAW="$(
    sbatch --parsable \
        --job-name=fp-s26-4d \
        --array=0-3%4 \
        --output="$RESULT_ROOT/logs/method-%A_%a.out" \
        --error="$RESULT_ROOT/logs/method-%A_%a.err" \
        --export="$EXPORTS" \
        "$RUN/riemann35_patch/stage26/run_unity_stage26_method.sbatch"
)"
ARRAY_JOB="${ARRAY_RAW%%;*}"
COLLECTOR_RAW="$(
    sbatch --parsable \
        --job-name=fp-s26-pack \
        --dependency="afterany:$ARRAY_JOB" \
        --output="$RESULT_ROOT/logs/collect-%j.out" \
        --error="$RESULT_ROOT/logs/collect-%j.err" \
        --export="$EXPORTS" \
        "$RUN/riemann35_patch/stage26/run_unity_stage26_collect.sbatch"
)"
COLLECTOR_JOB="${COLLECTOR_RAW%%;*}"

{
    printf 'run=%q\n' "$RUN"
    printf 'commit=%q\n' "$COMMIT"
    printf 'result_root=%q\n' "$RESULT_ROOT"
    printf 'result_bundle=%q\n' "$BUNDLE"
    printf 'array_job=%q\n' "$ARRAY_JOB"
    printf 'collector_job=%q\n' "$COLLECTOR_JOB"
} > "$RESULT_ROOT/submission.txt"

echo "RUN=$RUN"
echo "COMMIT=$COMMIT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "ARRAY=$ARRAY_JOB"
echo "COLLECTOR=$COLLECTOR_JOB"
echo "BUNDLE=$BUNDLE"
echo "Monitor: squeue -j $ARRAY_JOB,$COLLECTOR_JOB"

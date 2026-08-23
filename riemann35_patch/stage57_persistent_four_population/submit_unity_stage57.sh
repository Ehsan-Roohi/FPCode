#!/bin/bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/FPCode.git"
COMMIT="${MOMENT_STAGE57_COMMIT:-}"
SYNC_ROOT="${MOMENT_STAGE57_SYNC_ROOT:-/project/pi_roohie_umass_edu/github_sync}"
PYTHON_BIN="${MOMENT_STAGE57_PYTHON:-python}"

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: MOMENT_STAGE57_COMMIT must be the pinned 40-character commit SHA" >&2
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
RUN="$SYNC_ROOT/MomentCode-stage57-p4-$STAMP"
RESULT_ROOT="$RUN/results/riemann35_stage57/persistent4_$STAMP"
BUNDLE="$RUN/STAGE57_MOMENT_PERSISTENT4_RESULTS_$STAMP.zip"

git clone --filter=blob:none --no-checkout "$REPOSITORY_URL" "$RUN"
git -C "$RUN" fetch --no-tags origin "$COMMIT"
git -C "$RUN" checkout --detach "$COMMIT"
if [[ "$(git -C "$RUN" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo "ERROR: checkout did not resolve to the pinned commit" >&2
    exit 2
fi
mkdir -p "$RESULT_ROOT/logs"

EXPORTS="ALL,MOMENT_STAGE57_RUN=$RUN,MOMENT_STAGE57_RESULT_ROOT=$RESULT_ROOT,MOMENT_STAGE57_BUNDLE=$BUNDLE,MOMENT_STAGE57_PYTHON=$PYTHON_BIN"
ARRAY_RAW="$(
    sbatch --parsable \
        --array=0-5%6 \
        --job-name=mom-s57-p4 \
        --output="$RESULT_ROOT/logs/method-%A_%a.out" \
        --error="$RESULT_ROOT/logs/method-%A_%a.err" \
        --export="$EXPORTS" \
        "$RUN/riemann35_patch/stage57_persistent_four_population/run_unity_stage57_method.sbatch"
)"
ARRAY_JOB="${ARRAY_RAW%%;*}"
COLLECT_RAW="$(
    sbatch --parsable \
        --dependency="afterany:$ARRAY_JOB" \
        --job-name=mom-s57-pack \
        --output="$RESULT_ROOT/logs/collector-%j.out" \
        --error="$RESULT_ROOT/logs/collector-%j.err" \
        --export="$EXPORTS" \
        "$RUN/riemann35_patch/stage57_persistent_four_population/run_unity_stage57_collect.sbatch"
)"
COLLECT_JOB="${COLLECT_RAW%%;*}"

{
    printf 'run=%q\n' "$RUN"
    printf 'commit=%q\n' "$COMMIT"
    printf 'result_root=%q\n' "$RESULT_ROOT"
    printf 'result_bundle=%q\n' "$BUNDLE"
    printf 'array_job_id=%q\n' "$ARRAY_JOB"
    printf 'collector_job_id=%q\n' "$COLLECT_JOB"
} > "$RESULT_ROOT/submission.txt"

echo "RUN=$RUN"
echo "COMMIT=$COMMIT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "ARRAY_JOB=$ARRAY_JOB"
echo "COLLECT_JOB=$COLLECT_JOB"
echo "BUNDLE=$BUNDLE"
echo "Monitor: squeue -j $ARRAY_JOB,$COLLECT_JOB"


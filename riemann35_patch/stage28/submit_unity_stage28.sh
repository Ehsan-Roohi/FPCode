#!/bin/bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/FPCode.git"
COMMIT="${FP_STAGE28_COMMIT:-}"
SYNC_ROOT="${FP_STAGE28_SYNC_ROOT:-/project/pi_roohie_umass_edu/github_sync}"
PYTHON_BIN="${FP_STAGE28_PYTHON:-python}"

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: FP_STAGE28_COMMIT must be the pinned 40-character commit SHA" >&2
    exit 2
fi
for command_name in git sbatch zip "$PYTHON_BIN"; do
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
RUN="$SYNC_ROOT/FPCode-stage28-localized-$STAMP"
RESULT_ROOT="$RUN/results/riemann35_stage28/localized_$STAMP"
BUNDLE="$RUN/STAGE28_LOCALIZED_RESULTS_$STAMP.zip"

git clone --filter=blob:none --no-checkout "$REPOSITORY_URL" "$RUN"
git -C "$RUN" fetch --no-tags origin "$COMMIT"
git -C "$RUN" checkout --detach "$COMMIT"
if [[ "$(git -C "$RUN" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo "ERROR: checkout did not resolve to the pinned commit" >&2
    exit 2
fi
mkdir -p "$RESULT_ROOT/logs"

EXPORTS="ALL,FP_STAGE28_RUN=$RUN,FP_STAGE28_RESULT_ROOT=$RESULT_ROOT,FP_STAGE28_BUNDLE=$BUNDLE,FP_STAGE28_PYTHON=$PYTHON_BIN"
JOB_RAW="$(
    sbatch --parsable \
        --job-name=fp-s28-loc \
        --output="$RESULT_ROOT/logs/stage28-%j.out" \
        --error="$RESULT_ROOT/logs/stage28-%j.err" \
        --export="$EXPORTS" \
        "$RUN/riemann35_patch/stage28/run_unity_stage28.sbatch"
)"
JOB_ID="${JOB_RAW%%;*}"

{
    printf 'run=%q\n' "$RUN"
    printf 'commit=%q\n' "$COMMIT"
    printf 'result_root=%q\n' "$RESULT_ROOT"
    printf 'result_bundle=%q\n' "$BUNDLE"
    printf 'job_id=%q\n' "$JOB_ID"
} > "$RESULT_ROOT/submission.txt"

echo "RUN=$RUN"
echo "COMMIT=$COMMIT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "JOB=$JOB_ID"
echo "BUNDLE=$BUNDLE"
echo "Monitor: squeue -j $JOB_ID"

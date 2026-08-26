#!/bin/bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/FPCode.git"
COMMIT="${MOMENT_STAGE58_COMMIT:-}"
SYNC_ROOT="${MOMENT_STAGE58_SYNC_ROOT:-/project/pi_roohie_umass_edu/github_sync}"
PYTHON_BIN="${MOMENT_STAGE58_PYTHON:-python}"

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: MOMENT_STAGE58_COMMIT must be the pinned 40-character commit SHA" >&2
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
RUN="$SYNC_ROOT/MomentCode-stage58-blind-$STAMP"
RESULT_ROOT="$RUN/results/riemann35_stage58/blind_generalization_$STAMP"
BUNDLE="$RUN/STAGE58_MOMENT_BLIND_GENERALIZATION_RESULTS_$STAMP.zip"

git clone --filter=blob:none --no-checkout "$REPOSITORY_URL" "$RUN"
git -C "$RUN" fetch --no-tags origin "$COMMIT"
git -C "$RUN" checkout --detach "$COMMIT"
if [[ "$(git -C "$RUN" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo "ERROR: checkout did not resolve to the pinned commit" >&2
    exit 2
fi
mkdir -p "$RESULT_ROOT/logs"

EXPORTS="ALL,MOMENT_STAGE58_RUN=$RUN,MOMENT_STAGE58_RESULT_ROOT=$RESULT_ROOT,MOMENT_STAGE58_BUNDLE=$BUNDLE,MOMENT_STAGE58_PYTHON=$PYTHON_BIN"
ARRAY_RAW="$(
    sbatch --parsable \
        --array=0-4%5 \
        --job-name=mom-s58-blind \
        --output="$RESULT_ROOT/logs/case-%A_%a.out" \
        --error="$RESULT_ROOT/logs/case-%A_%a.err" \
        --export="$EXPORTS" \
        "$RUN/riemann35_patch/stage58_blind_generalization/run_unity_stage58_case.sbatch"
)"
ARRAY_JOB="${ARRAY_RAW%%;*}"
COLLECT_RAW="$(
    sbatch --parsable \
        --dependency="afterany:$ARRAY_JOB" \
        --job-name=mom-s58-pack \
        --output="$RESULT_ROOT/logs/collector-%j.out" \
        --error="$RESULT_ROOT/logs/collector-%j.err" \
        --export="$EXPORTS" \
        "$RUN/riemann35_patch/stage58_blind_generalization/run_unity_stage58_collect.sbatch"
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

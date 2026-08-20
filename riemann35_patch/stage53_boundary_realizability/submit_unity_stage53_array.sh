#!/bin/bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/FPCode.git"
COMMIT="${FP_STAGE53_COMMIT:-}"
SYNC_ROOT="${FP_STAGE53_SYNC_ROOT:-/project/pi_roohie_umass_edu/github_sync}"
PYTHON_BIN="${FP_STAGE53_PYTHON:-python}"

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: FP_STAGE53_COMMIT must be the pinned 40-character commit SHA" >&2
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
RUN="$SYNC_ROOT/FPCode-stage53-array-$STAMP"
RESULT_ROOT="$RUN/results/riemann35_stage53_array/boundary_$STAMP"
BUNDLE="$RUN/STAGE53_ARRAY_RESULTS_$STAMP.zip"

git clone --filter=blob:none --no-checkout "$REPOSITORY_URL" "$RUN"
git -C "$RUN" fetch --no-tags origin "$COMMIT"
git -C "$RUN" checkout --detach "$COMMIT"
if [[ "$(git -C "$RUN" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo "ERROR: checkout did not resolve to the pinned commit" >&2
    exit 2
fi
mkdir -p "$RESULT_ROOT/logs"
for index in 0 1 2 3 4; do
    mkdir -p "$RESULT_ROOT/epsilon_$index/logs"
done

EXPORTS="ALL,FP_STAGE53_RUN=$RUN,FP_STAGE53_RESULT_ROOT=$RESULT_ROOT,FP_STAGE53_BUNDLE=$BUNDLE,FP_STAGE53_PYTHON=$PYTHON_BIN"
ARRAY_RAW="$(
    sbatch --parsable \
        --array=0-4%5 \
        --job-name=fp-s53-eps \
        --output="$RESULT_ROOT/epsilon_%a/logs/stage53-%A_%a.out" \
        --error="$RESULT_ROOT/epsilon_%a/logs/stage53-%A_%a.err" \
        --export="$EXPORTS" \
        "$RUN/riemann35_patch/stage53_boundary_realizability/run_unity_stage53_array.sbatch"
)"
ARRAY_JOB="${ARRAY_RAW%%;*}"
COLLECT_RAW="$(
    sbatch --parsable \
        --dependency="afterany:$ARRAY_JOB" \
        --job-name=fp-s53-collect \
        --output="$RESULT_ROOT/logs/collector-%j.out" \
        --error="$RESULT_ROOT/logs/collector-%j.err" \
        --export="$EXPORTS" \
        "$RUN/riemann35_patch/stage53_boundary_realizability/collect_unity_stage53_array.sbatch"
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

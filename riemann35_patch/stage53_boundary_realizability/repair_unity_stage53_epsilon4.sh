#!/bin/bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/FPCode.git"
COMMIT="${FP_STAGE53_COMMIT:-}"
RESULT_ROOT="${FP_STAGE53_RESULT_ROOT:-}"
SYNC_ROOT="${FP_STAGE53_SYNC_ROOT:-/project/pi_roohie_umass_edu/github_sync}"
PYTHON_BIN="${FP_STAGE53_PYTHON:-python}"

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: FP_STAGE53_COMMIT must be the pinned repair commit" >&2
    exit 2
fi
if [[ -z "$RESULT_ROOT" || ! -f "$RESULT_ROOT/submission.txt" ]]; then
    echo "ERROR: FP_STAGE53_RESULT_ROOT must contain the original submission.txt" >&2
    exit 2
fi
for command_name in git sbatch zip "$PYTHON_BIN"; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $command_name" >&2
        exit 2
    fi
done
for index in 0 1 2 3; do
    if [[ ! -s "$RESULT_ROOT/epsilon_$index/stage53_summary.json" ]]; then
        echo "ERROR: successful source result epsilon_$index is missing" >&2
        exit 2
    fi
done

# shellcheck disable=SC1090
source "$RESULT_ROOT/submission.txt"
ORIGINAL_COMMIT="$commit"
BUNDLE="$result_bundle"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
REPAIR_RUN="$SYNC_ROOT/FPCode-stage53-epsilon4-repair-$STAMP"

git clone --filter=blob:none --no-checkout "$REPOSITORY_URL" "$REPAIR_RUN"
git -C "$REPAIR_RUN" fetch --no-tags origin "$COMMIT"
git -C "$REPAIR_RUN" checkout --detach "$COMMIT"
if [[ "$(git -C "$REPAIR_RUN" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo "ERROR: repair checkout did not resolve to the pinned commit" >&2
    exit 2
fi

mkdir -p "$RESULT_ROOT/epsilon_4/logs" "$RESULT_ROOT/logs"
EXPORTS="ALL,FP_STAGE53_RUN=$REPAIR_RUN,FP_STAGE53_RESULT_ROOT=$RESULT_ROOT,FP_STAGE53_BUNDLE=$BUNDLE,FP_STAGE53_PYTHON=$PYTHON_BIN"
TASK_RAW="$(
    sbatch --parsable \
        --array=4 \
        --job-name=fp-s53-e4-repair \
        --output="$RESULT_ROOT/epsilon_4/logs/stage53-repair-%A_%a.out" \
        --error="$RESULT_ROOT/epsilon_4/logs/stage53-repair-%A_%a.err" \
        --export="$EXPORTS" \
        "$REPAIR_RUN/riemann35_patch/stage53_boundary_realizability/run_unity_stage53_array.sbatch"
)"
TASK_JOB="${TASK_RAW%%;*}"
COLLECT_RAW="$(
    sbatch --parsable \
        --dependency="afterany:$TASK_JOB" \
        --job-name=fp-s53-recollect \
        --output="$RESULT_ROOT/logs/collector-repair-%j.out" \
        --error="$RESULT_ROOT/logs/collector-repair-%j.err" \
        --export="$EXPORTS" \
        "$REPAIR_RUN/riemann35_patch/stage53_boundary_realizability/collect_unity_stage53_array.sbatch"
)"
COLLECT_JOB="${COLLECT_RAW%%;*}"

{
    printf 'original_commit=%q\n' "$ORIGINAL_COMMIT"
    printf 'repair_commit=%q\n' "$COMMIT"
    printf 'repair_run=%q\n' "$REPAIR_RUN"
    printf 'result_root=%q\n' "$RESULT_ROOT"
    printf 'result_bundle=%q\n' "$BUNDLE"
    printf 'epsilon4_job_id=%q\n' "$TASK_JOB"
    printf 'collector_job_id=%q\n' "$COLLECT_JOB"
} > "$RESULT_ROOT/repair_submission.txt"

echo "ORIGINAL_COMMIT=$ORIGINAL_COMMIT"
echo "REPAIR_COMMIT=$COMMIT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "EPSILON4_JOB=$TASK_JOB"
echo "COLLECT_JOB=$COLLECT_JOB"
echo "BUNDLE=$BUNDLE"
echo "Monitor: squeue -j $TASK_JOB,$COLLECT_JOB"

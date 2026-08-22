#!/bin/bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/FPCode.git"
COMMIT="${FP_STAGE56_COMMIT:-}"
SYNC_ROOT="${FP_STAGE56_SYNC_ROOT:-/project/pi_roohie_umass_edu/github_sync}"
PYTHON_BIN="${FP_STAGE56_PYTHON:-python}"
REFERENCE_ROOT="${FP_STAGE56_REFERENCE_ROOT:-/project/pi_roohie_umass_edu/github_sync/FPCode-stage55-source-20260822-152214/results/riemann35_stage55/source_audit_20260822-152214}"

if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: FP_STAGE56_COMMIT must be the pinned 40-character commit SHA" >&2
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
for reference_name in stage55_qmc_reference.npz stage55_qmc_reference_summary.json; do
    if [[ ! -f "$REFERENCE_ROOT/$reference_name" ]]; then
        echo "ERROR: frozen Stage-55 reference is missing: $REFERENCE_ROOT/$reference_name" >&2
        exit 2
    fi
done

STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN="$SYNC_ROOT/FPCode-stage56-time-consistent-$STAMP"
RESULT_ROOT="$RUN/results/riemann35_stage56/time_consistent_$STAMP"
BUNDLE="$RUN/STAGE56_TIME_CONSISTENT_RESULTS_$STAMP.zip"

git clone --filter=blob:none --no-checkout "$REPOSITORY_URL" "$RUN"
git -C "$RUN" fetch --no-tags origin "$COMMIT"
git -C "$RUN" checkout --detach "$COMMIT"
if [[ "$(git -C "$RUN" rev-parse HEAD)" != "$COMMIT" ]]; then
    echo "ERROR: checkout did not resolve to the pinned commit" >&2
    exit 2
fi
mkdir -p "$RESULT_ROOT/logs"
cd "$RUN"
"$PYTHON_BIN" -m riemann35_patch.stage56_time_consistent_closure.test_time_consistent

EXPORTS="ALL,FP_STAGE56_RUN=$RUN,FP_STAGE56_RESULT_ROOT=$RESULT_ROOT,FP_STAGE56_REFERENCE_ROOT=$REFERENCE_ROOT,FP_STAGE56_BUNDLE=$BUNDLE,FP_STAGE56_PYTHON=$PYTHON_BIN"
ARRAY_RAW="$(
    sbatch --parsable \
        --array=0-5%6 \
        --job-name=fp-s56-tc \
        --output="$RESULT_ROOT/logs/method-%A_%a.out" \
        --error="$RESULT_ROOT/logs/method-%A_%a.err" \
        --export="$EXPORTS" \
        "$RUN/riemann35_patch/stage56_time_consistent_closure/run_unity_stage56_method.sbatch"
)"
ARRAY_JOB="${ARRAY_RAW%%;*}"
COLLECT_RAW="$(
    sbatch --parsable \
        --dependency="afterany:$ARRAY_JOB" \
        --job-name=fp-s56-pack \
        --output="$RESULT_ROOT/logs/collector-%j.out" \
        --error="$RESULT_ROOT/logs/collector-%j.err" \
        --export="$EXPORTS" \
        "$RUN/riemann35_patch/stage56_time_consistent_closure/run_unity_stage56_collect.sbatch"
)"
COLLECT_JOB="${COLLECT_RAW%%;*}"

{
    printf 'run=%q\n' "$RUN"
    printf 'commit=%q\n' "$COMMIT"
    printf 'reference_root=%q\n' "$REFERENCE_ROOT"
    printf 'result_root=%q\n' "$RESULT_ROOT"
    printf 'result_bundle=%q\n' "$BUNDLE"
    printf 'array_job_id=%q\n' "$ARRAY_JOB"
    printf 'collector_job_id=%q\n' "$COLLECT_JOB"
} > "$RESULT_ROOT/submission.txt"

echo "RUN=$RUN"
echo "REFERENCE_ROOT=$REFERENCE_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "ARRAY_JOB=$ARRAY_JOB"
echo "COLLECT_JOB=$COLLECT_JOB"
echo "BUNDLE=$BUNDLE"
echo "Monitor: squeue -j $ARRAY_JOB,$COLLECT_JOB"

#!/usr/bin/env bash
# Submit the V4 OOD array and its root-ZIP collector.

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(git -C "$module_dir" rev-parse --show-toplevel)"
cd "$repo_root"

array_job="$(sbatch --parsable "$script_dir/run_stage2_ood_v4_array.sbatch")"
array_job="${array_job%%;*}"
collector_job="$(
    sbatch --parsable \
        --dependency="afterany:${array_job}" \
        --export="ALL,FP_V4_ARRAY_JOB_ID=${array_job}" \
        --job-name="fp-v4-z${array_job}" \
        "$script_dir/collect_stage2_ood_v4.sbatch"
)"
collector_job="${collector_job%%;*}"

echo "V4_ARRAY_JOB_ID=$array_job"
echo "V4_COLLECTOR_JOB_ID=$collector_job"
echo "V4_COMPLETE_ZIP=$repo_root/FP_PINN_STAGE2_V4_OOD_JOB${array_job}_COMPLETE.zip"

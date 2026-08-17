#!/usr/bin/env bash
#SBATCH --job-name=fp-v4-collect
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=FP_PINN_STAGE2_V4_COLLECT_%x_%j.out
#SBATCH --error=FP_PINN_STAGE2_V4_COLLECT_%x_%j.err

set -euo pipefail

if [[ -z "${FP_V4_ARRAY_JOB_ID:-}" ]]; then
    echo "FP_V4_ARRAY_JOB_ID is required" >&2
    exit 4
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(git -C "$module_dir" rev-parse --show-toplevel)"
env_dir="${FP_V4_ENV_DIR:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu}"
source "$env_dir/bin/activate"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fp-pinn-v4-collect-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
run_root="$repo_root/outputs/stage2-v4-${FP_V4_ARRAY_JOB_ID}"
aggregate="$run_root/aggregate"
archive="$repo_root/FP_PINN_STAGE2_V4_OOD_JOB${FP_V4_ARRAY_JOB_ID}_COMPLETE.zip"
mkdir -p "$aggregate"

summary_exit=0
python "$module_dir/summarize_stage2_ood_v4.py" \
    --run-root "$run_root" \
    --output-dir "$aggregate" || summary_exit=$?
printf '%s\n' "$summary_exit" > "$aggregate/summary_exit_code.txt"

python "$module_dir/package_stage2_ood_v4.py" \
    --run-root "$run_root" \
    --archive "$archive" \
    --repo-root "$repo_root" \
    --array-job-id "$FP_V4_ARRAY_JOB_ID" \
    --collector-job-id "$SLURM_JOB_ID"

echo "V4_COMPLETE_ZIP=$archive"

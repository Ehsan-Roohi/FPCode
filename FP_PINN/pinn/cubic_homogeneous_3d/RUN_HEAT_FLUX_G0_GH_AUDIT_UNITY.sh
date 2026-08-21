#!/usr/bin/env bash
set -euo pipefail

FP_TEST="${FP_TEST:-/project/pi_roohie_umass_edu/github_sync/FPCode-pinn-g0-refine}"
DEFAULT_RESUME_ARCHIVE="/project/pi_roohie_umass_edu/github_sync/FPCode-pinn-g0/FP_PINN_STAGE2_JOB63178434_HEAT_FLUX_COMPLETE.zip"
export FP_RESUME_ARCHIVE="${FP_RESUME_ARCHIVE:-$DEFAULT_RESUME_ARCHIVE}"
export FP_GPU_CONSTRAINT="${FP_GPU_CONSTRAINT:-sm_75&vram12}"
export FP_MIN_GPU_COMPUTE_CAPABILITY="${FP_MIN_GPU_COMPUTE_CAPABILITY:-7.5}"

if [[ ! -d "$FP_TEST/.git" ]]; then
    echo "ERROR: audit checkout not found: $FP_TEST" >&2
    exit 2
fi
if [[ ! -f "$FP_RESUME_ARCHIVE" ]]; then
    echo "ERROR: successful G0 archive not found: $FP_RESUME_ARCHIVE" >&2
    exit 2
fi

cd "$FP_TEST"
mkdir -p slurm_logs

sbatch \
  --array=0-3%4 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --constraint="$FP_GPU_CONSTRAINT" \
  --exclude="${FP_EXCLUDE_NODES:-gypsum-gpu001,gypsum-gpu011,gypsum-gpu012,gypsum-gpu013,gypsum-gpu015}" \
  --output="$FP_TEST/slurm_logs/fp-g0-gh-%A_%a.out" \
  --error="$FP_TEST/slurm_logs/fp-g0-gh-%A_%a.err" \
  FP_PINN/pinn/cubic_homogeneous_3d/slurm/run_heat_flux_g0_gh_audit.sbatch

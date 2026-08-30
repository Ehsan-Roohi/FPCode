#!/usr/bin/env bash
# Submit the stress-relaxation G2 array on Unity with one command:
#
#   bash FP_PINN/pinn/cubic_homogeneous_3d/RUN_STRESS_G2_UNITY.sh
#
# Environment overrides (all optional):
#   FP_TEST              checkout to run (default: the G2 sync checkout below)
#   FP_EXPECTED_COMMIT   abort unless `git rev-parse HEAD` starts with this
#   FP_GPU_CONSTRAINT    Slurm --constraint (default sm_75&vram12: no K80/P100-class GPUs)
#   FP_EXCLUDE_NODES     nodes known to fail the TF 2.21 / CUDA build
#   FP_ARRAY             array spec (default 0-3%4: three seeds + no-mode ablation)
#   FP_G2_EPOCHS etc.    forwarded to slurm/run_stress_g2_array.sbatch
set -euo pipefail

FP_TEST="${FP_TEST:-/project/pi_roohie_umass_edu/github_sync/FPCode-pinn-g2}"
export FP_GPU_CONSTRAINT="${FP_GPU_CONSTRAINT:-sm_75&vram12}"
export FP_MIN_GPU_COMPUTE_CAPABILITY="${FP_MIN_GPU_COMPUTE_CAPABILITY:-7.5}"
FP_ARRAY="${FP_ARRAY:-0-3%4}"

if ! git -C "$FP_TEST" rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: G2 checkout not found: $FP_TEST" >&2
    echo "Clone the G2 branch there, or export FP_TEST=/abs/path." >&2
    exit 2
fi
if [[ ! -f "$FP_TEST/FP_PINN/pinn/cubic_homogeneous_3d/g2/train_g2.py" ]]; then
    echo "ERROR: $FP_TEST does not contain FP_PINN/pinn/cubic_homogeneous_3d/g2/train_g2.py" >&2
    exit 2
fi
DIRTY_STAGE="$(git -C "$FP_TEST" status --porcelain -- FP_PINN/pinn/cubic_homogeneous_3d)"
if [[ -n "$DIRTY_STAGE" && "${FP_ALLOW_DIRTY:-0}" != "1" ]]; then
    echo "ERROR: uncommitted changes exist under FP_PINN/pinn/cubic_homogeneous_3d." >&2
    echo "Commit/stash them, or set FP_ALLOW_DIRTY=1 only for an intentionally non-reproducible run." >&2
    exit 2
fi
HEAD_COMMIT="$(git -C "$FP_TEST" rev-parse HEAD)"
if [[ -n "${FP_EXPECTED_COMMIT:-}" && "$HEAD_COMMIT" != "$FP_EXPECTED_COMMIT"* ]]; then
    echo "ERROR: $FP_TEST is at $HEAD_COMMIT, expected $FP_EXPECTED_COMMIT" >&2
    exit 2
fi
export FP_EXPECTED_COMMIT="${FP_EXPECTED_COMMIT:-$HEAD_COMMIT}"
cd "$FP_TEST"
mkdir -p slurm_logs
echo "Checkout: $FP_TEST @ $HEAD_COMMIT"
echo "Array: $FP_ARRAY   GPU constraint: $FP_GPU_CONSTRAINT"

ARRAY_JOB="$(sbatch --parsable \
  --array="$FP_ARRAY" \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --constraint="$FP_GPU_CONSTRAINT" \
  --exclude="${FP_EXCLUDE_NODES:-gypsum-gpu001,gypsum-gpu011,gypsum-gpu012,gypsum-gpu013,gypsum-gpu015}" \
  --output="$FP_TEST/slurm_logs/fp-g2-%A_%a.out" \
  --error="$FP_TEST/slurm_logs/fp-g2-%A_%a.err" \
  FP_PINN/pinn/cubic_homogeneous_3d/slurm/run_stress_g2_array.sbatch)"
ARRAY_JOB="${ARRAY_JOB%%;*}"
echo "Submitted G2 array job $ARRAY_JOB"

# Seed-agreement aggregation after every task has finished (CPU only).  Use a
# real Bash batch script: Slurm's --wrap uses /bin/sh on Unity, where the
# environment-modules `module` shell function is not defined.
AGG_JOB="$(sbatch --parsable \
  --job-name=fp-g2-agg \
  --dependency="afterany:$ARRAY_JOB" \
  --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:20:00 \
  --output="$FP_TEST/slurm_logs/fp-g2-agg-%j.out" \
  --error="$FP_TEST/slurm_logs/fp-g2-agg-%j.err" \
  --export="ALL,FP_TEST=$FP_TEST,FP_G2_ARRAY_JOB=$ARRAY_JOB,FP_EXPECTED_COMMIT=$FP_EXPECTED_COMMIT,FP_CONDA_ENV=${FP_CONDA_ENV:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu}" \
  FP_PINN/pinn/cubic_homogeneous_3d/slurm/aggregate_stress_g2.sbatch)"
echo "Submitted aggregation job $AGG_JOB (after array $ARRAY_JOB)"
echo "Per-task archives: $FP_TEST/FP_PINN_G2_JOB${ARRAY_JOB}_STRESS_<VARIANT>_COMPLETE.zip (+ .sha256)"
echo "Overall verdict:   $FP_TEST/G2_SEED_SUMMARY.md"

#!/usr/bin/env bash
# Submit the heat-flux G1 array on Unity with one command:
#
#   bash FP_PINN/pinn/cubic_homogeneous_3d/RUN_HEAT_FLUX_G1_UNITY.sh
#
# Environment overrides (all optional):
#   FP_TEST              checkout to run (default: the G1 sync checkout below)
#   FP_EXPECTED_COMMIT   abort unless `git rev-parse HEAD` starts with this
#   FP_G0_BASE_WEIGHTS   G0 epoch-12500 DensityModel weights for the warm-start task
#   FP_GPU_CONSTRAINT    Slurm --constraint (default sm_75&vram12: no K80/P100-class GPUs)
#   FP_EXCLUDE_NODES     nodes known to fail the TF 2.21 / CUDA build
#   FP_ARRAY             array spec (default 0-3%4: three seeds + warm start)
#   FP_G1_EPOCHS etc.    forwarded to slurm/run_heat_flux_g1_array.sbatch
set -euo pipefail

FP_TEST="${FP_TEST:-/project/pi_roohie_umass_edu/github_sync/FPCode-pinn-g1}"
export FP_GPU_CONSTRAINT="${FP_GPU_CONSTRAINT:-sm_75&vram12}"
export FP_MIN_GPU_COMPUTE_CAPABILITY="${FP_MIN_GPU_COMPUTE_CAPABILITY:-7.5}"
FP_G0_ARCHIVE="${FP_G0_ARCHIVE:-/project/pi_roohie_umass_edu/github_sync/FPCode-pinn-g0/FP_PINN_STAGE2_JOB63178434_HEAT_FLUX_COMPLETE.zip}"
export FP_G0_BASE_WEIGHTS="${FP_G0_BASE_WEIGHTS:-$FP_TEST/g0_warm_start/epoch-012500.weights.h5}"
FP_ARRAY="${FP_ARRAY:-0-3%4}"

if ! git -C "$FP_TEST" rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: G1 checkout not found: $FP_TEST" >&2
    echo "Clone the branch that contains FP_PINN/pinn/cubic_homogeneous_3d/g1 there, or export FP_TEST=/abs/path." >&2
    exit 2
fi
if [[ ! -f "$FP_TEST/FP_PINN/pinn/cubic_homogeneous_3d/g1/train_g1.py" ]]; then
    echo "ERROR: $FP_TEST does not contain FP_PINN/pinn/cubic_homogeneous_3d/g1/train_g1.py" >&2
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
# Warm-start weights for task 3: extract the G0 epoch-12500 checkpoint from the
# successful G0 archive if it is not already present.
if [[ ! -f "$FP_G0_BASE_WEIGHTS" && -f "$FP_G0_ARCHIVE" ]]; then
    mkdir -p "$(dirname "$FP_G0_BASE_WEIGHTS")"
    WARM_MEMBER="$(unzip -Z1 "$FP_G0_ARCHIVE" 2>/dev/null | sed -n '/\(^\|\/\)checkpoints_h5\/epoch-012500\.weights\.h5$/p' | head -n 1)"
    if [[ -n "$WARM_MEMBER" ]] \
        && unzip -p "$FP_G0_ARCHIVE" "$WARM_MEMBER" > "$FP_G0_BASE_WEIGHTS.tmp" 2>/dev/null \
        && [[ -s "$FP_G0_BASE_WEIGHTS.tmp" ]]; then
        mv "$FP_G0_BASE_WEIGHTS.tmp" "$FP_G0_BASE_WEIGHTS"
        echo "Extracted G0 warm-start weights to $FP_G0_BASE_WEIGHTS"
    else
        rm -f "$FP_G0_BASE_WEIGHTS.tmp"
    fi
fi

array_has_index() {
    local target="$1" specification="${2%%%*}" item start end step value
    local -a items
    IFS=',' read -r -a items <<< "$specification"
    for item in "${items[@]}"; do
        if [[ "$item" =~ ^([0-9]+)-([0-9]+)(:([0-9]+))?$ ]]; then
            start="${BASH_REMATCH[1]}"; end="${BASH_REMATCH[2]}"; step="${BASH_REMATCH[4]:-1}"
            for ((value=start; value<=end; value+=step)); do
                (( value == target )) && return 0
            done
        elif [[ "$item" =~ ^[0-9]+$ ]] && (( item == target )); then
            return 0
        fi
    done
    return 1
}

if array_has_index 3 "$FP_ARRAY" && [[ ! -f "$FP_G0_BASE_WEIGHTS" ]]; then
    echo "WARNING: warm-start weights not found at $FP_G0_BASE_WEIGHTS; submitting the three from-scratch seeds only (array 0-2)." >&2
    FP_ARRAY="0-2%3"
fi

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
  --output="$FP_TEST/slurm_logs/fp-g1-%A_%a.out" \
  --error="$FP_TEST/slurm_logs/fp-g1-%A_%a.err" \
  FP_PINN/pinn/cubic_homogeneous_3d/slurm/run_heat_flux_g1_array.sbatch)"
ARRAY_JOB="${ARRAY_JOB%%;*}"
echo "Submitted G1 array job $ARRAY_JOB"

# Seed-agreement aggregation after every task has finished (CPU only).
AGG_JOB="$(sbatch --parsable \
  --job-name=fp-g1-agg \
  --dependency="afterany:$ARRAY_JOB" \
  --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:20:00 \
  --output="$FP_TEST/slurm_logs/fp-g1-agg-%j.out" \
  --error="$FP_TEST/slurm_logs/fp-g1-agg-%j.err" \
  --wrap="cd '$FP_TEST/FP_PINN/pinn/cubic_homogeneous_3d' && module load conda/latest && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '${FP_CONDA_ENV:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu}' && python g1/aggregate_g1_seeds.py --run-root outputs/g1-$ARRAY_JOB && cp outputs/g1-$ARRAY_JOB/G1_SEED_SUMMARY.* '$FP_TEST/'")"
echo "Submitted aggregation job $AGG_JOB (after array $ARRAY_JOB)"
echo "Per-task archives: $FP_TEST/FP_PINN_G1_JOB${ARRAY_JOB}_HEAT_FLUX_<VARIANT>_COMPLETE.zip (+ .sha256)"
echo "Overall verdict:   $FP_TEST/G1_SEED_SUMMARY.md"

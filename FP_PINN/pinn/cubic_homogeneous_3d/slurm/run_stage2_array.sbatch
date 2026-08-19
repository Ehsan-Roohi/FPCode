#!/usr/bin/env bash
#SBATCH --job-name=fp-pinn-s2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=fp-pinn-stage2-%A_%a.out
#SBATCH --error=fp-pinn-stage2-%A_%a.err

set -euo pipefail

CASES=(equilibrium stress heat_flux)
CASE_NAME="${FP_CASE_NAME:-${CASES[$SLURM_ARRAY_TASK_ID]}}"
RUN_SEED="${FP_SEED:-$((20260808 + SLURM_ARRAY_TASK_ID))}"

# Slurm executes a spool copy of this file, so BASH_SOURCE points into
# /var/spool/slurm rather than the submitted repository. Resolve the checkout
# from the submit directory, with explicit overrides for nonstandard launches.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="${FP_REPO_ROOT:-$SUBMIT_DIR}"
STAGE2_DIR="${FP_STAGE2_DIR:-$REPO_ROOT/FP_PINN/pinn/cubic_homogeneous_3d}"

if [[ ! -f "$STAGE2_DIR/train_stage2.py" ]]; then
    echo "ERROR: Stage-2 checkout not found: $STAGE2_DIR" >&2
    echo "Submit from the FPCode repository root or export FP_REPO_ROOT=/absolute/path/to/FPCode." >&2
    exit 2
fi
if [[ ! -f "$STAGE2_DIR/tests/__init__.py" ]]; then
    echo "ERROR: Importable Stage-2 tests are missing: $STAGE2_DIR/tests/__init__.py" >&2
    exit 2
fi

cd "$STAGE2_DIR"
echo "Slurm submit directory: $SUBMIT_DIR"
echo "Repository root: $REPO_ROOT"
echo "Stage-2 directory: $STAGE2_DIR"

RUN_ROOT="${FP_OUTPUT_ROOT:-outputs/stage2-${SLURM_ARRAY_JOB_ID}}"
CASE_OUTPUT="$RUN_ROOT/$CASE_NAME"
REFERENCE_OUTPUT="$CASE_OUTPUT/reference_particle"
mkdir -p "$REFERENCE_OUTPUT"

package_on_exit() {
    local run_status=$?
    local package_status=0
    trap - EXIT
    if [[ "${FP_PACKAGE_RESULTS:-1}" == "1" ]]; then
        local package_tag="${FP_PACKAGE_TAG:-${CASE_NAME^^}}"
        local archive="$REPO_ROOT/FP_PINN_STAGE2_JOB${SLURM_ARRAY_JOB_ID}_${package_tag}_COMPLETE.zip"
        local job_record
        local log_key
        local log_path
        local -a log_args=()
        job_record="$(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null || true)"
        for log_key in StdOut StdErr; do
            log_path="$(sed -n "s/.* ${log_key}=\([^ ]*\).*/\1/p" <<<"$job_record")"
            if [[ -n "$log_path" && -f "$log_path" ]]; then
                log_args+=(--log "$log_path")
            fi
        done
        python package_stage2.py \
            --case-output "$CASE_OUTPUT" \
            --archive "$archive" \
            --repo-root "$REPO_ROOT" \
            --job-id "$SLURM_JOB_ID" \
            --array-job-id "$SLURM_ARRAY_JOB_ID" \
            --task-id "$SLURM_ARRAY_TASK_ID" \
            --case "$CASE_NAME" \
            "${log_args[@]}" || package_status=$?
        if (( run_status == 0 && package_status != 0 )); then
            run_status=$package_status
        fi
    fi
    exit "$run_status"
}
trap package_on_exit EXIT

module load conda/latest

UNITY_CONDA_ENV="${FP_CONDA_ENV:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$UNITY_CONDA_ENV"

# TensorFlow 2.21 installed by pip needs the bundled NVIDIA shared libraries.
TF_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
for NVIDIA_LIB in "$TF_SITE"/nvidia/*/lib; do
    if [[ -d "$NVIDIA_LIB" ]]; then
        export LD_LIBRARY_PATH="$NVIDIA_LIB:${LD_LIBRARY_PATH:-}"
    fi
done

echo "Activated environment: $CONDA_PREFIX"
which python
python --version
nvidia-smi
python - <<'PY'
import os

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("Visible GPUs:", gpus)
if not gpus:
    raise SystemExit("No TensorFlow GPU is visible inside this Slurm allocation")

details = tf.config.experimental.get_device_details(gpus[0])
capability = tuple(int(value) for value in details.get("compute_capability", (0, 0)))
minimum_text = os.environ.get("FP_MIN_GPU_COMPUTE_CAPABILITY", "7.5")
try:
    minimum = tuple(int(value) for value in minimum_text.split("."))
except ValueError as error:
    raise SystemExit(
        f"Invalid FP_MIN_GPU_COMPUTE_CAPABILITY={minimum_text!r}; expected MAJOR.MINOR"
    ) from error
if len(minimum) != 2:
    raise SystemExit(
        f"Invalid FP_MIN_GPU_COMPUTE_CAPABILITY={minimum_text!r}; expected MAJOR.MINOR"
    )
print("GPU compute capability:", ".".join(map(str, capability)))
print("Minimum required compute capability:", minimum_text)
if capability < minimum:
    raise SystemExit(
        "Allocated GPU is incompatible with this TensorFlow/CUDA build: "
        f"compute capability {capability[0]}.{capability[1]} < {minimum_text}. "
        "Submit with a Unity constraint such as --constraint=sm_75&vram12."
    )
PY

python -m unittest discover -s "$STAGE2_DIR/tests" -t "$STAGE2_DIR" -v

python -u reference_particle.py \
    --case "$CASE_NAME" \
    --output-dir "$REFERENCE_OUTPUT" \
    --particles "${FP_REFERENCE_PARTICLES:-250000}" \
    --dt "${FP_REFERENCE_DT:-0.005}" \
    --tmax "${FP_TMAX:-1.0}" \
    --save-every "${FP_REFERENCE_SAVE_EVERY:-10}" \
    --seed "$RUN_SEED"

if [[ -n "${FP_RESUME_ARCHIVE:-}" && -n "${FP_RESUME_WEIGHTS:-}" ]]; then
    echo "ERROR: set only one of FP_RESUME_ARCHIVE or FP_RESUME_WEIGHTS" >&2
    exit 2
fi
if [[ -n "${FP_RESUME_ARCHIVE:-}" ]]; then
    python -u prepare_stage2_resume.py \
        --archive "$FP_RESUME_ARCHIVE" \
        --case-output "$CASE_OUTPUT"
    FP_RESUME_WEIGHTS="$CASE_OUTPUT/resume_input.weights.h5"
fi

TRAIN_ARGS=(
    --case "$CASE_NAME"
    --output-dir "$CASE_OUTPUT"
    --reference "$REFERENCE_OUTPUT/reference.npz"
    --epochs "${FP_STAGE2_EPOCHS:-30000}"
    --n-time-batch "${FP_N_TIME_BATCH:-8}"
    --n-velocity-per-time "${FP_N_VELOCITY_PER_TIME:-1024}"
    --width "${FP_STAGE2_WIDTH:-128}"
    --depth "${FP_STAGE2_DEPTH:-5}"
    --learning-rate "${FP_STAGE2_LEARNING_RATE:-5.0e-4}"
    --lr-decay-steps "${FP_STAGE2_LR_DECAY_STEPS:-10000}"
    --lr-decay-rate "${FP_STAGE2_LR_DECAY_RATE:-0.3}"
    --pde-weight "${FP_PDE_WEIGHT:-1.0}"
    --mass-weight "${FP_MASS_WEIGHT:-30.0}"
    --momentum-weight "${FP_MOMENTUM_WEIGHT:-20.0}"
    --energy-weight "${FP_ENERGY_WEIGHT:-20.0}"
    --heat-flux-weight "${FP_HEAT_FLUX_WEIGHT:-10.0}"
    --heat-flux-rate-weight "${FP_HEAT_FLUX_RATE_WEIGHT:-10.0}"
    --heat-flux-scale "${FP_HEAT_FLUX_SCALE:-0.25}"
    --heat-flux-rate-step "${FP_HEAT_FLUX_RATE_STEP:-0.01}"
    --tail-fraction "${FP_TAIL_FRACTION:-0.20}"
    --tail-variance "${FP_TAIL_VARIANCE:-4.0}"
    --evaluation-samples "${FP_EVALUATION_SAMPLES:-65536}"
    --checkpoint-every "${FP_CHECKPOINT_EVERY:-2500}"
    --quadrature-panels "${FP_QUADRATURE_PANELS:-1}"
    --seed "$RUN_SEED"
)

if [[ -n "${FP_RESUME_WEIGHTS:-}" ]]; then
    TRAIN_ARGS+=(--resume-weights "$FP_RESUME_WEIGHTS")
fi
if [[ "${FP_EVALUATE_ONLY:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--evaluate-only)
fi
if [[ "${FP_STRICT_GATE:-0}" == "1" && ! ( "$CASE_NAME" == "heat_flux" && "${FP_CHECKPOINT_SWEEP:-1}" == "1" ) ]]; then
    TRAIN_ARGS+=(--strict-gate)
fi
if [[ "${FP_FIXED_VELOCITY_QUADRATURE:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--fixed-velocity-quadrature)
fi
if [[ "${FP_AXISYMMETRIC_HEAT_FLUX:-1}" == "0" ]]; then
    TRAIN_ARGS+=(--no-axisymmetric-heat-flux)
fi
if [[ "${FP_ANTITHETIC_HEAT_FLUX_QUADRATURE:-1}" == "0" ]]; then
    TRAIN_ARGS+=(--no-antithetic-heat-flux-quadrature)
fi
if [[ "${FP_STOP_GRADIENT_CLOSURE:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--stop-gradient-closure)
fi

python -u train_stage2.py "${TRAIN_ARGS[@]}"

if [[ "$CASE_NAME" == "heat_flux" && "${FP_CHECKPOINT_SWEEP:-1}" == "1" ]]; then
    SWEEP_ARGS=(
        --case-output "$CASE_OUTPUT"
        --reference "$REFERENCE_OUTPUT/reference.npz"
    )
    if [[ "${FP_STRICT_GATE:-0}" == "1" ]]; then
        SWEEP_ARGS+=(--strict-gate)
    fi
    python -u evaluate_stage2_checkpoints.py "${SWEEP_ARGS[@]}"
fi

echo "Stage-2 case completed: $CASE_NAME"
echo "Artifacts: $STAGE2_DIR/$CASE_OUTPUT"

#!/usr/bin/env bash
#SBATCH --job-name=fp-g2-stress
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

# G2 axisymmetric stress stage (structure-preserving PINN, deterministic quadrature).
#
# Array tasks 0-2: three independent seeds from scratch (seed-agreement gate).
# Array task  3  : no-explicit-stress-mode ablation (diagnostic only).
#
# Submit with RUN_STRESS_G2_UNITY.sh (single line), which pins the GPU
# constraint, the excluded nodes, the checkout and the expected commit.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="${FP_REPO_ROOT:-$SUBMIT_DIR}"
STAGE2_DIR="${FP_STAGE2_DIR:-$REPO_ROOT/FP_PINN/pinn/cubic_homogeneous_3d}"
G2_DIR="$STAGE2_DIR/g2"

if [[ ! -f "$G2_DIR/train_g2.py" ]]; then
    echo "ERROR: G2 checkout not found: $G2_DIR" >&2
    exit 2
fi

SEEDS=(20260911 20260912 20260913 20260911)
VARIANTS=(stress_s1 stress_s2 stress_s3 no_stress_mode)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
RUN_SEED="${FP_SEED:-${SEEDS[$TASK]}}"
VARIANT="${VARIANTS[$TASK]}"

cd "$STAGE2_DIR"
echo "Slurm submit directory: $SUBMIT_DIR"
echo "Repository root: $REPO_ROOT"
echo "Stage-2 directory: $STAGE2_DIR"
echo "Task $TASK variant=$VARIANT seed=$RUN_SEED"

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Checkout commit: $ACTUAL_COMMIT"
if [[ -n "${FP_EXPECTED_COMMIT:-}" && "$ACTUAL_COMMIT" != "$FP_EXPECTED_COMMIT"* ]]; then
    echo "ERROR: checkout is at $ACTUAL_COMMIT but FP_EXPECTED_COMMIT=$FP_EXPECTED_COMMIT" >&2
    exit 2
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- FP_PINN/pinn/cubic_homogeneous_3d 2>/dev/null)" ]]; then
    echo "WARNING: uncommitted changes under FP_PINN/pinn/cubic_homogeneous_3d (run is not reproducible from the commit)"
fi

RUN_ROOT="${FP_OUTPUT_ROOT:-outputs/g2-${SLURM_ARRAY_JOB_ID:-local}}"
CASE_OUTPUT="$RUN_ROOT/$VARIANT"
REFERENCE_OUTPUT="$CASE_OUTPUT/reference_fv"
mkdir -p "$REFERENCE_OUTPUT"

package_on_exit() {
    local run_status=$?
    local package_status=0
    trap - EXIT
    if [[ "${FP_PACKAGE_RESULTS:-1}" == "1" ]]; then
        local archive="$REPO_ROOT/FP_PINN_G2_JOB${SLURM_ARRAY_JOB_ID:-local}_STRESS_${VARIANT^^}_COMPLETE.zip"
        local job_record log_key log_path
        local -a log_args=()
        job_record="$(scontrol show job -o "${SLURM_JOB_ID:-}" 2>/dev/null || true)"
        for log_key in StdOut StdErr; do
            log_path="$(sed -n "s/.* ${log_key}=\([^ ]*\).*/\1/p" <<<"$job_record")"
            if [[ -n "$log_path" && -f "$log_path" ]]; then
                log_args+=(--log "$log_path")
            fi
        done
        python g2/package_g2.py \
            --case-output "$CASE_OUTPUT" \
            --archive "$archive" \
            --repo-root "$REPO_ROOT" \
            --job-id "${SLURM_JOB_ID:-local}" \
            --array-job-id "${SLURM_ARRAY_JOB_ID:-local}" \
            --task-id "$TASK" \
            --seed "$RUN_SEED" \
            --variant "$VARIANT" \
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
capability = tuple(int(v) for v in details.get("compute_capability", (0, 0)))
minimum = tuple(int(v) for v in os.environ.get("FP_MIN_GPU_COMPUTE_CAPABILITY", "7.5").split("."))
print("GPU compute capability:", ".".join(map(str, capability)), "minimum:", minimum)
if capability < minimum:
    raise SystemExit("Allocated GPU is incompatible with this TensorFlow/CUDA build; submit with --constraint=sm_75&vram12")
PY

# 1. Structural unit tests (G0 operator + accepted G1 + new G2 tests).
python -m unittest discover -s "$STAGE2_DIR/tests" -t "$STAGE2_DIR" -v
python -m unittest discover -s "$STAGE2_DIR/g1/tests" -t "$STAGE2_DIR" -v
python -m unittest discover -s "$G2_DIR/tests" -t "$STAGE2_DIR" -v

# 2. Deterministic axisymmetric finite-volume reference (CPU, ~4 min at 400x200).
python -u g2/axisym_stress_reference.py \
    --output-dir "$REFERENCE_OUTPUT" \
    --nx "${FP_FV_NX:-400}" --nr "${FP_FV_NR:-200}" --dt "${FP_FV_DT:-2.0e-4}" \
    --L "${FP_FV_L:-8.0}" --tmax "${FP_TMAX:-1.0}" --save-interval "${FP_FV_SAVE_INTERVAL:-0.05}" \
    --nu "${FP_NU:-1.0}"

# 3. Training.
TRAIN_ARGS=(
    --output-dir "$CASE_OUTPUT"
    --reference "$REFERENCE_OUTPUT/reference.npz"
    --epochs "${FP_G2_EPOCHS:-25000}"
    --n-time-batch "${FP_G2_N_TIME_BATCH:-16}"
    --time-power "${FP_G2_TIME_POWER:-1.0}"
    --n-cx "${FP_G2_N_CX:-129}"
    --n-rho "${FP_G2_N_RHO:-32}"
    --cx-half-width "${FP_G2_CX_HALF_WIDTH:-8.0}"
    --rho-max "${FP_G2_RHO_MAX:-8.0}"
    --eval-n-cx "${FP_G2_EVAL_N_CX:-257}"
    --eval-n-rho "${FP_G2_EVAL_N_RHO:-64}"
    --eval-cx-half-width "${FP_G2_EVAL_CX_HALF_WIDTH:-9.0}"
    --eval-rho-max "${FP_G2_EVAL_RHO_MAX:-9.0}"
    --width "${FP_STAGE2_WIDTH:-128}"
    --depth "${FP_STAGE2_DEPTH:-5}"
    --learning-rate "${FP_G2_LEARNING_RATE:-2.0e-4}"
    --lr-decay-steps "${FP_G2_LR_DECAY_STEPS:-8000}"
    --lr-decay-rate "${FP_G2_LR_DECAY_RATE:-0.3}"
    --gradient-clip-norm "${FP_G2_GRADIENT_CLIP_NORM:-1.0}"
    --correction-cap "${FP_CORRECTION_CAP:-12.0}"
    --stress-mode-cap "${FP_G2_STRESS_MODE_CAP:-0.10}"
    --tilt-penalty-weight "${FP_G2_TILT_PENALTY_WEIGHT:-10.0}"
    --stress-rate-weight "${FP_G2_STRESS_RATE_WEIGHT:-0.0}"
    --print-every "${FP_G2_PRINT_EVERY:-100}"
    --checkpoint-every "${FP_G2_CHECKPOINT_EVERY:-1000}"
    --seed "$RUN_SEED"
)
if [[ "${FP_G2_STOP_GRADIENT_CLOSURE:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--stop-gradient-closure)
fi
if [[ "${FP_G2_USE_STRESS_MODE:-1}" == "0" || "$VARIANT" == "no_stress_mode" ]]; then
    TRAIN_ARGS+=(--no-use-stress-mode)
fi
if [[ "${FP_G2_SHIFT_CX_GRID:-1}" == "0" ]]; then
    TRAIN_ARGS+=(--no-shift-cx-grid)
fi
python -u g2/train_g2.py "${TRAIN_ARGS[@]}"

# 4. Deterministic checkpoint sweep and selection (held-out fine quadrature).
python -u g2/evaluate_g2.py \
    --config "$CASE_OUTPUT/config.json" \
    --sweep-dir "$CASE_OUTPUT/checkpoints_h5" \
    --reference "$REFERENCE_OUTPUT/reference.npz" \
    --output "$CASE_OUTPUT/checkpoint_sweep"

echo "G2 task completed: $VARIANT (seed $RUN_SEED)"
echo "Artifacts: $STAGE2_DIR/$CASE_OUTPUT"

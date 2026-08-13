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
CASE_NAME="${CASES[$SLURM_ARRAY_TASK_ID]}"

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
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    raise SystemExit("No TensorFlow GPU is visible inside this Slurm allocation")
PY

python -m unittest discover -s "$STAGE2_DIR/tests" -t "$STAGE2_DIR" -v

RUN_ROOT="${FP_OUTPUT_ROOT:-outputs/stage2-${SLURM_ARRAY_JOB_ID}}"
CASE_OUTPUT="$RUN_ROOT/$CASE_NAME"
REFERENCE_OUTPUT="$CASE_OUTPUT/reference_particle"
mkdir -p "$REFERENCE_OUTPUT"

python -u reference_particle.py \
    --case "$CASE_NAME" \
    --output-dir "$REFERENCE_OUTPUT" \
    --particles "${FP_REFERENCE_PARTICLES:-250000}" \
    --dt "${FP_REFERENCE_DT:-0.005}" \
    --tmax "${FP_TMAX:-1.0}" \
    --save-every "${FP_REFERENCE_SAVE_EVERY:-10}" \
    --seed "$((20260808 + SLURM_ARRAY_TASK_ID))"

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
    --heat-flux-weight "${FP_HEAT_FLUX_WEIGHT:-10.0}"
    --heat-flux-scale "${FP_HEAT_FLUX_SCALE:-0.25}"
    --tail-fraction "${FP_TAIL_FRACTION:-0.20}"
    --tail-variance "${FP_TAIL_VARIANCE:-4.0}"
    --evaluation-samples "${FP_EVALUATION_SAMPLES:-65536}"
    --seed "$((20260808 + SLURM_ARRAY_TASK_ID))"
)

if [[ -n "${FP_RESUME_WEIGHTS:-}" ]]; then
    TRAIN_ARGS+=(--resume-weights "$FP_RESUME_WEIGHTS")
fi
if [[ "${FP_EVALUATE_ONLY:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--evaluate-only)
fi
if [[ "${FP_STRICT_GATE:-0}" == "1" ]]; then
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

echo "Stage-2 case completed: $CASE_NAME"
echo "Artifacts: $STAGE2_DIR/$CASE_OUTPUT"

if [[ "${FP_PACKAGE_RESULTS:-1}" == "1" ]]; then
    ARCHIVE="$REPO_ROOT/FP_PINN_STAGE2_JOB${SLURM_ARRAY_JOB_ID}_${CASE_NAME^^}_COMPLETE.zip"
    LOG_ARGS=()
    JOB_RECORD="$(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null || true)"
    for LOG_KEY in StdOut StdErr; do
        LOG_PATH="$(sed -n "s/.* ${LOG_KEY}=\([^ ]*\).*/\1/p" <<<"$JOB_RECORD")"
        if [[ -n "$LOG_PATH" && -f "$LOG_PATH" ]]; then
            LOG_ARGS+=(--log "$LOG_PATH")
        fi
    done
    python package_stage2.py \
        --case-output "$CASE_OUTPUT" \
        --archive "$ARCHIVE" \
        --repo-root "$REPO_ROOT" \
        --job-id "$SLURM_JOB_ID" \
        --array-job-id "$SLURM_ARRAY_JOB_ID" \
        --task-id "$SLURM_ARRAY_TASK_ID" \
        --case "$CASE_NAME" \
        "${LOG_ARGS[@]}"
fi

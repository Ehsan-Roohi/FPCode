#!/usr/bin/env bash
#SBATCH --job-name=fp-s2-q-v3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=fp-pinn-stage2-v3-%j.out
#SBATCH --error=fp-pinn-stage2-v3-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="${FP_REPO_ROOT:-$SUBMIT_DIR}"
STAGE2_DIR="${FP_STAGE2_DIR:-$REPO_ROOT/FP_PINN/pinn/cubic_homogeneous_3d}"
if [[ ! -f "$STAGE2_DIR/calibrate_stage2_heat_flux.py" ]]; then
    echo "ERROR: corrected Stage-2 checkout not found: $STAGE2_DIR" >&2
    exit 2
fi
cd "$STAGE2_DIR"

RUN_ROOT="${FP_OUTPUT_ROOT:-outputs/stage2-v3-${SLURM_JOB_ID}}"
CASE_OUTPUT="$RUN_ROOT/heat_flux"
mkdir -p "$CASE_OUTPUT"

package_on_exit() {
    local run_status=$?
    local archive="$REPO_ROOT/FP_PINN_STAGE2_V3_JOB${SLURM_JOB_ID}_HEAT_FLUX_COMPLETE.zip"
    trap - EXIT
    python package_stage2.py \
        --case-output "$CASE_OUTPUT" \
        --archive "$archive" \
        --repo-root "$REPO_ROOT" \
        --job-id "$SLURM_JOB_ID" \
        --array-job-id "$SLURM_JOB_ID" \
        --task-id 0 \
        --case heat_flux \
        --log "$SUBMIT_DIR/fp-pinn-stage2-v3-${SLURM_JOB_ID}.out" \
        --log "$SUBMIT_DIR/fp-pinn-stage2-v3-${SLURM_JOB_ID}.err" || true
    exit "$run_status"
}
trap package_on_exit EXIT

module load conda/latest
UNITY_CONDA_ENV="${FP_CONDA_ENV:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$UNITY_CONDA_ENV"
TF_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
for NVIDIA_LIB in "$TF_SITE"/nvidia/*/lib; do
    if [[ -d "$NVIDIA_LIB" ]]; then
        export LD_LIBRARY_PATH="$NVIDIA_LIB:${LD_LIBRARY_PATH:-}"
    fi
done
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fp-pinn-matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

nvidia-smi
python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    raise SystemExit("No TensorFlow GPU is visible")
PY
python -m unittest discover -s "$STAGE2_DIR/tests" -t "$STAGE2_DIR" -v

if [[ -n "${FP_RESUME_ARCHIVE:-}" && -n "${FP_RESUME_WEIGHTS:-}" ]]; then
    echo "ERROR: set only one of FP_RESUME_ARCHIVE or FP_RESUME_WEIGHTS" >&2
    exit 2
fi
if [[ -n "${FP_RESUME_ARCHIVE:-}" ]]; then
    python -u prepare_stage2_resume.py \
        --archive "$FP_RESUME_ARCHIVE" \
        --case-output "$CASE_OUTPUT"
    FP_RESUME_WEIGHTS="$CASE_OUTPUT/resume_input.weights.h5"
    FP_REFERENCE="$CASE_OUTPUT/resume_source/reference_particle/reference.npz"
fi
if [[ -z "${FP_RESUME_WEIGHTS:-}" || -z "${FP_REFERENCE:-}" ]]; then
    echo "ERROR: set FP_RESUME_ARCHIVE, or both FP_RESUME_WEIGHTS and FP_REFERENCE" >&2
    exit 2
fi

python -u calibrate_stage2_heat_flux.py \
    --resume-weights "$FP_RESUME_WEIGHTS" \
    --reference "$FP_REFERENCE" \
    --output-dir "$CASE_OUTPUT" \
    --steps "${FP_PROJECTION_STEPS:-500}" \
    --learning-rate "${FP_PROJECTION_LR:-3.0e-5}" \
    --gradient-clip-norm "${FP_PROJECTION_GRAD_CLIP:-2.0}" \
    --train-order "${FP_PROJECTION_TRAIN_ORDER:-16}" \
    --heldout-order "${FP_PROJECTION_HELDOUT_ORDER:-20}" \
    --check-every "${FP_PROJECTION_CHECK_EVERY:-25}" \
    --evaluation-samples "${FP_EVALUATION_SAMPLES:-131072}" \
    --evaluation-quadrature-order "${FP_EVALUATION_QUADRATURE_ORDER:-48}" \
    --strict-gate

python -u audit_stage2_residual.py \
    --weights "$CASE_OUTPUT/resume_input.weights.h5" \
    --output "$CASE_OUTPUT/residual_audit_baseline.json"
python -u audit_stage2_residual.py \
    --weights "$CASE_OUTPUT/stage2_best.weights.h5" \
    --output "$CASE_OUTPUT/residual_audit_best.json"

echo "Stage-2 v3 heat-flux projection completed"
echo "Root archive will be created at exit"

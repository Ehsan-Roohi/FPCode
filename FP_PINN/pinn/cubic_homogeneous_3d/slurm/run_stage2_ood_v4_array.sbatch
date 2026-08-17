#!/usr/bin/env bash
#SBATCH --job-name=fp-pinn-v4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-8%3
#SBATCH --output=FP_PINN_STAGE2_V4_%A_%a.out
#SBATCH --error=FP_PINN_STAGE2_V4_%A_%a.err

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(git -C "$module_dir" rev-parse --show-toplevel)"
env_dir="${FP_V4_ENV_DIR:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu}"
if [[ ! -f "$env_dir/bin/activate" ]]; then
    echo "Missing Python environment: $env_dir" >&2
    exit 3
fi
source "$env_dir/bin/activate"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1
tf_site="$(python -c 'import site; print(site.getsitepackages()[0])')"
for nvidia_lib in "$tf_site"/nvidia/*/lib; do
    if [[ -d "$nvidia_lib" ]]; then
        export LD_LIBRARY_PATH="$nvidia_lib:${LD_LIBRARY_PATH:-}"
    fi
done
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fp-pinn-v4-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    raise SystemExit("No TensorFlow GPU is visible")
PY

cases=(
    heat_flux
    ood_hf_q0125
    ood_hf_q0400
    ood_hf_shape_w020
    ood_hf_nu050
    ood_hf_nu200
    ood_stress_mild
    ood_stress_strong
    ood_coupled_axisym
)
case_name="${cases[$SLURM_ARRAY_TASK_ID]}"
nu="$(PYTHONPATH="$module_dir" python -c 'import sys; from cubic_operator import case_default_nu; print(case_default_nu(sys.argv[1]))' "$case_name")"
has_heat_flux="$(PYTHONPATH="$module_dir" python -c 'import sys; from cubic_operator import case_has_heat_flux; print(int(case_has_heat_flux(sys.argv[1])))' "$case_name")"
if [[ "$has_heat_flux" == "1" ]]; then
    default_particles=500000
else
    default_particles=250000
fi
run_root="$repo_root/outputs/stage2-v4-${SLURM_ARRAY_JOB_ID}"
case_output="$run_root/$case_name"
reference_dir="$case_output/reference_particle"
training_dir="$case_output/training"
final_dir="$case_output/final"
status_file="$case_output/case_status.json"
result_dir=""
result_weights=""
started_monotonic="$(python -c 'import time; print(time.monotonic())')"
mkdir -p "$reference_dir" "$training_dir"

finish_case() {
    local exit_code=$?
    trap - EXIT
    python "$module_dir/record_stage2_ood_case.py" \
        --status "$status_file" \
        --case "$case_name" \
        --nu "$nu" \
        --job-id "$SLURM_JOB_ID" \
        --array-job-id "$SLURM_ARRAY_JOB_ID" \
        --task-id "$SLURM_ARRAY_TASK_ID" \
        --exit-code "$exit_code" \
        --started-monotonic "$started_monotonic" \
        --result-dir "$result_dir" || true
    exit "$exit_code"
}
trap finish_case EXIT

python "$module_dir/reference_particle.py" \
    --case "$case_name" \
    --output-dir "$reference_dir" \
    --particles "${FP_V4_PARTICLES:-$default_particles}" \
    --dt "${FP_V4_PARTICLE_DT:-0.005}" \
    --save-every "${FP_V4_PARTICLE_SAVE_EVERY:-10}" \
    --nu "$nu" \
    --seed "$((20260816 + SLURM_ARRAY_TASK_ID))"

python "$module_dir/train_stage2.py" \
    --case "$case_name" \
    --output-dir "$training_dir" \
    --reference "$reference_dir/reference.npz" \
    --epochs "${FP_V4_EPOCHS:-15000}" \
    --n-time-batch "${FP_V4_TIME_BATCH:-8}" \
    --n-velocity-per-time "${FP_V4_VELOCITY_BATCH:-1024}" \
    --fixed-velocity-quadrature \
    --quadrature-panels "${FP_V4_QUADRATURE_PANELS:-4}" \
    --evaluation-quadrature-order "${FP_V4_EVAL_ORDER:-48}" \
    --evaluation-samples "${FP_V4_EVAL_SAMPLES:-131072}" \
    --nu "$nu" \
    --seed "$((20260816 + 100 * SLURM_ARRAY_TASK_ID))"

if [[ "$has_heat_flux" == "1" ]]; then
    mkdir -p "$final_dir"
    python "$module_dir/calibrate_stage2_heat_flux.py" \
        --case "$case_name" \
        --resume-weights "$training_dir/stage2_final.weights.h5" \
        --reference "$reference_dir/reference.npz" \
        --output-dir "$final_dir" \
        --steps "${FP_V4_PROJECTION_STEPS:-500}" \
        --evaluation-quadrature-order "${FP_V4_EVAL_ORDER:-48}" \
        --evaluation-samples "${FP_V4_EVAL_SAMPLES:-131072}" \
        --nu "$nu" \
        --seed "$((20260816 + 1000 * SLURM_ARRAY_TASK_ID))"
    result_dir="$final_dir"
    result_weights="$final_dir/stage2_best.weights.h5"
else
    result_dir="$training_dir"
    result_weights="$training_dir/stage2_final.weights.h5"
fi

python "$module_dir/audit_stage2_residual.py" \
    --case "$case_name" \
    --weights "$result_weights" \
    --output "$result_dir/fresh_residual_audit.json" \
    --nu "$nu" \
    --seed "$((24681357 + SLURM_ARRAY_TASK_ID))"

echo "CASE_COMPLETE case=$case_name result=$result_dir"

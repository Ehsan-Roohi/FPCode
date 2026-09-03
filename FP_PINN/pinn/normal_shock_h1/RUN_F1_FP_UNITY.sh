#!/usr/bin/env bash
set -euo pipefail

ROOT="${FP_H1_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
HERE="$ROOT/FP_PINN/pinn/normal_shock_h1"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
REFERENCE="${FP_F1_REFERENCE:?FP_F1_REFERENCE must name the registered Mach-2 BGK full-state NPZ}"
RESTART="${FP_F1_RESTART:?FP_F1_RESTART must name the successful H2R2 weights file}"
MACH="${FP_F1_MACH:-2}"

[[ -z "${FP_EXPECTED_COMMIT:-}" || "$COMMIT" == "$FP_EXPECTED_COMMIT"* ]] || {
    echo "ERROR expected commit $FP_EXPECTED_COMMIT but found $COMMIT" >&2
    exit 2
}
[[ -f "$RESTART" ]] || { echo "ERROR H2R2 restart not found: $RESTART" >&2; exit 2; }

if [[ "${FP_F1_BATCH:-0}" != 1 ]]; then
    mkdir -p "$ROOT/slurm_logs"
    JOB=$(sbatch --parsable \
        --job-name=f1-fp \
        --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --mem=32G --time="${FP_F1_TIME:-06:00:00}" \
        --gres=gpu:1 --constraint="${FP_GPU_CONSTRAINT:-sm_75&vram12}" \
        --output="$ROOT/slurm_logs/f1-fp-%j.out" \
        --error="$ROOT/slurm_logs/f1-fp-%j.err" \
        --export="ALL,FP_F1_BATCH=1,FP_H1_ROOT=$ROOT,FP_F1_REFERENCE=$REFERENCE,FP_F1_RESTART=$RESTART,FP_F1_MACH=$MACH,FP_EXPECTED_COMMIT=$COMMIT,FP_F1_EPOCHS=${FP_F1_EPOCHS:-4500}" \
        "$HERE/RUN_F1_FP_UNITY.sh")
    echo "Submitted F1 Dougherty-FP pilot job ${JOB%%;*}"
    echo "Expected archive: $ROOT/FP_PINN_F1_FP_JOB${JOB%%;*}_M${MACH//./p}_COMPLETE.zip"
    exit 0
fi

module load conda/latest
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${FP_CONDA_ENV:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu}"
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
for DIR in "$SITE"/nvidia/*/lib; do
    [[ ! -d "$DIR" ]] || export LD_LIBRARY_PATH="$DIR:${LD_LIBRARY_PATH:-}"
done

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
python -c 'import tensorflow as tf,sys; g=tf.config.list_physical_devices("GPU"); print("F1_GPUS",g); sys.exit(0 if g else 2)'

cd "$HERE"
python -m unittest -v \
    test_shock_physics.py test_fp_operator.py test_h2_reference.py \
    test_h2_bgk.py test_f1_fp.py

AUDIT="$HERE/outputs/f1_reference_preflight_${SLURM_JOB_ID}"
python audit_h2_reference.py --reference "$REFERENCE" --mach "$MACH" --output "$AUDIT"

OUT="$HERE/outputs/f1_fp_m${MACH//./p}_${SLURM_JOB_ID}"
set +e
python train_f1_fp.py \
    --reference "$REFERENCE" --restart "$RESTART" --mach "$MACH" \
    --epochs "${FP_F1_EPOCHS:-4500}" \
    --nx-pde "${FP_F1_NX_PDE:-129}" \
    --projection-steps "${FP_F1_PROJECTION_STEPS:-10}" \
    --output "$OUT"
RC=$?
set -e

if [[ -d "$OUT" ]] && find "$OUT" -maxdepth 1 -type f -print -quit | grep -q .; then
    (cd "$OUT" && find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum > SHA256SUMS)
    ARCHIVE="$ROOT/FP_PINN_F1_FP_JOB${SLURM_JOB_ID}_M${MACH//./p}_COMPLETE.zip"
    (cd "$HERE/outputs" && zip -qr "$ARCHIVE" "$(basename "$OUT")" "$(basename "$AUDIT")")
    (cd "$ROOT" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
    echo "F1_ARCHIVE=$ARCHIVE"
else
    echo "F1_OUTPUT_NOT_CREATED=$OUT" >&2
fi
exit "$RC"

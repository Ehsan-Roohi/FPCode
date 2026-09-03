#!/usr/bin/env bash
set -euo pipefail

ROOT="${FP_H1_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
HERE="$ROOT/FP_PINN/pinn/normal_shock_h1"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
REFERENCE="${FP_H2_REFERENCE:?FP_H2_REFERENCE must name the registered Mach-2 full-state NPZ}"
MACH="${FP_H2_MACH:-2}"

[[ -z "${FP_EXPECTED_COMMIT:-}" || "$COMMIT" == "$FP_EXPECTED_COMMIT"* ]] || {
    echo "ERROR expected commit $FP_EXPECTED_COMMIT but found $COMMIT" >&2
    exit 2
}

if [[ "${FP_H2_BATCH:-0}" != 1 ]]; then
    mkdir -p "$ROOT/slurm_logs"
    JOB=$(sbatch --parsable \
        --job-name=h2-bgk \
        --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --mem=32G --time="${FP_H2_TIME:-06:00:00}" \
        --gres=gpu:1 --constraint="${FP_GPU_CONSTRAINT:-sm_75&vram12}" \
        --output="$ROOT/slurm_logs/h2-bgk-%j.out" \
        --error="$ROOT/slurm_logs/h2-bgk-%j.err" \
        --export="ALL,FP_H2_BATCH=1,FP_H1_ROOT=$ROOT,FP_H2_REFERENCE=$REFERENCE,FP_H2_MACH=$MACH,FP_EXPECTED_COMMIT=$COMMIT" \
        "$HERE/RUN_H2_BGK_UNITY.sh")
    echo "Submitted H2 BGK job ${JOB%%;*}"
    echo "Expected archive: $ROOT/FP_PINN_H2_BGK_JOB${JOB%%;*}_M${MACH//./p}_COMPLETE.zip"
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
python -c 'import tensorflow as tf,sys; g=tf.config.list_physical_devices("GPU"); print("H2_GPUS",g); sys.exit(0 if g else 2)'

cd "$HERE"
python -m unittest -v test_shock_physics.py test_fp_operator.py test_h2_reference.py test_h2_bgk.py
AUDIT="$HERE/outputs/h2_reference_preflight_${SLURM_JOB_ID}"
python audit_h2_reference.py --reference "$REFERENCE" --mach "$MACH" --output "$AUDIT"

OUT="$HERE/outputs/h2_bgk_m${MACH//./p}_${SLURM_JOB_ID}"
set +e
python train_h2r2_bgk.py \
    --reference "$REFERENCE" --mach "$MACH" \
    --epochs "${FP_H2_EPOCHS:-6000}" \
    --macro-epochs "${FP_H2_MACRO_EPOCHS:-1400}" \
    --projection-steps "${FP_H2_PROJECTION_STEPS:-10}" \
    --output "$OUT"
RC=$?
set -e

if [[ -d "$OUT" ]] && find "$OUT" -maxdepth 1 -type f -print -quit | grep -q .; then
    (cd "$OUT" && find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum > SHA256SUMS)
    ARCHIVE="$ROOT/FP_PINN_H2_BGK_JOB${SLURM_JOB_ID}_M${MACH//./p}_COMPLETE.zip"
    (cd "$HERE/outputs" && zip -qr "$ARCHIVE" "$(basename "$OUT")" "$(basename "$AUDIT")")
    (cd "$ROOT" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
    echo "H2_ARCHIVE=$ARCHIVE"
else
    echo "H2_OUTPUT_NOT_CREATED=$OUT" >&2
fi
exit "$RC"

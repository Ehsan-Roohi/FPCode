#!/usr/bin/env bash
set -euo pipefail
ROOT="${FP_H1_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"; HERE="$ROOT/FP_PINN/pinn/normal_shock_h1"; C="$(git -C "$ROOT" rev-parse HEAD)"; MACH="${FP_H1_MACH:-5}"; OUT="$HERE/outputs/h1_gate1_m${MACH//./p}_${SLURM_JOB_ID:-local}"
[[ -z "${FP_EXPECTED_COMMIT:-}" || "$C" == "$FP_EXPECTED_COMMIT"* ]] || { echo "ERROR commit $C" >&2; exit 2; }
if [[ "${FP_H1_BATCH_ALLOCATED:-0}" != "1" ]]; then
 mkdir -p "$ROOT/slurm_logs"; J=$(sbatch --parsable --job-name=h1-g1-fp --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=06:00:00 --gres=gpu:1 --constraint="${FP_GPU_CONSTRAINT:-sm_75&vram12}" --output="$ROOT/slurm_logs/h1-g1-%j.out" --error="$ROOT/slurm_logs/h1-g1-%j.err" --export="ALL,FP_H1_BATCH_ALLOCATED=1,FP_H1_ROOT=$ROOT,FP_H1_MACH=$MACH,FP_EXPECTED_COMMIT=$C" "$HERE/RUN_H1_GATE1_UNITY.sh"); echo "Submitted H1 Gate 1 job ${J%%;*}"; exit 0
fi
module load conda/latest; CONDA_BASE="$(conda info --base)"; source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "${FP_CONDA_ENV:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu}"
TF_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"; for D in "$TF_SITE"/nvidia/*/lib; do [[ ! -d "$D" ]] || export LD_LIBRARY_PATH="$D:${LD_LIBRARY_PATH:-}"; done
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
python - <<'PY'
import tensorflow as tf
gpus=tf.config.list_physical_devices("GPU")
print("H1_GATE1_GPUS",gpus,flush=True)
if not gpus: raise SystemExit("ERROR: H1 Gate 1 requires a visible CUDA GPU")
PY
cd "$HERE"; python -m unittest -v test_shock_physics.py test_fp_operator.py; set +e; python train_gate1.py --mach "$MACH" --epochs "${FP_H1_EPOCHS:-12000}" --output "$OUT"; RC=$?; set -e
sha256sum "$OUT"/* > "$OUT/SHA256SUMS"; ARCHIVE="$ROOT/FP_PINN_H1_GATE1_JOB${SLURM_JOB_ID}_M${MACH//./p}_COMPLETE.zip"; (cd "$HERE/outputs" && zip -qr "$ARCHIVE" "$(basename "$OUT")"); sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"; echo "H1_GATE1_OUTPUT=$OUT"; echo "H1_GATE1_ARCHIVE=$ARCHIVE"
exit "$RC"

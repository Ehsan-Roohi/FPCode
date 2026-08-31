#!/usr/bin/env bash
set -euo pipefail
ROOT="${FP_H1_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"; HERE="$ROOT/FP_PINN/pinn/normal_shock_h1"; C="$(git -C "$ROOT" rev-parse HEAD)"; MACH="${FP_H1_MACH:-5}"
[[ -z "${FP_EXPECTED_COMMIT:-}" || "$C" == "$FP_EXPECTED_COMMIT"* ]] || { echo "ERROR commit $C" >&2; exit 2; }
if [[ "${FP_H1R_BATCH:-0}" != 1 ]]; then
 mkdir -p "$ROOT/slurm_logs"; J=$(sbatch --parsable --job-name=h1r-pilot --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=04:00:00 --gres=gpu:1 --constraint="${FP_GPU_CONSTRAINT:-sm_75&vram12}" --output="$ROOT/slurm_logs/h1r-%j.out" --error="$ROOT/slurm_logs/h1r-%j.err" --export="ALL,FP_H1R_BATCH=1,FP_H1_ROOT=$ROOT,FP_H1_MACH=$MACH,FP_EXPECTED_COMMIT=$C" "$HERE/RUN_H1R_PILOT_UNITY.sh"); echo "Submitted H1R pilot job ${J%%;*}"; exit 0
fi
module load conda/latest; B="$(conda info --base)"; source "$B/etc/profile.d/conda.sh"; conda activate "${FP_CONDA_ENV:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu}"; S="$(python -c 'import site; print(site.getsitepackages()[0])')"; for D in "$S"/nvidia/*/lib; do [[ ! -d "$D" ]] || export LD_LIBRARY_PATH="$D:${LD_LIBRARY_PATH:-}"; done
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader; python -c 'import tensorflow as tf,sys; g=tf.config.list_physical_devices("GPU"); print("H1R_GPUS",g); sys.exit(0 if g else 2)'
OUT="$HERE/outputs/h1r_m${MACH//./p}_${SLURM_JOB_ID}"; cd "$HERE"; python -m unittest -v test_shock_physics.py test_fp_operator.py; set +e; python train_h1r.py --mach "$MACH" --epochs "${FP_H1R_EPOCHS:-3000}" --output "$OUT"; RC=$?; set -e
if [[ -d "$OUT" ]] && compgen -G "$OUT/*" >/dev/null; then
 sha256sum "$OUT"/* > "$OUT/SHA256SUMS"; A="$ROOT/FP_PINN_H1R_JOB${SLURM_JOB_ID}_M${MACH//./p}_COMPLETE.zip"; (cd "$HERE/outputs" && zip -qr "$A" "$(basename "$OUT")"); sha256sum "$A" > "$A.sha256"; echo "H1R_OUTPUT=$OUT"; echo "H1R_ARCHIVE=$A"
else
 echo "H1R_OUTPUT_NOT_CREATED=$OUT" >&2
fi
exit "$RC"

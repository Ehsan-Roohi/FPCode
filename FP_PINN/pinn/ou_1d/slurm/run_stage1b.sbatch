#!/bin/bash
#SBATCH --job-name=fp-pinn-s1b
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=fp-pinn-stage1b-%j.out
#SBATCH --error=fp-pinn-stage1b-%j.err

set -euo pipefail

module purge
module load conda/latest

FP_CONDA_ENV="${FP_CONDA_ENV:-dsmc-gpu}"
FP_CONDA_BASE="$(conda info --base)"
MAMBA_SH="$FP_CONDA_BASE/etc/profile.d/mamba.sh"
if [[ ! -f "$MAMBA_SH" ]]; then
    echo "Cannot find Unity Mamba profile script: $MAMBA_SH" >&2
    exit 2
fi
source "$MAMBA_SH"
mamba activate "$FP_CONDA_ENV"

echo "Activated environment: $CONDA_PREFIX"
command -v python
python --version
nvidia-smi

FP_NVIDIA_LIBS="$(
python - <<'PY'
import site
from pathlib import Path

roots = [Path(path) for path in site.getsitepackages()]
user_site = Path(site.getusersitepackages())
if user_site not in roots:
    roots.append(user_site)
lib_dirs = []
for root in roots:
    base = root / "nvidia"
    if base.is_dir():
        lib_dirs.extend(
            path.resolve() for path in sorted(base.glob("*/lib")) if path.is_dir()
        )
print(":".join(dict.fromkeys(map(str, lib_dirs))))
PY
)"
if [[ -z "$FP_NVIDIA_LIBS" ]]; then
    echo "No pip NVIDIA library directories were found." >&2
    exit 3
fi
export LD_LIBRARY_PATH="$FP_NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - <<'PY'
import ctypes
import tensorflow as tf

for library in (
    "libcudart.so.12",
    "libcublas.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcusolver.so.11",
    "libcusparse.so.12",
    "libcudnn.so.9",
):
    ctypes.CDLL(library)
    print("Loaded", library)
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow", tf.__version__)
print("Visible GPUs", gpus)
if not gpus:
    raise SystemExit("No TensorFlow GPU is visible inside this Slurm allocation")
PY

REPO_ROOT="${FP_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
WORK_DIR="$REPO_ROOT/FP_PINN/pinn/ou_1d"
TRAINER="$WORK_DIR/train_stage1b.py"
if [[ ! -f "$TRAINER" ]]; then
    echo "Cannot find $TRAINER" >&2
    echo "Submit this script from the FPCode repository root." >&2
    exit 2
fi

cd "$WORK_DIR"
OUTPUT_DIR="${FP_OUTPUT_DIR:-outputs/stage1b-${SLURM_JOB_ID}}"

if [[ "${FP_STAGE1B_RESUME:-0}" == "1" ]]; then
    python -u train_stage1b.py \
        --output-dir "$OUTPUT_DIR" \
        --resume \
        "$@"
else
    if [[ -z "${FP_INIT_CHECKPOINT:-}" ]]; then
        echo "Set FP_INIT_CHECKPOINT to the Stage-1 ckpt-30000 path." >&2
        exit 2
    fi
    python -u train_stage1b.py \
        --output-dir "$OUTPUT_DIR" \
        --init-checkpoint "$FP_INIT_CHECKPOINT" \
        "$@"
fi

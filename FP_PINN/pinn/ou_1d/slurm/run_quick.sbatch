#!/bin/bash
#SBATCH --job-name=fp-pinn-ou
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=fp-pinn-ou-%j.out
#SBATCH --error=fp-pinn-ou-%j.err

set -euo pipefail

module purge
module load conda/latest

# Unity's Mamba 1.5.9 does not support "mamba shell hook", and its
# "mamba info --base" prints an extra version line. Use Conda's clean base path.
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
mamba list | grep -Ei 'tensorflow|cuda|cudnn|nvidia' || true

# TensorFlow's pip CUDA packages keep their shared libraries below
# site-packages/nvidia/*/lib. Prepend all of those directories so they take
# precedence over the older/mixed Conda CUDA runtimes in this environment.
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
    nvidia_root = root / "nvidia"
    if not nvidia_root.is_dir():
        continue
    lib_dirs.extend(
        path.resolve()
        for path in sorted(nvidia_root.glob("*/lib"))
        if path.is_dir()
    )

# Preserve order while removing duplicates.
print(":".join(dict.fromkeys(map(str, lib_dirs))))
PY
)"

if [[ -z "$FP_NVIDIA_LIBS" ]]; then
    echo "No pip NVIDIA library directories were found under site-packages." >&2
    exit 3
fi

export LD_LIBRARY_PATH="$FP_NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "TensorFlow NVIDIA library paths:"
tr ':' '\n' <<< "$FP_NVIDIA_LIBS"

# Fail early with the exact missing shared-library name instead of the generic
# TensorFlow "Cannot dlopen some GPU libraries" message.
python - <<'PY'
import ctypes

libraries = [
    "libcudart.so.12",
    "libcublas.so.12",
    "libcupti.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcusolver.so.11",
    "libcusparse.so.12",
    "libcudnn.so.9",
    "libnvJitLink.so.12",
    "libnvrtc.so.12",
]
for library in libraries:
    ctypes.CDLL(library)
    print("Loaded", library)
PY

REPO_ROOT="${FP_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
WORK_DIR="$REPO_ROOT/FP_PINN/pinn/ou_1d"
if [[ ! -f "$WORK_DIR/train.py" ]]; then
    echo "Cannot find $WORK_DIR/train.py" >&2
    echo "Submit this script from the FPCode repository root." >&2
    exit 2
fi

cd "$WORK_DIR"
python - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow", tf.__version__)
print("Visible GPUs", gpus)
if not gpus:
    raise SystemExit("No TensorFlow GPU is visible inside this Slurm allocation")
PY

OUTPUT_DIR="outputs/quick-${SLURM_JOB_ID}"
python -u train.py --quick --output-dir "$OUTPUT_DIR" "$@"

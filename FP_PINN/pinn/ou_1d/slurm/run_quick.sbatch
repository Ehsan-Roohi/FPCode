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

# Unity's installed Mamba does not support "mamba shell hook".
# Source its profile script before activating, matching prior successful Unity jobs.
FP_CONDA_ENV="${FP_CONDA_ENV:-dsmc-gpu}"
MAMBA_SH="$(mamba info --base)/etc/profile.d/mamba.sh"
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
print("TensorFlow", tf.__version__)
print("Visible GPUs", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    raise SystemExit("No TensorFlow GPU is visible inside this Slurm allocation")
PY

OUTPUT_DIR="outputs/quick-${SLURM_JOB_ID}"
python -u train.py --quick --output-dir "$OUTPUT_DIR" "$@"

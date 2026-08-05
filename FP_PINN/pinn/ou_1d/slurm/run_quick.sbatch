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
source "$(conda info --base)/etc/profile.d/conda.sh"

# A conda environment name or an absolute environment path can be supplied:
#   FP_CONDA_ENV=/project/.../envs/fp-pinn sbatch .../run_quick.sbatch
FP_CONDA_ENV="${FP_CONDA_ENV:-dsmc-gpu}"
conda activate "$FP_CONDA_ENV"

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

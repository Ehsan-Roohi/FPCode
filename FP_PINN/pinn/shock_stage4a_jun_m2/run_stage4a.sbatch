#!/usr/bin/env bash
#SBATCH --job-name=fp-shock-m2
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=fp-shock-m2-%j.out
#SBATCH --error=fp-shock-m2-%j.err

set -euo pipefail

module purge || true
module load conda/latest || true

REPO_ROOT="${STAGE4A_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
STAGE_ROOT="$REPO_ROOT/FP_PINN/pinn/shock_stage4a_jun_m2"
VENV_ROOT="${STAGE4A_VENV:-$REPO_ROOT/.venv-stage4a}"

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
    python3 -m venv "$VENV_ROOT"
    "$VENV_ROOT/bin/python" -m pip install --upgrade pip
    "$VENV_ROOT/bin/python" -m pip install -r "$STAGE_ROOT/requirements.txt"
fi

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-stage4a-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
export STAGE4A_PYTHON="$VENV_ROOT/bin/python"
export STAGE4A_OUTPUT="$STAGE_ROOT/outputs/job-${SLURM_JOB_ID}"

cd "$STAGE_ROOT"
bash run_one_line.sh


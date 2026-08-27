#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$FP_STAGE55_RUN"
"$FP_STAGE55_PYTHON" riemann35_patch/stage55_closure_source/collect_stage55.py --root "$FP_STAGE55_RESULT_ROOT" --bundle "$FP_STAGE55_BUNDLE" --no-fail

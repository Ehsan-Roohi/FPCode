#!/usr/bin/env bash
#SBATCH --job-name=fp-g1-agg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

# Fail-closed aggregation for the three preregistered G1 seeds.  This is a
# real Bash script rather than an sbatch --wrap command because Unity starts
# wrapped commands with /bin/sh, which does not define the `module` function.

set -euo pipefail

: "${FP_TEST:?FP_TEST must point to the pinned G1 checkout}"
: "${FP_G1_ARRAY_JOB:?FP_G1_ARRAY_JOB must be the completed array job ID}"

ACTUAL_COMMIT="$(git -C "$FP_TEST" rev-parse HEAD)"
if [[ -n "${FP_EXPECTED_COMMIT:-}" && "$ACTUAL_COMMIT" != "$FP_EXPECTED_COMMIT"* ]]; then
    echo "ERROR: checkout is at $ACTUAL_COMMIT but FP_EXPECTED_COMMIT=$FP_EXPECTED_COMMIT" >&2
    exit 2
fi

module load conda/latest
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${FP_CONDA_ENV:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu}"

STAGE2_DIR="$FP_TEST/FP_PINN/pinn/cubic_homogeneous_3d"
RUN_ROOT="$STAGE2_DIR/outputs/g1-$FP_G1_ARRAY_JOB"
cd "$STAGE2_DIR"

python g1/aggregate_g1_seeds.py --run-root "$RUN_ROOT"
cp "$RUN_ROOT/G1_SEED_SUMMARY.md" "$RUN_ROOT/G1_SEED_SUMMARY.json" "$FP_TEST/"

#!/usr/bin/env bash
#SBATCH --job-name=fp-g2-agg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

set -euo pipefail

: "${FP_TEST:?FP_TEST must point to the pinned G2 checkout}"
: "${FP_G2_ARRAY_JOB:?FP_G2_ARRAY_JOB must be the completed array job ID}"

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
RUN_ROOT="$STAGE2_DIR/outputs/g2-$FP_G2_ARRAY_JOB"
cd "$STAGE2_DIR"

ARGS=(--run-root "$RUN_ROOT")
if [[ -s "$FP_TEST/G1_SEED_SUMMARY.json" ]]; then
    ARGS+=(--g1-summary "$FP_TEST/G1_SEED_SUMMARY.json")
fi
python g2/aggregate_g2_seeds.py "${ARGS[@]}"
cp "$RUN_ROOT/G2_SEED_SUMMARY.md" "$RUN_ROOT/G2_SEED_SUMMARY.json" "$FP_TEST/"

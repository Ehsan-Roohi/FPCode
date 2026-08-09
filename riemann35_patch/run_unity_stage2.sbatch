#!/usr/bin/env bash
#SBATCH --job-name=riemann35-fp-stage2
#SBATCH --partition=cpu-preempt
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=riemann35-stage2-%j.out

set -euo pipefail

JULIA_MODULE="${JULIA_MODULE:-julia/1.10.5}"
module purge
module load "$JULIA_MODULE"

FP_REPO="${SLURM_SUBMIT_DIR}"
PATCH="$FP_REPO/riemann35_patch/0001-Add-provisional-cubic-FP-collision-source.patch"
STAGE2="$FP_REPO/riemann35_patch/stage2"
RESULTS="$FP_REPO/results/riemann35_stage2_${SLURM_JOB_ID}"
WORKDIR="$(mktemp -d /tmp/riemann35-stage2.XXXXXX)"
R35="$WORKDIR/Riemann35-stage2"
trap 'rm -rf "$WORKDIR"' EXIT

echo "Node: $(hostname)"
echo "Julia module: $JULIA_MODULE"
julia --version
python3 --version
mkdir -p "$RESULTS"

python3 "$STAGE2/export_particle_reference.py" \
  --particles 100000 \
  --steps 200 \
  --dt 2.5e-4 \
  --tau 1.0 \
  --prandtl 0.6666666666666666 \
  --gamma-scale 0.05 \
  --seed 42 \
  --sample-every 10 \
  --initial-output "$RESULTS/initial_moments.csv" \
  --history-output "$RESULTS/particle_history.csv"

git clone --quiet https://github.com/comp-physics/Riemann35.jl.git "$R35"
git -C "$R35" checkout --quiet 3ec8735
git -C "$R35" apply --check "$PATCH"
git -C "$R35" apply "$PATCH"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}"
export HYQMOM_SKIP_PLOTTING=true
export CI=true
cd "$R35"
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. "$STAGE2/run_homogeneous_stage2.jl" \
  "$RESULTS/initial_moments.csv" \
  "$RESULTS/julia_chyqmom_m6_history.csv" \
  200 2.5e-4 1.0 0.6666666666666666 0.05

python3 "$STAGE2/compare_stage2.py" \
  --julia "$RESULTS/julia_chyqmom_m6_history.csv" \
  --particle "$RESULTS/particle_history.csv" \
  --summary "$RESULTS/summary.json"

echo "Results: $RESULTS"
echo "PASS: Riemann35 cubic FP Stage 2 homogeneous comparison completed."

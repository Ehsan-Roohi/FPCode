#!/usr/bin/env bash
#SBATCH --job-name=riemann35-fp-stage4
#SBATCH --partition=cpu-preempt
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=riemann35-stage4-%j.out

set -euo pipefail

JULIA_MODULE="${JULIA_MODULE:-julia/1.10.5}"
module purge
module load "$JULIA_MODULE"

FP_REPO="${SLURM_SUBMIT_DIR}"
PATCH1="$FP_REPO/riemann35_patch/0001-Add-provisional-cubic-FP-collision-source.patch"
PATCH2="$FP_REPO/riemann35_patch/0002-Add-bounded-CHyQMOM-FP-collision-source.patch"
PATCH3="$FP_REPO/riemann35_patch/0003-Add-finite-step-CHyQMOM-FP-map.patch"
STAGE="$FP_REPO/riemann35_patch/stage2"
RESULTS="$FP_REPO/results/riemann35_stage4_${SLURM_JOB_ID}"
WORKDIR="$(mktemp -d /tmp/riemann35-stage4.XXXXXX)"
R35="$WORKDIR/Riemann35-stage4"
trap 'rm -rf "$WORKDIR"' EXIT

echo "Node: $(hostname)"
echo "Julia module: $JULIA_MODULE"
julia --version
python3 --version
mkdir -p "$RESULTS"

python3 "$STAGE/export_particle_reference.py" \
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
for patch in "$PATCH1" "$PATCH2" "$PATCH3"; do
  git -C "$R35" apply --check "$patch"
  git -C "$R35" apply "$patch"
done

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}"
export HYQMOM_SKIP_PLOTTING=true
export CI=true
cd "$R35"
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Riemann35; include("test/test_fp_collision35.jl")'
julia --project=. "$STAGE/run_homogeneous_stage2.jl" \
  "$RESULTS/initial_moments.csv" \
  "$RESULTS/julia_finite_history.csv" \
  "$RESULTS/julia_finite_metrics.csv" \
  200 2.5e-4 1.0 0.6666666666666666 0.05 finite

python3 "$STAGE/compare_stage2.py" \
  --julia "$RESULTS/julia_finite_history.csv" \
  --particle "$RESULTS/particle_history.csv" \
  --julia-metrics "$RESULTS/julia_finite_metrics.csv" \
  --summary "$RESULTS/summary.json"

echo "Results: $RESULTS"
echo "PASS: Riemann35 cubic FP Stage 4 finite-map comparison completed."

#!/usr/bin/env bash
#SBATCH --job-name=riemann35-fp-stage1
#SBATCH --partition=cpu-preempt
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=riemann35-stage1-%j.out

set -euo pipefail

module purge
if [[ -z "${JULIA_MODULE:-}" ]]; then
    echo "JULIA_MODULE is not set."
    echo "On the Unity login node, run: module spider julia"
    echo "Then submit with, for example:"
    echo "  sbatch --export=ALL,JULIA_MODULE=julia/1.10.5 riemann35_patch/run_unity_stage1.sbatch"
    exit 2
fi
module load "${JULIA_MODULE}"

echo "Node: $(hostname)"
echo "Julia module: ${JULIA_MODULE}"
julia --version
julia -e 'VERSION.major == 1 && VERSION.minor in (9, 10, 11) || error("Riemann35 supports Julia 1.9, 1.10, or 1.11; got $VERSION")'

submit_dir="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
patch="${submit_dir}/riemann35_patch/0001-Add-provisional-cubic-FP-collision-source.patch"
if [[ ! -f "${patch}" ]]; then
    echo "Patch not found: ${patch}"
    echo "Submit this job from the FPCode repository root on branch agent/hyqmom-fp-hybrid."
    exit 3
fi

scratch_root="${SLURM_TMPDIR:-/tmp}"
work_dir="$(mktemp -d "${scratch_root}/riemann35-stage1.XXXXXX")"
repo_dir="${work_dir}/Riemann35-stage1"

echo "Working directory: ${work_dir}"
git clone https://github.com/comp-physics/Riemann35.jl.git "${repo_dir}"
git -C "${repo_dir}" checkout --detach 3ec8735
git -C "${repo_dir}" apply --check "${patch}"
git -C "${repo_dir}" apply "${patch}"
git -C "${repo_dir}" diff --check

export HYQMOM_SKIP_PLOTTING=true
export CI=true

cd "${repo_dir}"
for attempt in 1 2 3; do
    if julia --project=. -e 'using Pkg; Pkg.instantiate()'; then
        break
    fi
    if [[ "${attempt}" -eq 3 ]]; then
        JULIA_PKG_SERVER="" julia --project=. -e 'using Pkg; Pkg.instantiate()'
    else
        echo "Julia package instantiate attempt ${attempt} failed; retrying in 15 seconds"
        sleep 15
    fi
done

julia --project=. -e 'using Riemann35; include("test/test_fp_collision35.jl")'

echo "PASS: Riemann35 cubic FP Stage 1 focused tests completed."

#!/usr/bin/env bash
set -euo pipefail

# Submit from the repository root.  FP_TEST can override the standard Unity
# checkout, and any FP_* variable set by the caller overrides the frozen G0
# default below.
FP_TEST="${FP_TEST:-/project/pi_roohie_umass_edu/github_sync/FPCode-pinn-test}"
cd "$FP_TEST"
mkdir -p slurm_logs

export FP_STAGE2_EPOCHS="${FP_STAGE2_EPOCHS:-30000}"
export FP_STAGE2_LEARNING_RATE="${FP_STAGE2_LEARNING_RATE:-1.0e-4}"
export FP_STAGE2_LR_DECAY_STEPS="${FP_STAGE2_LR_DECAY_STEPS:-10000}"
export FP_STAGE2_LR_DECAY_RATE="${FP_STAGE2_LR_DECAY_RATE:-0.3}"
export FP_N_TIME_BATCH="${FP_N_TIME_BATCH:-12}"
export FP_N_VELOCITY_PER_TIME="${FP_N_VELOCITY_PER_TIME:-4096}"
export FP_PDE_WEIGHT="${FP_PDE_WEIGHT:-1.0}"
export FP_HEAT_FLUX_WEIGHT="${FP_HEAT_FLUX_WEIGHT:-12.0}"
export FP_HEAT_FLUX_RATE_WEIGHT="${FP_HEAT_FLUX_RATE_WEIGHT:-10.0}"
export FP_HEAT_FLUX_RATE_STEP="${FP_HEAT_FLUX_RATE_STEP:-0.01}"
export FP_MASS_WEIGHT="${FP_MASS_WEIGHT:-50.0}"
export FP_MOMENTUM_WEIGHT="${FP_MOMENTUM_WEIGHT:-30.0}"
export FP_ENERGY_WEIGHT="${FP_ENERGY_WEIGHT:-50.0}"
export FP_TAIL_FRACTION="${FP_TAIL_FRACTION:-0.20}"
export FP_TAIL_VARIANCE="${FP_TAIL_VARIANCE:-4.0}"
export FP_FIXED_VELOCITY_QUADRATURE="${FP_FIXED_VELOCITY_QUADRATURE:-1}"
export FP_QUADRATURE_PANELS="${FP_QUADRATURE_PANELS:-4}"
export FP_CHECKPOINT_EVERY="${FP_CHECKPOINT_EVERY:-2500}"
export FP_CHECKPOINT_SWEEP="${FP_CHECKPOINT_SWEEP:-1}"
export FP_REFERENCE_PARTICLES="${FP_REFERENCE_PARTICLES:-500000}"
export FP_REFERENCE_DT="${FP_REFERENCE_DT:-0.0025}"
export FP_REFERENCE_SAVE_EVERY="${FP_REFERENCE_SAVE_EVERY:-20}"
export FP_EVALUATION_SAMPLES="${FP_EVALUATION_SAMPLES:-131072}"
export FP_AXISYMMETRIC_HEAT_FLUX="${FP_AXISYMMETRIC_HEAT_FLUX:-1}"
export FP_ANTITHETIC_HEAT_FLUX_QUADRATURE="${FP_ANTITHETIC_HEAT_FLUX_QUADRATURE:-1}"
export FP_PACKAGE_RESULTS="${FP_PACKAGE_RESULTS:-1}"
export FP_STRICT_GATE="${FP_STRICT_GATE:-1}"

sbatch \
  --array=2 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --exclude="${FP_EXCLUDE_NODES:-gypsum-gpu001,gypsum-gpu011,gypsum-gpu012,gypsum-gpu013,gypsum-gpu015}" \
  --output="$FP_TEST/slurm_logs/fp-pinn-g0-%A_%a.out" \
  --error="$FP_TEST/slurm_logs/fp-pinn-g0-%A_%a.err" \
  FP_PINN/pinn/cubic_homogeneous_3d/slurm/run_stage2_array.sbatch

#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BASE=${STAGE69_DIR:-/project/pi_roohie_umass_edu/github_sync/FPCode-stage69-fixed-qmc/results/riemann35_stage69/fixed_qmc_blind_20260830-214729}
[[ -f "$BASE/stage58_generalization_summary.json" ]] || { echo "Stage69 directory missing: $BASE" >&2; exit 2; }
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$ROOT/results/riemann35_stage70/broad_qmc_refinement_${STAMP}"; mkdir -p "$OUT/logs"
for c in stage57_anchor hot_dense_shifted alternate_weights anisotropic_3d; do cp "$BASE/stage58_${c}.npz" "$OUT/"; cp "$BASE/stage58_${c}_summary.json" "$OUT/"; done
BUNDLE="$OUT/STAGE70_BROAD_QMC_REFINEMENT_RESULTS_${STAMP}.zip"
J=$(sbatch --parsable -p cpu -N 1 -n 1 -c 8 -t 04:30:00 --mem=20G -J m70-broad -o "$OUT/logs/m70-broad_%j.out" -e "$OUT/logs/m70-broad_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage69_fixed_qmc_blind/run_fixed_case.py --case broad_shifted --output '$OUT' --points-per-component 131072 --replicates 8 --workers 8")
C=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 00:10:00 --mem=3G -J m70-coll --dependency=afterok:$J -o "$OUT/logs/m70-coll_%j.out" -e "$OUT/logs/m70-coll_%j.err" --wrap="cd '$ROOT' && python riemann35_patch/stage58_blind_generalization/collect_generalization_gate.py --root '$OUT' --bundle '$BUNDLE' && python riemann35_patch/stage70_broad_qmc_refinement/collect_refinement.py --old-dir '$BASE' --new-dir '$OUT' && sha256sum '$BUNDLE' > '$BUNDLE.sha256.txt'")
echo "STAGE69_DIR=$BASE"; echo "STAGE70_JOB=$J"; echo "COLLECT_JOB=$C"; echo "RESULT_DIR=$OUT"; echo "BUNDLE=$BUNDLE"

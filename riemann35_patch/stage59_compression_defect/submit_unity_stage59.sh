#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/project/pi_roohie_umass_edu/github_sync/FPCode-stage59-compression-defect}
BRANCH=${BRANCH:-stage59-compression-defect-audit}
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${OUT:-$ROOT/results/riemann35_stage59/compression_defect_$STAMP}
mkdir -p "$OUT" "$ROOT/logs"
cd "$ROOT"
git fetch origin "$BRANCH" && git checkout -B "$BRANCH" "origin/$BRANCH"
CASES=(stage57_anchor hot_dense_shifted broad_shifted alternate_weights anisotropic_3d)
JOBS=()
for i in "${!CASES[@]}"; do
 c=${CASES[$i]}
 j=$(sbatch --parsable --job-name="m59-$i" --partition=cpu --cpus-per-task=2 --mem=8G --time=02:00:00 --output="$ROOT/logs/stage59-%j-$i.out" --error="$ROOT/logs/stage59-%j-$i.err" --wrap="cd '$ROOT' && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m riemann35_patch.stage59_compression_defect.run_compression_defect_case --case '$c' --output '$OUT'")
 JOBS+=("$j")
done
DEP=$(IFS=:; echo "${JOBS[*]}")
COLLECT=$(sbatch --parsable --dependency="afterok:$DEP" --job-name=m59-collect --partition=cpu --cpus-per-task=1 --mem=4G --time=00:20:00 --output="$ROOT/logs/stage59-collect-%j.out" --error="$ROOT/logs/stage59-collect-%j.err" --wrap="cd '$ROOT' && python -m riemann35_patch.stage59_compression_defect.collect_compression_defect --input '$OUT' && cd '$OUT' && zip -qr 'STAGE59_COMPRESSION_DEFECT_RESULTS_${STAMP}.zip' . -x '*.zip' && sha256sum 'STAGE59_COMPRESSION_DEFECT_RESULTS_${STAMP}.zip' > 'STAGE59_COMPRESSION_DEFECT_RESULTS_${STAMP}.zip.sha256.txt'")
printf 'STAGE59_JOBS=%s\nCOLLECT_JOB=%s\nRESULT_DIR=%s\n' "${JOBS[*]}" "$COLLECT" "$OUT"

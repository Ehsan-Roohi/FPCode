#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
python -m py_compile riemann35_patch/stage57_persistent_four_population/persistent_mixture.py riemann35_patch/stage75_core_spatial/test_core_promotion.py riemann35_patch/stage75_core_spatial/run_spatial_density_wave.py
COUNT=$(grep -c 'populations.rho \* probability \* (' riemann35_patch/stage57_persistent_four_population/persistent_mixture.py || true)
[[ "$COUNT" -eq 2 ]] || { echo "STAGE75_PREFLIGHT=FAIL rho_jacobian_rows=$COUNT" >&2; exit 2; }
echo "STAGE75_PREFLIGHT=PASS rho_jacobian_rows=$COUNT"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/riemann35_stage75/core_spatial_${STAMP}"
mkdir -p "$OUT/logs"
BUNDLE="$OUT/STAGE75_CORE_SPATIAL_RESULTS_${STAMP}.zip"
JOB=$(sbatch --parsable -p cpu -N 1 -n 1 -c 1 -t 01:30:00 --mem=4G -J m75-core-spatial -o "$OUT/logs/m75_%j.out" -e "$OUT/logs/m75_%j.err" --wrap="set -euo pipefail; cd '$ROOT'; python riemann35_patch/stage75_core_spatial/test_core_promotion.py | tee '$OUT/core_promotion.log'; python riemann35_patch/stage75_core_spatial/run_spatial_density_wave.py --output '$OUT'; python -c \"import zipfile,pathlib; d=pathlib.Path('$OUT'); z=zipfile.ZipFile('$BUNDLE','w',zipfile.ZIP_DEFLATED); [z.write(p,p.relative_to(d)) for p in d.rglob('*') if p.is_file() and p.name != pathlib.Path('$BUNDLE').name and not p.name.endswith('.sha256.txt')]; z.close()\"; sha256sum '$BUNDLE' > '$BUNDLE.sha256.txt'; echo STAGE75_BUNDLE='$BUNDLE'")
echo "STAGE75_JOB=$JOB"
echo "RESULT_DIR=$OUT"
echo "BUNDLE=$BUNDLE"

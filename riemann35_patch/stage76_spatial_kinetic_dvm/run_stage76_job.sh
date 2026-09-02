#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:?ROOT required}
OUT=${OUT:?OUT required}
BUNDLE=${BUNDLE:?BUNDLE required}
cd "$ROOT"
python riemann35_patch/stage76_spatial_kinetic_dvm/run_stage76.py \
  --output "$OUT" \
  --coarse-vcells 41 \
  --fine-vcells 49 \
  | tee "$OUT/stage76.stdout.log"
python - <<'PY' "$OUT" "$BUNDLE"
import hashlib,pathlib,sys,zipfile
out=pathlib.Path(sys.argv[1]); bundle=pathlib.Path(sys.argv[2])
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.resolve()!=bundle.resolve() and not p.name.endswith('.sha256.txt'):
            z.write(p,p.relative_to(out))
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
bundle.with_name(bundle.name+'.sha256.txt').write_text(f'{digest}  {bundle.name}\n')
print(f'STAGE76_BUNDLE={bundle}')
PY

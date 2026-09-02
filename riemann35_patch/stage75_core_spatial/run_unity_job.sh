#!/usr/bin/env bash
set -euo pipefail

ROOT=${STAGE75_ROOT:?STAGE75_ROOT is required}
OUT=${STAGE75_OUT:?STAGE75_OUT is required}
BUNDLE=${STAGE75_BUNDLE:?STAGE75_BUNDLE is required}

cd "$ROOT"
python riemann35_patch/stage75_core_spatial/test_core_promotion.py | tee "$OUT/core_promotion.log"
python riemann35_patch/stage75_core_spatial/run_spatial_density_wave.py --output "$OUT"

python - "$OUT" "$BUNDLE" <<'PY'
import pathlib
import sys
import zipfile

root = pathlib.Path(sys.argv[1])
bundle = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == bundle.resolve() or path.name.endswith(".sha256.txt"):
            continue
        archive.write(path, path.relative_to(root))
PY

sha256sum "$BUNDLE" > "$BUNDLE.sha256.txt"
echo "STAGE75_BUNDLE=$BUNDLE"

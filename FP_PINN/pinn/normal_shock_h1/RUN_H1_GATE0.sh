#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MACH="${FP_H1_MACH:-3}"; OUT="${FP_H1_OUTPUT:-$HERE/outputs/h1_gate0_m${MACH//./p}}"
cd "$HERE"; python -m unittest -v test_shock_physics.py; python run_gate0.py --mach "$MACH" --output "$OUT"
sha256sum "$OUT/metrics.json" "$OUT/profiles.csv" "$OUT/shock_gate0.npz" "$OUT/h1_gate0_physics.png" > "$OUT/SHA256SUMS"; echo "H1_GATE0_OUTPUT=$OUT"

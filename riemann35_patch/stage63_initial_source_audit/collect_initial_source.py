#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

CASES = ('stage57_anchor','hot_dense_shifted','broad_shifted','alternate_weights','anisotropic_3d')


def main() -> None:
    out = Path(sys.argv[1])
    rows = []
    for case in CASES:
        rows.append(json.loads((out / f'stage63_{case}_summary.json').read_text()))
    anchor = next(r for r in rows if r['case'] == 'stage57_anchor')
    aerr = anchor['finest_relative_source_error_all35']
    for r in rows:
        r['ratio_to_anchor'] = r['finest_relative_source_error_all35'] / max(aerr, 1e-16)
    coll = {'schema':'riemann35-stage63-collection-v1','cases':rows}
    (out/'stage63_initial_source_summary.json').write_text(json.dumps(coll, indent=2)+'\n')
    lines = ['# Stage 63 — initial source / finite-time-map audit','',
             '| case | finest all-35 source error | order-2 | order-3 | order-4 | ratio to anchor |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        o=r['finest_relative_source_error_by_order']
        lines.append(f"| {r['case']} | {r['finest_relative_source_error_all35']:.6e} | {o['2']:.6e} | {o['3']:.6e} | {o['4']:.6e} | {r['ratio_to_anchor']:.3f} |")
    lines += ['', 'Interpretation: blind-only source errors that remain large under dt refinement implicate the persistent finite-time source map immediately; similar small errors across all cases would instead point back to accumulated latent/history effects.']
    (out/'STAGE63_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__ == '__main__': main()

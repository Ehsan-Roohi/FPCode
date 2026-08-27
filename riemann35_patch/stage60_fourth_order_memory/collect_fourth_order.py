#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(sys.argv[1]); files=sorted(ROOT.glob('stage60_*_summary.json'))
if not files: raise SystemExit('no Stage60 summaries')
rows=[json.loads(p.read_text()) for p in files]; anchor=next(r for r in rows if r['case']=='stage57_anchor'); a=anchor['rms_fourth_excess']
for r in rows: r['rms_fourth_ratio_to_anchor']=r['rms_fourth_excess']/a if a else float('inf')
summary={'schema':'riemann35-stage60-fourth-order-memory-collection-v1','scientific_scope':'diagnostic only; tests whether Stage-58 blind failure correlates with fourth-order/tail memory erased by Gaussian recompression','qmc_used':False,'cases':rows,'largest_ratio_case':max(rows,key=lambda r:r['rms_fourth_ratio_to_anchor'])['case']}
(ROOT/'stage60_fourth_order_memory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
md=['# Stage 60 — fourth-order / tail-memory audit','','| case | RMS fourth excess | RMS radial | RMS anisotropic | ratio to anchor |','|---|---:|---:|---:|---:|']
for r in rows: md.append(f"| {r['case']} | {r['rms_fourth_excess']:.6e} | {r['rms_radial_excess']:.6e} | {r['rms_anisotropic_excess']:.6e} | {r['rms_fourth_ratio_to_anchor']:.3f} |")
md += ['','Interpretation: blind/anchor ratios clearly above one support missing fourth-order/tail memory as the Stage-58 generalization bottleneck; ratios near or below one reject that hypothesis and point toward population-label/history coupling rather than local cumulant truncation.']
(ROOT/'STAGE60_RESULTS.md').write_text('\n'.join(md)+'\n')
print('\n'.join(md))

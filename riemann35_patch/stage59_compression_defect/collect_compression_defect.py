#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES,ANCHOR_CASE

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); a=p.parse_args(); rows=[]
 for c in CASE_NAMES:
  q=a.input/f'stage59_{c}_summary.json'
  if not q.exists(): raise FileNotFoundError(q)
  rows.append(json.loads(q.read_text()))
 anchor=next(r for r in rows if r['case']==ANCHOR_CASE); base=max(anchor['rms_discarded_third'],1e-30)
 for r in rows: r['rms_third_ratio_to_anchor']=r['rms_discarded_third']/base
 out={'schema':'riemann35-stage59-compression-defect-collection-v1','scientific_scope':'diagnostic only; tests whether Stage-58 failure is consistent with third-order information erased by per-population Gaussian recompression','qmc_used':False,'cases':rows,'largest_ratio_case':max(rows,key=lambda r:r['rms_third_ratio_to_anchor'])['case']}
 (a.input/'stage59_compression_defect_summary.json').write_text(json.dumps(out,indent=2)+'\n')
 md=['# Stage 59 — Gaussian recompression defect audit','','This is a structural diagnostic, not a new generalization claim. QMC is not used.','','| case | RMS discarded third | RMS trace-free | RMS heat-flux | ratio to anchor |','|---|---:|---:|---:|---:|']
 for r in rows: md.append(f"| {r['case']} | {r['rms_discarded_third']:.6e} | {r['rms_discarded_trace_free']:.6e} | {r['rms_discarded_heat_flux']:.6e} | {r['rms_third_ratio_to_anchor']:.3f} |")
 md += ['','Interpretation: a large blind/anchor ratio supports the hypothesis that Stage 58 fails because each mapped labelled subpopulation becomes non-Gaussian and Stage 57 erases that within-population third-order information. If ratios remain near one, the next audit should target fourth-order/nonlocal memory instead.']
 (a.input/'STAGE59_RESULTS.md').write_text('\n'.join(md)+'\n'); print('\n'.join(md))
if __name__=='__main__': main()

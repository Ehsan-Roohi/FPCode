#!/usr/bin/env python3
from __future__ import annotations
import json, sys, zipfile, hashlib
from pathlib import Path
CASES=('stage57_anchor','hot_dense_shifted','broad_shifted','alternate_weights','anisotropic_3d')

def main():
 out=Path(sys.argv[1]); rows=[]
 for c in CASES: rows.append(json.loads((out/f'stage64_{c}_summary.json').read_text()))
 report=['# Stage 64 — generator-consistent fourth-order map prototype','', '| case | baseline RMS | corrected RMS | improvement | final corrected | max source err |', '|---|---:|---:|---:|---:|---:|']
 for r in rows: report.append(f"| {r['case']} | {r['baseline_rms_relative_error']:.6e} | {r['corrected_rms_relative_error']:.6e} | {r['improvement_factor']:.3f} | {r['corrected_final_relative_error']:.6e} | {r['max_fourth_source_error_after_projection']:.3e} |")
 report += ['', 'Interpretation: improvement factors above one, especially for broad_shifted, support replacing the Stage-57 fourth-order finite-time increment by a generator-consistent projection. Factors near/below one reject this repair as the generalization fix.']
 (out/'STAGE64_RESULTS.md').write_text('\n'.join(report)+'\n')
 summary={'schema':'riemann35-stage64-collection-v1','cases':rows}; (out/'stage64_generator_consistent_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 z=out/f"STAGE64_GENERATOR_CONSISTENT_RESULTS_{out.name.split('_')[-1]}.zip"
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
  for p in out.iterdir():
   if p.is_file() and p!=z and not p.name.endswith('.sha256.txt'): f.write(p,p.name)
 h=hashlib.sha256(z.read_bytes()).hexdigest(); (Path(str(z)+'.sha256.txt')).write_text(f'{h}  {z.name}\n')
 print('\n'.join(report))
if __name__=='__main__': main()

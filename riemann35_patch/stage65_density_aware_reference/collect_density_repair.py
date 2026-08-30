#!/usr/bin/env python3
from __future__ import annotations
import json, sys, zipfile, hashlib
from pathlib import Path
CASES=('stage57_anchor','hot_dense_shifted','broad_shifted','alternate_weights','anisotropic_3d')
def main():
    out=Path(sys.argv[1]); rows=[]
    for c in CASES: rows.append(json.loads((out/f'stage65_{c}_summary.json').read_text()))
    summary={'schema':'riemann35-stage65-density-aware-collection-v1','scientific_scope':'repairs the frozen Stage-58 QMC density normalization only; no new fitting and no new QMC','cases':rows}
    (out/'stage65_density_aware_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Stage 65 — density-aware repair of frozen Stage-58 QMC reference','', '| case | true rho | frozen QMC rho | old RMS | corrected RMS | reduction |','|---|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['case']} | {r['true_density']:.3f} | {r['frozen_qmc_density']:.3f} | {r['old_rms_relative_error']:.6e} | {r['corrected_rms_relative_error']:.6e} | {r['error_reduction_factor']:.2f} |")
    lines += ['', 'Interpretation: if the large blind errors collapse after restoring the conserved density, the Stage-58 generalization failure was a QMC reference normalization artifact rather than a physical failure of the persistent closure.']
    (out/'STAGE65_RESULTS.md').write_text('\n'.join(lines)+'\n')
    stamp=out.name.split('_')[-1]; zp=out/f'STAGE65_DENSITY_AWARE_RESULTS_{stamp}.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.glob('stage65_*_summary.json')): z.write(p,p.name)
        z.write(out/'stage65_density_aware_summary.json','stage65_density_aware_summary.json'); z.write(out/'STAGE65_RESULTS.md','STAGE65_RESULTS.md')
    h=hashlib.sha256(zp.read_bytes()).hexdigest(); (Path(str(zp)+'.sha256.txt')).write_text(f'{h}  {zp.name}\n')
    print((out/'STAGE65_RESULTS.md').read_text())
if __name__=='__main__': main()

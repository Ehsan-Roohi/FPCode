#!/usr/bin/env python3
from __future__ import annotations
import json, sys, zipfile, hashlib
from pathlib import Path
CASES=('stage57_anchor','hot_dense_shifted','broad_shifted','alternate_weights','anisotropic_3d')

def main():
    out=Path(sys.argv[1]).resolve()
    rows=[]
    for case in CASES:
        p=out/f'stage61_{case}_summary.json'
        if not p.exists(): raise FileNotFoundError(p)
        rows.append(json.loads(p.read_text()))
    summary={'schema':'riemann35-stage61-error-onset-collection-v1','scientific_scope':'uses frozen Stage-58 QMC/persistent trajectories to identify which retained moment order first departs beyond QMC uncertainty; no new fitting and no new QMC','cases':rows}
    (out/'stage61_error_onset_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Stage 61 — error-onset hierarchy audit','', '| case | order-2 onset | order-3 onset | order-4 onset | earliest |', '|---|---:|---:|---:|---|']
    for r in rows:
        vals=[]
        for k in ('2','3','4'):
            v=r['orders'][k]['onset_time_over_tau']; vals.append('none' if v is None else f'{v:.4f}')
        lines.append(f"| {r['case']} | {vals[0]} | {vals[1]} | {vals[2]} | {r['earliest_diverging_order'] or 'none'} |")
    lines += ['', 'Interpretation: systematic early order-4 onset would support a local higher-moment bottleneck. If order-4 does not lead while lower orders later diverge, the Stage-58 failure is more consistent with hidden population-history / non-Markovian closure dependence than with simple local cumulant truncation.']
    (out/'STAGE61_RESULTS.md').write_text('\n'.join(lines)+'\n')
    stamp=out.name.split('_')[-1] if '_' in out.name else 'results'
    z=out/f'STAGE61_ERROR_ONSET_RESULTS_{stamp}.zip'
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as zz:
        for p in sorted(out.glob('stage61_*_summary.json')): zz.write(p,p.name)
        zz.write(out/'STAGE61_RESULTS.md','STAGE61_RESULTS.md')
    h=hashlib.sha256(z.read_bytes()).hexdigest()
    (out/(z.name+'.sha256.txt')).write_text(f'{h}  {z.name}\n')
    print('\n'.join(lines))
if __name__=='__main__': main()

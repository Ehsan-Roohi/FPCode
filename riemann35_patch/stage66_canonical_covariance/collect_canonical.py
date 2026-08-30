#!/usr/bin/env python3
import json,sys,zipfile,hashlib
from pathlib import Path
out=Path(sys.argv[1]); rows=[]
for p in sorted(out.glob('stage66_*_summary.json')): rows.append(json.loads(p.read_text()))
summary={'schema':'riemann35-stage66-collection-v1','cases':rows}; (out/'stage66_canonical_covariance_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Stage 66 — canonical-variable covariance audit','','| case | initial defect | max defect | final defect |','|---|---:|---:|---:|']
for r in rows: lines.append(f"| {r['case']} | {r['initial_relative_covariance_defect']:.6e} | {r['max_relative_covariance_defect']:.6e} | {r['final_relative_covariance_defect']:.6e} |")
lines += ['','Interpretation: growth from roundoff demonstrates that direct Stage-57 evolution is not covariant under the center/thermal-scale transform.']
(out/'STAGE66_RESULTS.md').write_text('\n'.join(lines)+'\n')
z=out/f"STAGE66_CANONICAL_COVARIANCE_RESULTS_{out.name.split('_')[-1]}.zip"
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as q:
 for p in out.glob('stage66_*_summary.json'): q.write(p,p.name)
 q.write(out/'STAGE66_RESULTS.md','STAGE66_RESULTS.md')
h=hashlib.sha256(z.read_bytes()).hexdigest(); (Path(str(z)+'.sha256.txt')).write_text(f'{h}  {z.name}\n')
print((out/'STAGE66_RESULTS.md').read_text())

#!/usr/bin/env python3
import json,sys,zipfile,hashlib
from pathlib import Path
out=Path(sys.argv[1]); rows=[]
for p in sorted(out.glob('stage67_*_summary.json')): rows.append(json.loads(p.read_text()))
summary={'schema':'riemann35-stage67-collection-v1','cases':rows}; (out/'stage67_rotation_covariance_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Stage 67 — rotation covariance audit','','| case | worst max defect |','|---|---:|']
for r in rows: lines.append(f"| {r['case']} | {r['worst_max_defect']:.6e} |")
lines += ['','Interpretation: roundoff-level defects support rotational covariance; larger defects identify a rotation-covariance failure in the persistent closure.']
(out/'STAGE67_RESULTS.md').write_text('\n'.join(lines)+'\n')
z=out/f"STAGE67_ROTATION_COVARIANCE_RESULTS_{out.name.split('_')[-1]}.zip"
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as q:
 for p in out.glob('stage67_*_summary.json'): q.write(p,p.name)
 q.write(out/'STAGE67_RESULTS.md','STAGE67_RESULTS.md')
h=hashlib.sha256(z.read_bytes()).hexdigest(); Path(str(z)+'.sha256.txt').write_text(f'{h}  {z.name}\n')
print((out/'STAGE67_RESULTS.md').read_text())

#!/usr/bin/env python3
import json,sys,zipfile,hashlib
from pathlib import Path
out=Path(sys.argv[1]); rows=[json.loads(p.read_text()) for p in sorted(out.glob('stage68_*_summary.json'))]
summary={'schema':'riemann35-stage68-collection-v1','cases':rows}; (out/'stage68_qmc_density_continuity_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Stage 68 — QMC density-continuity repair','','| case | rho true | rho t0 | rho first | old RMS | repaired RMS | reduction |','|---|---:|---:|---:|---:|---:|---:|']
for r in rows: lines.append(f"| {r['case']} | {r['true_density']:.6f} | {r['qmc_mass_t0_mean']:.6f} | {r['qmc_mass_first_positive_time_mean']:.6f} | {r['old_rms_relative_error']:.6e} | {r['repaired_rms_relative_error']:.6e} | {r['error_reduction_factor']:.3f} |")
lines += ['','Interpretation: a jump from the true density at t=0 to unity at the first positive time proves a Stage-58 QMC moment-extraction normalization bug. Large error reduction after per-snapshot mass repair means the prior blind failure was dominated by that harness defect.']
(out/'STAGE68_RESULTS.md').write_text('\n'.join(lines)+'\n')
z=out/f"STAGE68_QMC_DENSITY_CONTINUITY_RESULTS_{out.name.split('_')[-1]}.zip"
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as q:
 for p in out.glob('stage68_*_summary.json'): q.write(p,p.name)
 q.write(out/'STAGE68_RESULTS.md','STAGE68_RESULTS.md')
h=hashlib.sha256(z.read_bytes()).hexdigest(); Path(str(z)+'.sha256.txt').write_text(f'{h}  {z.name}\n'); print((out/'STAGE68_RESULTS.md').read_text())

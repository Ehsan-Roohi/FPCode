#!/usr/bin/env python3
from __future__ import annotations
import json, sys, zipfile, hashlib
from pathlib import Path
CASES=('stage57_anchor','hot_dense_shifted','broad_shifted','alternate_weights','anisotropic_3d')
def main():
 out=Path(sys.argv[1]); rows=[]
 for c in CASES: rows.append(json.loads((out/f'stage62_{c}_summary.json').read_text()))
 lines=['# Stage 62 — latent-decomposition equivalence audit','','| case | initial moment mismatch | latent distance | final future divergence | max future divergence |','|---|---:|---:|---:|---:|']
 for r in rows: lines.append(f"| {r['case']} | {r['initial_relative_moment_mismatch']:.3e} | {r['latent_parameter_rms_distance']:.3e} | {r['final_relative_future_divergence']:.3e} | {r['max_relative_future_divergence']:.3e} |")
 lines += ['','Interpretation: a tightly moment-matched alternate decomposition that subsequently diverges demonstrates latent-decomposition dependence of the persistent closure.']
 (out/'STAGE62_RESULTS.md').write_text('\n'.join(lines)+'\n')
 (out/'stage62_latent_equivalence_summary.json').write_text(json.dumps({'schema':'riemann35-stage62-collection-v1','cases':rows},indent=2)+'\n')
 stamp=out.name.split('_')[-1]; z=out/f'STAGE62_LATENT_EQUIVALENCE_RESULTS_{stamp}.zip'
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
  for p in out.iterdir():
   if p.is_file() and p!=z and not p.name.endswith('.sha256.txt'): f.write(p,p.name)
 h=hashlib.sha256(z.read_bytes()).hexdigest(); (Path(str(z)+'.sha256.txt')).write_text(f'{h}  {z.name}\n')
 print((out/'STAGE62_RESULTS.md').read_text())
if __name__=='__main__': main()

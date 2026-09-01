#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from riemann35_patch.stage71_harder_unseen.hard_cases import CASE_NAMES

def invmax(s): return max(float(s['maximum_mass_drift']),float(s['maximum_momentum_drift']),float(s['maximum_energy_trace_drift']))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--old-dir',type=Path,required=True); p.add_argument('--new-dir',type=Path,required=True); a=p.parse_args()
 old_gate=json.loads((a.old_dir/'stage71_generalization_summary.json').read_text()); new_gate=json.loads((a.new_dir/'stage71_generalization_summary.json').read_text()); rows=[]
 for n in CASE_NAMES:
  o=old_gate['case_results'][n]; f=new_gate['case_results'][n]; os=json.loads((a.old_dir/f'stage71_{n}_summary.json').read_text()); fs=json.loads((a.new_dir/f'stage71_{n}_summary.json').read_text()); od=os['persistent_fine']['diagnostics']; fd=fs['persistent_fine']['diagnostics']
  rows.append({'case':n,'rho':fs['configuration']['density'],'old_pass':o['pass'],'fixed_pass':f['pass'],'old_heat_flux_error':o['errors']['heat_flux'],'fixed_heat_flux_error':f['errors']['heat_flux'],'old_third_error':o['errors']['third_tensor'],'fixed_third_error':f['errors']['third_tensor'],'old_tracefree_error':o['errors']['trace_free'],'fixed_tracefree_error':f['errors']['trace_free'],'old_projection_fraction':od['minimum_projection_fraction'],'fixed_projection_fraction':fd['minimum_projection_fraction'],'old_projection_residual':od['maximum_projection_residual'],'fixed_projection_residual':fd['maximum_projection_residual'],'old_invariant_drift':invmax(os['persistent_fine']['invariants']),'fixed_invariant_drift':invmax(fs['persistent_fine']['invariants'])})
 summary={'schema':'riemann35-stage72-density-jacobian-fix-v1','hypothesis':'Stage57 Newton Jacobian omitted rho in raw third-moment rows. Restoring rho should remove density-dependent projection failure without changing closure physics, QMC reference, or gates.','old_qualification_pass':old_gate['qualification_pass'],'fixed_qualification_pass':new_gate['qualification_pass'],'cases':rows}; (a.new_dir/'stage72_density_jacobian_fix_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Stage 72 — density-consistent projection Jacobian audit','',f"Old Stage71 qualification: **{'PASS' if old_gate['qualification_pass'] else 'FAIL'}**",f"After rho-Jacobian fix: **{'PASS' if new_gate['qualification_pass'] else 'FAIL'}**",'', '| case | rho | old q err | fixed q err | old third | fixed third | old proj frac | fixed proj frac | old proj resid | fixed proj resid | fixed result |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
 for r in rows:
  lines.append(f"| {r['case']} | {r['rho']:.3f} | {100*r['old_heat_flux_error']:.3f}% | {100*r['fixed_heat_flux_error']:.3f}% | {100*r['old_third_error']:.3f}% | {100*r['fixed_third_error']:.3f}% | {r['old_projection_fraction']:.3e} | {r['fixed_projection_fraction']:.3e} | {r['old_projection_residual']:.3e} | {r['fixed_projection_residual']:.3e} | {'PASS' if r['fixed_pass'] else 'FAIL'} |")
 (a.new_dir/'STAGE72_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()

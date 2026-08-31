#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux import central_third_components,irreducible_decomposition,normalized_component_rmse,relative_history_error,replicate_spread,symmetric_tensor
from riemann35_patch.stage57_persistent_four_population.collect_persistent_gate import component_qualification
from riemann35_patch.stage71_harder_unseen.hard_cases import CASE_NAMES,registry_manifest

def args():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--bundle',type=Path,required=True); p.add_argument('--reference-spread-gate',type=float,default=0.02); p.add_argument('--time-change-gate',type=float,default=0.01); p.add_argument('--heat-flux-gate',type=float,default=0.01); p.add_argument('--third-gate',type=float,default=0.03); p.add_argument('--tracefree-gate',type=float,default=0.05); p.add_argument('--component-gate',type=float,default=0.03); p.add_argument('--component-relative-allowance',type=float,default=0.20); p.add_argument('--source-gate',type=float,default=1e-9); p.add_argument('--projection-gate',type=float,default=1e-8); p.add_argument('--invariant-gate',type=float,default=2e-8); return p.parse_args()

def load(root,name):
 npz=root/f'stage71_{name}.npz'; js=root/f'stage71_{name}_summary.json'
 if not npz.is_file() or not js.is_file(): return None
 a=np.load(npz); return {'qmc':np.asarray(a['qmc_histories'],float),'coarse':np.asarray(a['persistent_coarse_histories'],float),'fine':np.asarray(a['persistent_fine_histories'],float),'summary':json.loads(js.read_text())}

def derived(h):
 c=np.asarray([central_third_components(x) for x in h]); q,carry,tf=irreducible_decomposition(c); return {'components':c,'heat_flux':q,'trace_free':tf,'full_tensor':symmetric_tensor(c)}

def maxinv(s): return max(float(s['maximum_mass_drift']),float(s['maximum_momentum_drift']),float(s['maximum_energy_trace_drift']))

def jsonable(v):
 if isinstance(v,np.generic): return v.item()
 if isinstance(v,np.ndarray): return v.tolist()
 if isinstance(v,dict): return {k:jsonable(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [jsonable(x) for x in v]
 return v

def evaluate(name,d,a):
 q=derived(d['qmc']); co=derived(d['coarse']); fi=derived(d['fine']); ref={k:np.mean(v,axis=0) for k,v in q.items()}; cand={k:np.mean(v,axis=0) for k,v in fi.items()}; cc={k:np.mean(v,axis=0) for k,v in co.items()}
 errors={'heat_flux':relative_history_error(cand['heat_flux'],ref['heat_flux']),'third_tensor':relative_history_error(cand['full_tensor'],ref['full_tensor']),'trace_free':relative_history_error(cand['trace_free'],ref['trace_free']),'component_normalized_rmse':normalized_component_rmse(cand['components'],ref['components'])}
 mc=float(np.max(errors['component_normalized_rmse'])); tc=relative_history_error(cand['full_tensor'],cc['full_tensor']); rows=component_qualification(q['components'],cand['components'],a.component_relative_allowance); mr=max(float(r['qualification_ratio']) for r in rows); spread=replicate_spread(q['full_tensor']); s=d['summary']; fd=s['persistent_fine']['diagnostics']; ini=s['initial_audit']; reg=registry_manifest(); flags=s['controls']
 gates={'fingerprint_frozen':s['case_fingerprint']==reg['case_fingerprints'][name]==s['registry']['case_fingerprints'][name],'no_qmc_case_design':flags.get('qmc_used_to_define_case') is False,'no_closure_refit':flags.get('closure_parameters_refit') is False,'density_preserving_reference':flags.get('density_preserving_reference') is True,'initial_constraints':max(float(ini['mass_error']),float(ini['bulk_velocity_error']),float(ini['energy_trace_error']))<1e-10,'qmc_scramble_spread':spread<a.reference_spread_gate,'time_change':tc<a.time_change_gate,'collision_invariants':max(maxinv(s['qmc']['invariants']),maxinv(s['persistent_coarse']['invariants']),maxinv(s['persistent_fine']['invariants']))<a.invariant_gate,'initial_source_and_tail_exactness':max(float(fd['initial_moment_relative_residual']),float(fd['initial_tail_relative_error']),float(fd['initial_third_source_relative_error']))<a.source_gate,'H2_realizability':float(fd['minimum_H2_margin'])>=-5e-13,'positive_weights':float(fd['minimum_weight'])>0,'positive_covariances':float(fd['minimum_covariance_eigenvalue'])>0,'full_heat_flux_projection':float(fd['minimum_projection_fraction'])>=1-2e-13,'projection_residual':float(fd['maximum_projection_residual'])<a.projection_gate,'compact_state':int(fd['persistent_state_scalars'])<=41,'heat_flux_error':float(errors['heat_flux'])<a.heat_flux_gate,'full_third_error':float(errors['third_tensor'])<a.third_gate,'tracefree_error':float(errors['trace_free'])<a.tracefree_gate,'component_normalized_error':mc<a.component_gate,'component_sem_aware_error':mr<1.0}
 return {'case':name,'pass':all(gates.values()),'gates':gates,'qmc_scramble_spread':spread,'time_change':tc,'errors':errors,'maximum_component_normalized_rmse':mc,'maximum_component_qualification_ratio':mr}

def main():
 a=args(); a.root.mkdir(parents=True,exist_ok=True); loaded={n:load(a.root,n) for n in CASE_NAMES}; missing=[n for n,v in loaded.items() if v is None]; results={n:evaluate(n,v,a) for n,v in loaded.items() if v is not None}; passed=not missing and all(r['pass'] for r in results.values())
 summary={'schema':'riemann35-stage71-harder-unseen-gate-v1','scientific_scope':'prospective harder unseen homogeneous cubic-FP generalization test using density-preserving QMC internal reference','registry':registry_manifest(),'missing_cases':missing,'qualification_pass':passed,'thresholds':{'reference_spread':a.reference_spread_gate,'time_change':a.time_change_gate,'heat_flux':a.heat_flux_gate,'third_tensor':a.third_gate,'trace_free':a.tracefree_gate,'component':a.component_gate},'case_results':results}; (a.root/'stage71_generalization_summary.json').write_text(json.dumps(jsonable(summary),indent=2,allow_nan=True)+'\n')
 with (a.root/'stage71_case_metrics.csv').open('w',newline='') as f:
  fields=['case','pass','qmc_scramble_spread','time_change','heat_flux_error','third_tensor_error','trace_free_error','maximum_component_normalized_rmse','maximum_component_qualification_ratio']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  for n in CASE_NAMES:
   if n not in results: continue
   r=results[n]; w.writerow({'case':n,'pass':r['pass'],'qmc_scramble_spread':r['qmc_scramble_spread'],'time_change':r['time_change'],'heat_flux_error':r['errors']['heat_flux'],'third_tensor_error':r['errors']['third_tensor'],'trace_free_error':r['errors']['trace_free'],'maximum_component_normalized_rmse':r['maximum_component_normalized_rmse'],'maximum_component_qualification_ratio':r['maximum_component_qualification_ratio']})
 lines=['# Stage 71 — harder truly-unseen blind generalization','',f"Qualification objective: **{'PASS' if passed else 'FAIL'}**",'', 'Cases, closure, and thresholds were frozen before Stage71 QMC evaluation.','', '| Case | q error | Full third | Trace-free | Time change | QMC spread | Result |','|---|---:|---:|---:|---:|---:|---:|']
 for n in CASE_NAMES:
  r=results.get(n)
  if r is None: lines.append(f'| {n} | -- | -- | -- | -- | -- | FAIL |'); continue
  lines.append(f"| {n} | {100*r['errors']['heat_flux']:.3f}% | {100*r['errors']['third_tensor']:.3f}% | {100*r['errors']['trace_free']:.3f}% | {100*r['time_change']:.3f}% | {100*r['qmc_scramble_spread']:.3f}% | {'PASS' if r['pass'] else 'FAIL'} |")
 (a.root/'STAGE71_RESULTS.md').write_text('\n'.join(lines)+'\n')
 a.bundle.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in sorted(a.root.rglob('*')):
   if p.is_file() and p.resolve()!=a.bundle.resolve(): z.write(p,arcname=p.relative_to(a.root))
 digest=hashlib.sha256(a.bundle.read_bytes()).hexdigest(); a.bundle.with_name(a.bundle.name+'.sha256.txt').write_text(f'{digest}  {a.bundle.name}\n'); print(f'[stage71] qualification_pass={passed} bundle={a.bundle}',flush=True)
if __name__=='__main__': main()

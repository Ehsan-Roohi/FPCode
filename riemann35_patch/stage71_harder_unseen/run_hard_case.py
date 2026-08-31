#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from hyqmom_fp import moments_35_from_particles, positive_microstate_from_components, qmc_cubic_fp_step, realizability_margin_35
from riemann35_patch.stage55_closure_source_audit.run_closure_method import _qmc_sample,_invariants
from riemann35_patch.stage57_persistent_four_population.run_persistent_method import _run_persistent_candidate
from riemann35_patch.stage71_harder_unseen.hard_cases import CASE_NAMES,hard_case,registry_manifest

def iratio(a,b):
 n=int(round(a/b))
 if n<1 or not np.isclose(n*b,a,rtol=0,atol=2e-13): raise ValueError('time ratio must be integer')
 return n

def sample_steps(tf,si,dt):
 n=iratio(tf,dt); e=iratio(si,dt); s=tuple([0,*range(e,n+1,e)]); return s if s[-1]==n else (*s,n)

def jsonable(v):
 if isinstance(v,np.generic): return v.item()
 if isinstance(v,np.ndarray): return v.tolist()
 if isinstance(v,dict): return {k:jsonable(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [jsonable(x) for x in v]
 return v

def run_qmc(task):
 components,replicate,ppc,dt,steps,sample_steps_,tau,pr,seed=task
 micro,target,projection=positive_microstate_from_components(components,points_per_component=ppc,seed=seed,provenance=f'stage71-density-preserving-qmc-{replicate}')
 rho=float(target[0]); nodes=micro.velocities.copy(); weights=micro.weights.copy(); histories=[target.copy()]
 source,tail=_qmc_sample(target,nodes,weights,tau,pr); sources=[source]; tails=[tail]; ss=set(sample_steps_[1:]); mm=0.0; me=0.0; start=time.perf_counter()
 for step in range(1,steps+1):
  nodes,d=qmc_cubic_fp_step(nodes,weights,dt=dt,tau=tau,seed=seed+1000003+104729*step,prandtl=pr)
  mm=max(mm,d.momentum_drift); me=max(me,abs(d.energy_drift))
  if step in ss:
   moments=moments_35_from_particles(nodes,weights,rho=rho)
   source,tail=_qmc_sample(moments,nodes,weights,tau,pr); histories.append(moments); sources.append(source); tails.append(tail)
  if step==steps or step%max(steps//8,1)==0: print(f'[stage71] qmc replicate={replicate} step={step}/{steps}',flush=True)
 return {'histories':np.asarray(histories),'sources':np.asarray(sources),'tails':np.asarray(tails),'minimum_weight':float(np.min(weights)),'projection_relative_residual':projection.relative_moment_residual,'minimum_probability':projection.minimum_probability,'maximum_momentum_drift':mm,'maximum_energy_drift':me,'minimum_H2_margin':float(min(realizability_margin_35(x) for x in histories)),'elapsed_seconds':time.perf_counter()-start,'density_preserved':rho}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--case',choices=CASE_NAMES,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--qmc-dt',type=float,default=3.125e-4); p.add_argument('--coarse-dt',type=float,default=6.25e-4); p.add_argument('--fine-dt',type=float,default=3.125e-4); p.add_argument('--final-time',type=float,default=1.0); p.add_argument('--sample-interval',type=float,default=2.5e-2); p.add_argument('--tau',type=float,default=1.0); p.add_argument('--prandtl',type=float,default=2/3); p.add_argument('--points-per-component',type=int,default=131072); p.add_argument('--replicates',type=int,default=8); p.add_argument('--workers',type=int,default=8); p.add_argument('--seed',type=int,default=20260831); p.add_argument('--quadrature-nodes',type=int,default=5); a=p.parse_args()
 case=hard_case(a.case); off=CASE_NAMES.index(a.case)*104729001; qs=iratio(a.final_time,a.qmc_dt); qss=sample_steps(a.final_time,a.sample_interval,a.qmc_dt); cs=iratio(a.final_time,a.coarse_dt); css=sample_steps(a.final_time,a.sample_interval,a.coarse_dt); fs=iratio(a.final_time,a.fine_dt); fss=sample_steps(a.final_time,a.sample_interval,a.fine_dt)
 tasks=[(case.components,r,a.points_per_component,a.qmc_dt,qs,qss,a.tau,a.prandtl,a.seed+off+15485863*r) for r in range(a.replicates)]
 with ProcessPoolExecutor(max_workers=min(a.workers,a.replicates)) as ex: qr=list(ex.map(run_qmc,tasks))
 coarse=_run_persistent_candidate(case.components,dt=a.coarse_dt,steps=cs,sample_steps=css,tau=a.tau,prandtl=a.prandtl,quadrature_nodes=a.quadrature_nodes)
 fine=_run_persistent_candidate(case.components,dt=a.fine_dt,steps=fs,sample_steps=fss,tau=a.tau,prandtl=a.prandtl,quadrature_nodes=a.quadrature_nodes)
 qh=np.asarray([x['histories'] for x in qr]); ch=np.asarray(coarse['histories'])[None,...]; fh=np.asarray(fine['histories'])[None,...]; times=np.asarray(qss)*a.qmc_dt/a.tau; a.output.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output/f'stage71_{a.case}.npz',times=times,qmc_histories=qh,persistent_coarse_histories=ch,persistent_fine_histories=fh)
 def cand(x):
  h=np.asarray(x['histories']); return {'invariants':_invariants(h[None,...]),'diagnostics':{k:v for k,v in x.items() if k!='histories'}}
 summary={'schema':'riemann35-stage71-hard-blind-case-v1','case':a.case,'role':case.role,'case_fingerprint':case.fingerprint,'registry':registry_manifest(),'configuration':case.configuration,'initial_audit':case.audit,'controls':{'qmc_dt_over_tau':a.qmc_dt/a.tau,'coarse_dt_over_tau':a.coarse_dt/a.tau,'fine_dt_over_tau':a.fine_dt/a.tau,'final_time_over_tau':a.final_time/a.tau,'sample_interval_over_tau':a.sample_interval/a.tau,'points_per_component':a.points_per_component,'replicates':a.replicates,'prandtl':a.prandtl,'quadrature_nodes_per_population':a.quadrature_nodes,'qmc_used_to_define_case':False,'closure_parameters_refit':False,'density_preserving_reference':True},'qmc':{'invariants':_invariants(qh),'replicate_diagnostics':[{k:v for k,v in x.items() if k not in ('histories','sources','tails')} for x in qr]},'persistent_coarse':cand(coarse),'persistent_fine':cand(fine)}
 (a.output/f'stage71_{a.case}_summary.json').write_text(json.dumps(jsonable(summary),indent=2,allow_nan=True)+'\n')
 print(json.dumps({'case':a.case,'fingerprint':case.fingerprint,'rho':float(case.moments[0]),'qmc_replicates':len(qr)},indent=2))
if __name__=='__main__': main()

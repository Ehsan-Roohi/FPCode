#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from riemann35_patch.stage55_closure_source_audit.run_closure_method import _run_qmc_replicate,_invariants
from riemann35_patch.stage57_persistent_four_population.run_persistent_method import _run_persistent_candidate
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES,blind_case,registry_manifest

def iratio(a,b):
 n=int(round(a/b));
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

def fix_density(result,rho):
 h=np.asarray(result['histories'],float).copy(); masses=h[:,0].copy(); factors=rho/masses; h*=factors[:,None]; result=dict(result); result['histories']=h; result['density_before_repair']=masses.tolist(); result['density_after_repair']=h[:,0].tolist(); return result

def main():
 p=argparse.ArgumentParser(); p.add_argument('--case',choices=CASE_NAMES,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--qmc-dt',type=float,default=3.125e-4); p.add_argument('--coarse-dt',type=float,default=6.25e-4); p.add_argument('--fine-dt',type=float,default=3.125e-4); p.add_argument('--final-time',type=float,default=1.0); p.add_argument('--sample-interval',type=float,default=2.5e-2); p.add_argument('--tau',type=float,default=1.0); p.add_argument('--prandtl',type=float,default=2/3); p.add_argument('--points-per-component',type=int,default=65536); p.add_argument('--replicates',type=int,default=8); p.add_argument('--workers',type=int,default=8); p.add_argument('--seed',type=int,default=20260826); p.add_argument('--quadrature-nodes',type=int,default=5); a=p.parse_args()
 case=blind_case(a.case); rho=float(case.moments[0]); off=CASE_NAMES.index(a.case)*104729001
 qs=iratio(a.final_time,a.qmc_dt); qss=sample_steps(a.final_time,a.sample_interval,a.qmc_dt); cs=iratio(a.final_time,a.coarse_dt); css=sample_steps(a.final_time,a.sample_interval,a.coarse_dt); fs=iratio(a.final_time,a.fine_dt); fss=sample_steps(a.final_time,a.sample_interval,a.fine_dt)
 tasks=[(case.components,r,a.points_per_component,a.qmc_dt,qs,qss,a.tau,a.prandtl,a.seed+off+15485863*r) for r in range(a.replicates)]
 with ProcessPoolExecutor(max_workers=min(a.workers,a.replicates)) as ex: qr=list(ex.map(_run_qmc_replicate,tasks))
 qr=[fix_density(x,rho) for x in qr]
 coarse=_run_persistent_candidate(case.components,dt=a.coarse_dt,steps=cs,sample_steps=css,tau=a.tau,prandtl=a.prandtl,quadrature_nodes=a.quadrature_nodes)
 fine=_run_persistent_candidate(case.components,dt=a.fine_dt,steps=fs,sample_steps=fss,tau=a.tau,prandtl=a.prandtl,quadrature_nodes=a.quadrature_nodes)
 qh=np.asarray([x['histories'] for x in qr]); ch=np.asarray(coarse['histories'])[None,...]; fh=np.asarray(fine['histories'])[None,...]; times=np.asarray(qss)*a.qmc_dt/a.tau
 a.output.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output/f'stage58_{a.case}.npz',times=times,qmc_histories=qh,persistent_coarse_histories=ch,persistent_fine_histories=fh)
 def cand(x):
  h=np.asarray(x['histories']); return {'invariants':_invariants(h[None,...]),'diagnostics':{k:v for k,v in x.items() if k!='histories'}}
 summary={'schema':'riemann35-stage58-generalization-case-v1','case':a.case,'role':case.role,'case_fingerprint':case.fingerprint,'registry':registry_manifest(),'configuration':case.configuration,'initial_audit':case.audit,'controls':{'qmc_dt_over_tau':a.qmc_dt/a.tau,'coarse_dt_over_tau':a.coarse_dt/a.tau,'fine_dt_over_tau':a.fine_dt/a.tau,'final_time_over_tau':a.final_time/a.tau,'sample_interval_over_tau':a.sample_interval/a.tau,'points_per_component':a.points_per_component,'replicates':a.replicates,'prandtl':a.prandtl,'quadrature_nodes_per_population':a.quadrature_nodes,'qmc_used_to_define_case':False,'closure_parameters_refit':False,'stage69_density_preserving_reference':True},'qmc':{'invariants':_invariants(qh),'replicate_diagnostics':[{k:v for k,v in x.items() if k not in ('histories','sources','tails')} for x in qr]},'persistent_coarse':cand(coarse),'persistent_fine':cand(fine)}
 (a.output/f'stage58_{a.case}_summary.json').write_text(json.dumps(jsonable(summary),indent=2,allow_nan=True)+'\n')
 print(json.dumps({'case':a.case,'rho':rho,'first_qmc_density':float(qh[0,1,0]),'qmc_replicates':len(qr)},indent=2))
if __name__=='__main__': main()

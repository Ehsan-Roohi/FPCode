#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from riemann35_patch.stage55_closure_source_audit.run_closure_method import _invariants
from riemann35_patch.stage57_persistent_four_population.run_persistent_method import _run_persistent_candidate
from riemann35_patch.stage71_harder_unseen.hard_cases import CASE_NAMES,hard_case
from riemann35_patch.stage72_density_jacobian_fix.fixed_projection import install_density_consistent_projection

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

def candidate_summary(result):
 h=np.asarray(result['histories']); return {'invariants':_invariants(h[None,...]),'diagnostics':{k:v for k,v in result.items() if k!='histories'}}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--case',choices=CASE_NAMES,required=True); p.add_argument('--stage71-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--coarse-dt',type=float,default=6.25e-4); p.add_argument('--fine-dt',type=float,default=3.125e-4); p.add_argument('--final-time',type=float,default=1.0); p.add_argument('--sample-interval',type=float,default=2.5e-2); p.add_argument('--tau',type=float,default=1.0); p.add_argument('--prandtl',type=float,default=2/3); p.add_argument('--quadrature-nodes',type=int,default=5); a=p.parse_args()
 case=hard_case(a.case); old_npz=np.load(a.stage71_dir/f'stage71_{a.case}.npz'); old_summary=json.loads((a.stage71_dir/f'stage71_{a.case}_summary.json').read_text())
 cs=iratio(a.final_time,a.coarse_dt); css=sample_steps(a.final_time,a.sample_interval,a.coarse_dt); fs=iratio(a.final_time,a.fine_dt); fss=sample_steps(a.final_time,a.sample_interval,a.fine_dt)
 install_density_consistent_projection()
 coarse=_run_persistent_candidate(case.components,dt=a.coarse_dt,steps=cs,sample_steps=css,tau=a.tau,prandtl=a.prandtl,quadrature_nodes=a.quadrature_nodes)
 fine=_run_persistent_candidate(case.components,dt=a.fine_dt,steps=fs,sample_steps=fss,tau=a.tau,prandtl=a.prandtl,quadrature_nodes=a.quadrature_nodes)
 qmc=np.asarray(old_npz['qmc_histories'],float); times=np.asarray(old_npz['times'],float); ch=np.asarray(coarse['histories'])[None,...]; fh=np.asarray(fine['histories'])[None,...]
 a.output.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output/f'stage71_{a.case}.npz',times=times,qmc_histories=qmc,persistent_coarse_histories=ch,persistent_fine_histories=fh)
 summary=dict(old_summary); summary['schema']='riemann35-stage72-density-jacobian-fixed-case-v1'; summary['persistent_coarse']=candidate_summary(coarse); summary['persistent_fine']=candidate_summary(fine); summary['controls']=dict(old_summary['controls']); summary['controls']['stage72_density_jacobian_fix']=True; summary['controls']['closure_parameters_refit']=False
 (a.output/f'stage71_{a.case}_summary.json').write_text(json.dumps(jsonable(summary),indent=2,allow_nan=True)+'\n')
 print(json.dumps({'case':a.case,'rho':float(case.moments[0]),'fixed_projection_fraction':fine['minimum_projection_fraction'],'fixed_projection_residual':fine['maximum_projection_residual'],'fixed_invariants':summary['persistent_fine']['invariants']},indent=2))
if __name__=='__main__': main()

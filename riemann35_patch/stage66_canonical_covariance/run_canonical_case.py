#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES,blind_case
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import initialize_persistent_gaussian_mixture,persistent_gaussian_mixture_fp_step,persistent_gaussian_mixture_moments

def canonical_components(case):
 rho=sum(float(c[0]) for c in case.components); u=np.asarray(case.configuration['bulk_velocity'],float); E=float(case.configuration['energy_trace']); s=np.sqrt(E)
 comps=tuple((float(w)/rho,(np.asarray(m)-u)/s,np.asarray(C)/(s*s)) for w,m,C in case.components)
 return rho,u,s,comps

def raw_components(state,rho,u,s):
 return tuple((rho*float(p),u+s*np.asarray(m),(s*s)*np.asarray(C)) for p,m,C in zip(state.probabilities,state.means,state.covariances))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--case',choices=CASE_NAMES,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--dt',type=float,default=3.125e-4); p.add_argument('--final-time',type=float,default=.25); a=p.parse_args(); case=blind_case(a.case)
 rho,u,s,cc=canonical_components(case); direct=initialize_persistent_gaussian_mixture(case.components); canon=initialize_persistent_gaussian_mixture(cc); n=int(round(a.final_time/a.dt)); every=int(round(.025/a.dt)); traj=[]
 for k in range(n+1):
  if k%every==0 or k==n:
   md=persistent_gaussian_mixture_moments(direct); mapped=initialize_persistent_gaussian_mixture(raw_components(canon,rho,u,s)); mc=persistent_gaussian_mixture_moments(mapped); traj.append([k*a.dt,float(np.linalg.norm(mc-md)/max(np.linalg.norm(md),1e-14))])
  if k<n:
   direct,_,_=persistent_gaussian_mixture_fp_step(direct,a.dt,1.0); canon,_,_=persistent_gaussian_mixture_fp_step(canon,a.dt,1.0)
 out={'schema':'riemann35-stage66-canonical-covariance-v1','case':a.case,'density':rho,'bulk_velocity':u.tolist(),'thermal_scale':s,'initial_relative_covariance_defect':traj[0][1],'max_relative_covariance_defect':max(x[1] for x in traj),'final_relative_covariance_defect':traj[-1][1],'trajectory':traj,'interpretation':'Compares direct physical-variable Stage-57 evolution with center/scale canonical evolution mapped back to physical variables. Nonzero growth measures broken translation/scale covariance of the persistent closure.'}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/f'stage66_{a.case}_summary.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

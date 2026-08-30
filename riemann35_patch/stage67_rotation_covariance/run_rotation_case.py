#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES,blind_case
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import initialize_persistent_gaussian_mixture,persistent_gaussian_mixture_fp_step,persistent_gaussian_mixture_moments

def rx(a):
 a=np.deg2rad(a); c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],float)
def ry(a):
 a=np.deg2rad(a); c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],float)
def rz(a):
 a=np.deg2rad(a); c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
def rotate_components(comps,u,Q):
 return tuple((float(w),u+(np.asarray(m)-u)@Q.T,Q@np.asarray(C)@Q.T) for w,m,C in comps)
def map_back_state(state,u,Q):
 comps=[]
 for p,m,C in zip(state.probabilities,state.means,state.covariances):
  mb=u+(np.asarray(m)-u)@Q
  Cb=Q.T@np.asarray(C)@Q
  comps.append((state.rho*float(p),mb,Cb))
 return initialize_persistent_gaussian_mixture(tuple(comps))
def run_for_rotation(case,Q,dt,final_time):
 u=np.asarray(case.configuration['bulk_velocity'],float)
 direct=initialize_persistent_gaussian_mixture(case.components)
 rot=initialize_persistent_gaussian_mixture(rotate_components(case.components,u,Q))
 n=int(round(final_time/dt)); every=max(1,int(round(.025/dt))); rec=[]
 for k in range(n+1):
  if k%every==0 or k==n:
   md=persistent_gaussian_mixture_moments(direct)
   mr=persistent_gaussian_mixture_moments(map_back_state(rot,u,Q))
   rec.append([k*dt,float(np.linalg.norm(mr-md)/max(np.linalg.norm(md),1e-14))])
  if k<n:
   direct,_,_=persistent_gaussian_mixture_fp_step(direct,dt,1.0)
   rot,_,_=persistent_gaussian_mixture_fp_step(rot,dt,1.0)
 return rec

def main():
 p=argparse.ArgumentParser(); p.add_argument('--case',choices=CASE_NAMES,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--dt',type=float,default=3.125e-4); p.add_argument('--final-time',type=float,default=.25); a=p.parse_args(); case=blind_case(a.case)
 rotations=[('R1',rx(37)@ry(-23)@rz(61)),('R2',rx(-41)@ry(32)@rz(-17))]
 tests=[]
 for name,Q in rotations:
  tr=run_for_rotation(case,Q,a.dt,a.final_time); tests.append({'rotation':name,'matrix':Q.tolist(),'initial_defect':tr[0][1],'max_defect':max(x[1] for x in tr),'final_defect':tr[-1][1],'trajectory':tr})
 out={'schema':'riemann35-stage67-rotation-covariance-v1','case':a.case,'tests':tests,'worst_max_defect':max(t['max_defect'] for t in tests),'interpretation':'Applies two independent rigid rotations to the identical physical state, evolves them with Stage-57, maps them back, and measures violation of rotational covariance.'}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/f'stage67_{a.case}_summary.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

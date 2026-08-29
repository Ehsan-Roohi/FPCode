#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from hyqmom_fp import HYQMOM_35_INDICES, mixture_of_gaussians_moments_35
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES, blind_case
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import initialize_persistent_gaussian_mixture, persistent_gaussian_mixture_fp_step, persistent_gaussian_mixture_moments

PAIRS=((0,0),(1,0),(1,1),(2,0),(2,1),(2,2))

def unpack(x,rho):
 logits=x[:4]; w=np.exp(logits-np.max(logits)); w/=w.sum(); means=x[4:16].reshape(4,3); z=x[16:].reshape(4,6)
 cov=[]
 for row in z:
  L=np.zeros((3,3)); L[0,0]=np.exp(row[0]); L[1,0]=row[1]; L[1,1]=np.exp(row[2]); L[2,0]=row[3]; L[2,1]=row[4]; L[2,2]=np.exp(row[5]); cov.append(L@L.T)
 return tuple((rho*float(wi),mi,ci) for wi,mi,ci in zip(w,means,cov))

def pack(case):
 comps=case.components; rho=sum(c[0] for c in comps); w=np.array([c[0]/rho for c in comps]); means=np.array([c[1] for c in comps]); zz=[]
 for _,_,C in comps:
  L=np.linalg.cholesky(C); zz.append([np.log(L[0,0]),L[1,0],np.log(L[1,1]),L[2,0],L[2,1],np.log(L[2,2])])
 return np.r_[np.log(w),means.ravel(),np.asarray(zz).ravel()],rho

def main():
 p=argparse.ArgumentParser(); p.add_argument('--case',choices=CASE_NAMES,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--dt',type=float,default=3.125e-4); p.add_argument('--final-time',type=float,default=0.25); p.add_argument('--seed',type=int,default=62001); a=p.parse_args()
 case=blind_case(a.case); x0,rho=pack(case); target=case.moments.copy(); scale=np.maximum(np.abs(target),1e-3*np.max(np.abs(target)))
 rng=np.random.default_rng(a.seed+CASE_NAMES.index(a.case)*1009); direction=rng.normal(size=x0.size); direction/=np.linalg.norm(direction); xstart=x0+0.08*direction
 def residual(x): return (mixture_of_gaussians_moments_35(unpack(x,rho))-target)/scale
 sol=least_squares(residual,xstart,max_nfev=2500,xtol=1e-12,ftol=1e-12,gtol=1e-12)
 alt=unpack(sol.x,rho); init_rel=float(np.linalg.norm(mixture_of_gaussians_moments_35(alt)-target)/max(np.linalg.norm(target),1e-14)); latent_distance=float(np.linalg.norm(sol.x-x0)/np.sqrt(x0.size))
 s0=initialize_persistent_gaussian_mixture(case.components); s1=initialize_persistent_gaussian_mixture(alt); steps=int(round(a.final_time/a.dt)); sample=max(1,int(round(0.025/a.dt))); records=[]
 for k in range(steps+1):
  if k%sample==0 or k==steps:
   m0=persistent_gaussian_mixture_moments(s0); m1=persistent_gaussian_mixture_moments(s1); rel=float(np.linalg.norm(m1-m0)/max(np.linalg.norm(m0),1e-14)); records.append([k*a.dt,rel])
  if k<steps:
   s0,_,_=persistent_gaussian_mixture_fp_step(s0,a.dt,1.0); s1,_,_=persistent_gaussian_mixture_fp_step(s1,a.dt,1.0)
 out={'schema':'riemann35-stage62-latent-equivalence-v1','case':a.case,'optimizer_success':bool(sol.success),'optimizer_cost':float(sol.cost),'initial_relative_moment_mismatch':init_rel,'latent_parameter_rms_distance':latent_distance,'final_relative_future_divergence':records[-1][1],'max_relative_future_divergence':float(max(r[1] for r in records)),'trajectory':records,'interpretation':'If initial retained moments match tightly while future trajectories diverge materially, the persistent-Gaussian state is not uniquely determined by the retained 35 moments and exposes latent-decomposition dependence.'}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/f'stage62_{a.case}_summary.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

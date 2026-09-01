#!/usr/bin/env python3
"""Manufactured 1-D spatial transport/collision gate for the promoted closure.

A sinusoidal density wave carrying the Stage71 dense-hot mixture is advected at
a prescribed common carrier speed on a periodic mesh.  All normalized velocity
statistics are spatially uniform, so transport acts only on density and the
homogeneous cubic-FP collision map acts identically at every x.  This gives an
analytic density translation and analytic heat-flux decay while exercising the
core projector over a continuum of densities above and below the Stage71 rho.
This is a manufactured spatial consistency test, not a kinetic-flux validation.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from hyqmom_fp import macroscopic_state
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import PersistentGaussianMixtureState,initialize_persistent_gaussian_mixture,persistent_gaussian_mixture_fp_step,persistent_gaussian_mixture_moments
from riemann35_patch.stage71_harder_unseen.hard_cases import hard_case

def jsonable(v):
 if isinstance(v,np.generic): return v.item()
 if isinstance(v,np.ndarray): return v.tolist()
 if isinstance(v,dict): return {k:jsonable(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [jsonable(x) for x in v]
 return v

def state_at_density(template,rho):
 return PersistentGaussianMixtureState(rho=float(rho),probabilities=template.probabilities.copy(),means=template.means.copy(),covariances=template.covariances.copy())

def one_grid(nx,*,carrier=0.4,cfl=0.4,final_time=0.25,amplitude=0.2,prandtl=2/3):
 case=hard_case('dense_hot_extreme'); template=initialize_persistent_gaussian_mixture(case.components); rho0=float(template.rho); dx=1.0/nx; dt=cfl*dx/abs(carrier); steps=int(round(final_time/dt)); dt=final_time/steps; effective_cfl=abs(carrier)*dt/dx
 if effective_cfl>1: raise ValueError('unstable CFL')
 x=(np.arange(nx)+0.5)*dx; rho=rho0*(1+amplitude*np.sin(2*np.pi*x)); states=[state_at_density(template,r) for r in rho]
 q0=macroscopic_state(persistent_gaussian_mixture_moments(template)).heat_flux/rho0; mass0=float(np.sum(rho)*dx); minrho=float(np.min(rho)); minfrac=1.0; maxres=0.0
 for _ in range(steps):
  if carrier>=0: rho=(1-effective_cfl)*rho+effective_cfl*np.roll(rho,1)
  else: rho=(1-effective_cfl)*rho+effective_cfl*np.roll(rho,-1)
  new=[]
  for i,s in enumerate(states):
   transported=PersistentGaussianMixtureState(rho=float(rho[i]),probabilities=s.probabilities.copy(),means=s.means.copy(),covariances=s.covariances.copy())
   updated,m,d=persistent_gaussian_mixture_fp_step(transported,dt,1.0,prandtl=prandtl,quadrature_nodes=5,enforce_heat_flux_rate=True)
   minfrac=min(minfrac,d.heat_flux_projection_fraction); maxres=max(maxres,d.heat_flux_projection_residual); new.append(updated)
  states=new; minrho=min(minrho,float(np.min(rho)))
 exact_rho=rho0*(1+amplitude*np.sin(2*np.pi*(x-carrier*final_time))); density_error=float(np.linalg.norm(rho-exact_rho)/np.linalg.norm(exact_rho)); mass_drift=abs(float(np.sum(rho)*dx)-mass0)
 q_exact=q0*np.exp(-2*prandtl*final_time); q_errors=[]
 for s in states:
  q=macroscopic_state(persistent_gaussian_mixture_moments(s)).heat_flux/s.rho
  q_errors.append(np.linalg.norm(q-q_exact)/max(np.linalg.norm(q_exact),1e-14))
 means=np.asarray([s.means for s in states]); covs=np.asarray([s.covariances for s in states]); shape_spread=max(float(np.max(np.abs(means-means[:1]))),float(np.max(np.abs(covs-covs[:1]))))
 return {'nx':nx,'dx':dx,'dt':dt,'steps':steps,'cfl':effective_cfl,'rho_min':minrho,'rho_max':float(np.max(rho)),'density_relative_l2':density_error,'global_mass_drift':mass_drift,'minimum_projection_fraction':minfrac,'maximum_projection_residual':maxres,'maximum_q_over_rho_exact_error':float(max(q_errors)),'maximum_shape_spread':shape_spread}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 coarse=one_grid(32); fine=one_grid(64); ratio=fine['density_relative_l2']/max(coarse['density_relative_l2'],1e-30)
 gates={'positive_density':min(coarse['rho_min'],fine['rho_min'])>0,'mass_conservation':max(coarse['global_mass_drift'],fine['global_mass_drift'])<1e-12,'full_projection':min(coarse['minimum_projection_fraction'],fine['minimum_projection_fraction'])>=1-2e-13,'projection_residual':max(coarse['maximum_projection_residual'],fine['maximum_projection_residual'])<1e-10,'normalized_heat_flux_exact':max(coarse['maximum_q_over_rho_exact_error'],fine['maximum_q_over_rho_exact_error'])<1e-10,'density_refines':ratio<0.75,'fine_density_accuracy':fine['density_relative_l2']<0.02,'density_covariant_shape':max(coarse['maximum_shape_spread'],fine['maximum_shape_spread'])<1e-10}; passed=all(gates.values())
 summary={'schema':'riemann35-stage75-spatial-density-wave-v1','scope':'manufactured periodic common-carrier spatial transport plus promoted persistent collision closure','case':'dense_hot_extreme','case_fingerprint':hard_case('dense_hot_extreme').fingerprint,'coarse':coarse,'fine':fine,'fine_over_coarse_density_error':ratio,'gates':gates,'qualification_pass':passed}
 (a.output/'stage75_spatial_summary.json').write_text(json.dumps(jsonable(summary),indent=2)+'\n')
 lines=['# Stage 75 — promoted-core manufactured spatial gate','',f"Qualification objective: **{'PASS' if passed else 'FAIL'}**",'', '| grid | density L2 | mass drift | min projection | max projection residual | q/rho exact error | shape spread |','|---:|---:|---:|---:|---:|---:|---:|']
 for r in (coarse,fine): lines.append(f"| {r['nx']} | {100*r['density_relative_l2']:.4f}% | {r['global_mass_drift']:.3e} | {r['minimum_projection_fraction']:.6f} | {r['maximum_projection_residual']:.3e} | {r['maximum_q_over_rho_exact_error']:.3e} | {r['maximum_shape_spread']:.3e} |")
 lines+=['',f"Fine/coarse density-error ratio: **{ratio:.4f}**",'', 'This is a manufactured spatial consistency gate with prescribed common-carrier advection; it is not an independent kinetic-flux or DSMC validation.']
 (a.output/'STAGE75_RESULTS.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines)); print('STAGE75_SPATIAL_PASS='+str(passed))
 if not passed: raise SystemExit(2)
if __name__=='__main__': main()

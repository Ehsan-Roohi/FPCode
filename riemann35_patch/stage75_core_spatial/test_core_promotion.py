#!/usr/bin/env python3
"""Regression gate for the density-consistent Stage57 projector promoted to core."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from hyqmom_fp import HYQMOM_35_INDICES, macroscopic_state
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import initialize_persistent_gaussian_mixture,persistent_gaussian_mixture_fp_step,persistent_gaussian_mixture_moments
from riemann35_patch.stage71_harder_unseen.hard_cases import hard_case
POS={a:i for i,a in enumerate(HYQMOM_35_INDICES)}

def invariant_vector(m):
 return np.asarray([m[POS[(0,0,0)]],m[POS[(1,0,0)]],m[POS[(0,1,0)]],m[POS[(0,0,1)]],m[POS[(2,0,0)]]+m[POS[(0,2,0)]]+m[POS[(0,0,2)]]])

def run(name,steps=256,dt=3.125e-4):
 case=hard_case(name); state=initialize_persistent_gaussian_mixture(case.components); m0=persistent_gaussian_mixture_moments(state); inv0=invariant_vector(m0); q0=macroscopic_state(m0).heat_flux.copy(); minfrac=1.0; maxres=0.0; maxinv=0.0
 for _ in range(steps):
  state,m,d=persistent_gaussian_mixture_fp_step(state,dt,1.0,prandtl=2/3,quadrature_nodes=5,enforce_heat_flux_rate=True)
  minfrac=min(minfrac,d.heat_flux_projection_fraction); maxres=max(maxres,d.heat_flux_projection_residual); maxinv=max(maxinv,float(np.max(np.abs(invariant_vector(m)-inv0))))
 exact=q0*np.exp(-2*(2/3)*steps*dt); q=macroscopic_state(m).heat_flux; qerr=float(np.linalg.norm(q-exact)/max(np.linalg.norm(exact),1e-14))
 assert minfrac >= 1.0-2e-13,(name,minfrac)
 assert maxres < 1e-10,(name,maxres)
 assert maxinv < 2e-10,(name,maxinv)
 assert qerr < 1e-10,(name,qerr)
 return name,float(case.moments[0]),minfrac,maxres,maxinv,qerr

def main():
 rows=[run('dense_hot_extreme'),run('dilute_broad'),run('rare_beam_3d')]
 for r in rows: print('STAGE75_CORE_CASE name=%s rho=%.6g projection_fraction=%.16g projection_residual=%.3e invariant_drift=%.3e q_exact_error=%.3e'%r)
 print('STAGE75_CORE_PROMOTION=PASS')
if __name__=='__main__': main()

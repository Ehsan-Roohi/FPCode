#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES,blind_case
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import initialize_persistent_gaussian_mixture,persistent_gaussian_mixture_fp_step
from riemann35_patch.stage60_fourth_order_memory.fourth_order_memory import fourth_order_memory_defect

def main():
 p=argparse.ArgumentParser(); p.add_argument('--case',choices=CASE_NAMES,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--dt',type=float,default=3.125e-4); p.add_argument('--final-time',type=float,default=1.0); p.add_argument('--tau',type=float,default=1.0); p.add_argument('--prandtl',type=float,default=2/3); p.add_argument('--quadrature-nodes',type=int,default=5); a=p.parse_args()
 case=blind_case(a.case); state=initialize_persistent_gaussian_mixture(case.components); n=int(round(a.final_time/a.dt)); rows=[]
 for step in range(n):
  d=fourth_order_memory_defect(state,a.dt,a.tau,prandtl=a.prandtl,quadrature_nodes=a.quadrature_nodes)
  rows.append((step*a.dt/a.tau,d.total_excess,d.radial_excess,d.anisotropic_excess,d.max_population,d.affine_scale))
  state,_,_=persistent_gaussian_mixture_fp_step(state,a.dt,a.tau,prandtl=a.prandtl,quadrature_nodes=a.quadrature_nodes)
 arr=np.asarray(rows,float); a.output.mkdir(parents=True,exist_ok=True)
 with (a.output/f'stage60_{a.case}.csv').open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['time_over_tau','fourth_excess','radial_excess','anisotropic_excess','max_population_fourth','affine_scale']); w.writerows(rows)
 m={'schema':'riemann35-stage60-fourth-order-memory-v1','case':a.case,'role':case.role,'fingerprint':case.fingerprint,'dt_over_tau':a.dt/a.tau,'final_time_over_tau':a.final_time/a.tau,'rms_fourth_excess':float(np.sqrt(np.mean(arr[:,1]**2))),'rms_radial_excess':float(np.sqrt(np.mean(arr[:,2]**2))),'rms_anisotropic_excess':float(np.sqrt(np.mean(arr[:,3]**2))),'max_fourth_excess':float(np.max(arr[:,1])),'max_population_fourth':float(np.max(arr[:,4])),'interpretation':'Measures within-population fourth cumulant erased by Gaussian recompression; no QMC is used.'}
 (a.output/f'stage60_{a.case}_summary.json').write_text(json.dumps(m,indent=2)+'\n'); print(json.dumps(m,indent=2))
if __name__=='__main__': main()

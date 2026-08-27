#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES, blind_case
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import initialize_persistent_gaussian_mixture, persistent_gaussian_mixture_fp_step
from riemann35_patch.stage59_compression_defect.compression_defect import compression_defect

def main():
 p=argparse.ArgumentParser(); p.add_argument('--case',choices=CASE_NAMES,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--dt',type=float,default=3.125e-4); p.add_argument('--final-time',type=float,default=1.0); p.add_argument('--tau',type=float,default=1.0); p.add_argument('--prandtl',type=float,default=2/3); p.add_argument('--quadrature-nodes',type=int,default=5); a=p.parse_args()
 case=blind_case(a.case); state=initialize_persistent_gaussian_mixture(case.components); n=int(round(a.final_time/a.dt));
 if n<1 or not np.isclose(n*a.dt,a.final_time,atol=2e-13,rtol=0): raise ValueError('final-time must be integer multiple of dt')
 rows=[]
 for step in range(n):
  d=compression_defect(state,a.dt,a.tau,prandtl=a.prandtl,quadrature_nodes=a.quadrature_nodes)
  rows.append((step*a.dt/a.tau,d.total,d.trace_free,d.heat_flux,d.max_population,d.affine_scale))
  state,_,_=persistent_gaussian_mixture_fp_step(state,a.dt,a.tau,prandtl=a.prandtl,quadrature_nodes=a.quadrature_nodes)
 arr=np.asarray(rows,float); a.output.mkdir(parents=True,exist_ok=True)
 with (a.output/f'stage59_{a.case}.csv').open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['time_over_tau','discarded_third','discarded_trace_free','discarded_heat_flux','max_population_third','affine_scale']); w.writerows(rows)
 metrics={'schema':'riemann35-stage59-compression-defect-v1','case':a.case,'role':case.role,'fingerprint':case.fingerprint,'dt_over_tau':a.dt/a.tau,'final_time_over_tau':a.final_time/a.tau,'rms_discarded_third':float(np.sqrt(np.mean(arr[:,1]**2))),'rms_discarded_trace_free':float(np.sqrt(np.mean(arr[:,2]**2))),'rms_discarded_heat_flux':float(np.sqrt(np.mean(arr[:,3]**2))),'max_discarded_third':float(np.max(arr[:,1])),'max_population_third':float(np.max(arr[:,4])),'interpretation':'Measures within-population third central moment erased by the Stage-57 Gaussian recompression; no QMC is used.'}
 (a.output/f'stage59_{a.case}_summary.json').write_text(json.dumps(metrics,indent=2)+'\n')
 print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()

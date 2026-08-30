#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES,blind_case

def rel(a,b):
 return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-14))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--case',choices=CASE_NAMES,required=True); p.add_argument('--stage58-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 case=blind_case(a.case); f=a.stage58_dir/f'stage58_{a.case}.npz'; z=np.load(f)
 times=np.asarray(z['times'],float); q=np.asarray(z['qmc_histories'],float); fine=np.asarray(z['persistent_fine_histories'],float)[0]
 true_rho=float(case.moments[0]); masses=q[:,:,0]
 repaired=q.copy(); factors=true_rho/np.maximum(masses,1e-300); repaired*=factors[:,:,None]
 qmean=q.mean(axis=0); rmean=repaired.mean(axis=0)
 old=np.array([rel(qmean[i],fine[i]) for i in range(len(times))]); new=np.array([rel(rmean[i],fine[i]) for i in range(len(times))])
 out={'schema':'riemann35-stage68-qmc-density-continuity-v1','case':a.case,'true_density':true_rho,'qmc_mass_t0_mean':float(masses[:,0].mean()),'qmc_mass_first_positive_time_mean':float(masses[:,1].mean()) if masses.shape[1]>1 else None,'qmc_mass_final_mean':float(masses[:,-1].mean()),'max_qmc_mass_jump_from_true':float(np.max(np.abs(masses-true_rho))),'old_rms_relative_error':float(np.sqrt(np.mean(old**2))),'repaired_rms_relative_error':float(np.sqrt(np.mean(new**2))),'old_final_relative_error':float(old[-1]),'repaired_final_relative_error':float(new[-1]),'error_reduction_factor':float(np.sqrt(np.mean(old**2))/max(np.sqrt(np.mean(new**2)),1e-300)),'trajectory':[[float(t),float(o),float(n),float(masses[:,i].mean())] for i,(t,o,n) in enumerate(zip(times,old,new))],'interpretation':'Stage 58 stores the exact density at t=0, but post-step moments were reconstructed through moments_35_from_qmc with rho=1. Rescaling every QMC snapshot by conserved true_rho/current_mass repairs this pure mass-normalization defect without altering velocity dynamics.'}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/f'stage68_{a.case}_summary.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES, blind_case

def rel(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-14))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--case',choices=CASE_NAMES,required=True); p.add_argument('--stage58-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    case=blind_case(a.case); f=a.stage58_dir/f'stage58_{a.case}.npz'
    z=np.load(f); times=np.asarray(z['times'],float); q=np.asarray(z['qmc_histories'],float); fine=np.asarray(z['persistent_fine_histories'],float)
    if q.ndim!=3 or fine.ndim!=3 or fine.shape[0]!=1: raise ValueError('unexpected Stage58 history shapes')
    true_rho=float(case.moments[0]); frozen_mass=float(np.median(q[:,0,0])); factor=true_rho/frozen_mass
    corrected=q*factor; q0=np.mean(q[:,0,:],axis=0); cq0=np.mean(corrected[:,0,:],axis=0); qmean=np.mean(q,axis=0); cqmean=np.mean(corrected,axis=0); cand=fine[0]
    old=np.array([rel(cand[i],qmean[i]) for i in range(len(times))]); new=np.array([rel(cand[i],cqmean[i]) for i in range(len(times))])
    qnoise=np.linalg.norm(np.std(corrected,axis=0,ddof=1),axis=1)/np.maximum(np.linalg.norm(cqmean,axis=1),1e-14)
    out={
      'schema':'riemann35-stage65-density-aware-reference-v1','case':a.case,'role':case.role,'true_density':true_rho,'frozen_qmc_density':frozen_mass,'density_correction_factor':factor,
      'initial_relative_mismatch_before':rel(q0,case.moments),'initial_relative_mismatch_after':rel(cq0,case.moments),
      'old_rms_relative_error':float(np.sqrt(np.mean(old**2))),'corrected_rms_relative_error':float(np.sqrt(np.mean(new**2))),
      'old_final_relative_error':float(old[-1]),'corrected_final_relative_error':float(new[-1]),
      'error_reduction_factor':float(np.sqrt(np.mean(old**2))/max(np.sqrt(np.mean(new**2)),1e-16)),
      'max_corrected_qmc_relative_std':float(np.max(qnoise)),'trajectory':[[float(t),float(o),float(n)] for t,o,n in zip(times,old,new)],
      'interpretation':'Stage-58 QMC weights were normalized to unit mass and moments_35_from_qmc used rho=1. For homogeneous collision dynamics the velocity evolution is density-scale invariant, so multiplying every frozen QMC raw moment by the true conserved density restores the physical reference without rerunning QMC.'}
    a.output.mkdir(parents=True,exist_ok=True); (a.output/f'stage65_{a.case}_summary.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES, blind_case
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import initialize_persistent_gaussian_mixture, persistent_gaussian_mixture_fp_step, persistent_gaussian_mixture_moments
from riemann35_patch.stage64_generator_consistent_map.generator_consistent_map import generator_consistent_step


def rel(a,b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-14))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--case',choices=CASE_NAMES,required=True); p.add_argument('--stage58-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--dt',type=float,default=3.125e-4); p.add_argument('--final-time',type=float,default=0.25); a=p.parse_args()
 case=blind_case(a.case); data=np.load(a.stage58_dir/f'stage58_{a.case}.npz'); times=np.asarray(data['times']); qmc=np.asarray(data['qmc_histories']).mean(axis=0)
 keep=np.where(times<=a.final_time+1e-13)[0]; sample_times=times[keep]; qmc=qmc[keep]
 sbase=initialize_persistent_gaussian_mixture(case.components); scorr=initialize_persistent_gaussian_mixture(case.components)
 every=int(round(0.025/a.dt)); steps=int(round(a.final_time/a.dt)); records=[]; diag=[]
 qi=0
 for k in range(steps+1):
  if k%every==0 or k==steps:
   mb=persistent_gaussian_mixture_moments(sbase); mc=persistent_gaussian_mixture_moments(scorr); ref=qmc[qi]
   records.append([k*a.dt,rel(mb,ref),rel(mc,ref),rel(mc,mb)]); qi+=1
  if k<steps:
   sbase,_,_=persistent_gaussian_mixture_fp_step(sbase,a.dt,1.0)
   scorr,_,d=generator_consistent_step(scorr,a.dt,1.0); diag.append(d)
 arr=np.asarray(records); base_rms=float(np.sqrt(np.mean(arr[:,1]**2))); corr_rms=float(np.sqrt(np.mean(arr[:,2]**2)))
 out={'schema':'riemann35-stage64-generator-consistent-v1','case':a.case,'dt_over_tau':a.dt,'final_time_over_tau':a.final_time,'baseline_rms_relative_error':base_rms,'corrected_rms_relative_error':corr_rms,'improvement_factor':float(base_rms/max(corr_rms,1e-14)),'baseline_final_relative_error':float(arr[-1,1]),'corrected_final_relative_error':float(arr[-1,2]),'max_corrected_vs_baseline_divergence':float(np.max(arr[:,3])),'max_fourth_source_error_after_projection':float(max(d['fourth_source_error'] for d in diag)),'max_lower_order_change_from_projection':float(max(d['lower_order_change'] for d in diag)),'min_projection_fraction':float(min(d['projection_fraction'] for d in diag)),'trajectory':records,'interpretation':'Tests whether enforcing the continuous-generator order-4 increment inside the persistent Gaussian map reduces frozen Stage-58 QMC trajectory error without altering lower-order Stage-57 output.'}
 a.output.mkdir(parents=True,exist_ok=True); (a.output/f'stage64_{a.case}_summary.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

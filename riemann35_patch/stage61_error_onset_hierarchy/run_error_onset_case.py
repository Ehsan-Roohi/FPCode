#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from hyqmom_fp import HYQMOM_35_INDICES
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES

def first_sustained(mask: np.ndarray, count: int=3):
    for i in range(0, max(0, len(mask)-count+1)):
        if bool(np.all(mask[i:i+count])):
            return i
    return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=CASE_NAMES,required=True)
    p.add_argument('--stage58-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--floor',type=float,default=0.02)
    p.add_argument('--sigma',type=float,default=5.0)
    p.add_argument('--sustain',type=int,default=3)
    a=p.parse_args()
    src=a.stage58_dir/f'stage58_{a.case}.npz'
    if not src.exists(): raise FileNotFoundError(src)
    data=np.load(src)
    t=np.asarray(data['times'],float)
    q=np.asarray(data['qmc_histories'],float)
    c=np.asarray(data['persistent_fine_histories'],float)[0]
    if q.ndim!=3 or c.ndim!=2 or q.shape[1:]!=c.shape: raise ValueError('unexpected Stage58 array shapes')
    mean=q.mean(axis=0)
    sem=q.std(axis=0,ddof=1)/np.sqrt(q.shape[0]) if q.shape[0]>1 else np.zeros_like(mean)
    orders={k:np.asarray([i for i,m in enumerate(HYQMOM_35_INDICES) if sum(m)==k],int) for k in (2,3,4)}
    out={'schema':'riemann35-stage61-error-onset-hierarchy-v1','case':a.case,'source_npz':str(src),'qmc_replicates':int(q.shape[0]),'floor':a.floor,'sigma':a.sigma,'sustain_samples':a.sustain,'orders':{}}
    for order,idx in orders.items():
        ref_norm=np.linalg.norm(mean[:,idx],axis=1)
        scale=max(float(np.max(ref_norm)),1e-14)
        err=np.linalg.norm(c[:,idx]-mean[:,idx],axis=1)/scale
        noise=np.linalg.norm(sem[:,idx],axis=1)/scale
        threshold=np.maximum(a.floor,a.sigma*noise)
        exceed=err>threshold
        pos=first_sustained(exceed,a.sustain)
        out['orders'][str(order)]={'component_count':int(idx.size),'scale':scale,'rms_relative_error':float(np.sqrt(np.mean(err**2))),'max_relative_error':float(np.max(err)),'median_qmc_noise':float(np.median(noise)),'onset_time_over_tau':None if pos is None else float(t[pos]),'onset_error':None if pos is None else float(err[pos]),'onset_threshold':None if pos is None else float(threshold[pos])}
    onsets={k:v['onset_time_over_tau'] for k,v in out['orders'].items() if v['onset_time_over_tau'] is not None}
    out['earliest_diverging_order']=None if not onsets else min(onsets,key=lambda k:onsets[k])
    a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/f'stage61_{a.case}_summary.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()

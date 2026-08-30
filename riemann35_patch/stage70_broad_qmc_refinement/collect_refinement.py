#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from riemann35_patch.stage58_blind_generalization.collect_generalization_gate import _derived
from riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux import relative_history_error,replicate_spread

def main():
 p=argparse.ArgumentParser(); p.add_argument('--old-dir',type=Path,required=True); p.add_argument('--new-dir',type=Path,required=True); a=p.parse_args()
 old=np.load(a.old_dir/'stage58_broad_shifted.npz'); new=np.load(a.new_dir/'stage58_broad_shifted.npz')
 oq=_derived(np.asarray(old['qmc_histories'],float)); nq=_derived(np.asarray(new['qmc_histories'],float))
 old_spread=float(replicate_spread(oq['full_tensor'])); new_spread=float(replicate_spread(nq['full_tensor']))
 old_mean=np.mean(oq['full_tensor'],axis=0); new_mean=np.mean(nq['full_tensor'],axis=0)
 mean_shift=float(relative_history_error(new_mean,old_mean))
 gate=json.loads((a.new_dir/'stage58_generalization_summary.json').read_text())
 broad=gate['case_results']['broad_shifted']
 summary={'schema':'riemann35-stage70-broad-qmc-refinement-v1','old_points_per_component':65536,'new_points_per_component':131072,'replicates':8,'old_qmc_spread':old_spread,'new_qmc_spread':new_spread,'qmc_mean_refinement_shift':mean_shift,'frozen_reference_spread_gate':0.02,'broad_pass':bool(broad['pass']),'qualification_pass':bool(gate['qualification_pass']),'broad_errors':broad['errors']}
 (a.new_dir/'stage70_refinement_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Stage 70 — broad-case QMC reference refinement','',f"Old QMC spread (65536/component): **{100*old_spread:.4f}%**",f"Refined QMC spread (131072/component): **{100*new_spread:.4f}%**",f"Reference mean refinement shift: **{100*mean_shift:.4f}%**",f"Frozen spread gate: **2.0000%**",'',f"Broad case: **{'PASS' if broad['pass'] else 'FAIL'}**",f"Full five-case qualification: **{'PASS' if gate['qualification_pass'] else 'FAIL'}**",'',"No closure parameter, case definition, or qualification threshold was changed in Stage 70."]
 (a.new_dir/'STAGE70_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

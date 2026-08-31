#!/usr/bin/env python3
import numpy as np
from riemann35_patch.stage71_harder_unseen.hard_cases import CASE_NAMES,hard_case,registry_manifest

def main():
 reg=registry_manifest(); assert reg['qmc_used_to_define_cases'] is False; assert reg['closure_parameters_refit'] is False
 for name in CASE_NAMES:
  c=hard_case(name); assert c.fingerprint==reg['case_fingerprints'][name]; assert c.moments.shape==(35,); assert np.all(np.isfinite(c.moments)); assert c.moments[0]>0
  assert max(c.audit['mass_error'],c.audit['bulk_velocity_error'],c.audit['energy_trace_error'])<1e-10
  assert c.audit['minimum_covariance_eigenvalue']>0
  print(name,c.fingerprint,c.audit['minimum_covariance_eigenvalue'])
 print('STAGE71_REGISTRY_FINGERPRINT='+reg['registry_fingerprint'])
 print('STAGE71_PREFLIGHT=PASS')
if __name__=='__main__': main()

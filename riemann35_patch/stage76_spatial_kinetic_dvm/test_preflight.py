#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from riemann35_patch.stage71_harder_unseen.hard_cases import hard_case
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import initialize_persistent_gaussian_mixture
from riemann35_patch.stage76_spatial_kinetic_dvm.run_stage76 import state_low,state_from_low,transport_candidate

EXPECTED={
 "dense_hot_extreme":"11a27e07eac086371a0df7c6fd85f5ba717e6ff73841005a44cb78fdc6798a31",
 "dilute_broad":"011d6ae4b1e1ca76eca468881c58f26a66f3988f02f974d5a6e4784dff0684d7",
}
for name,fp in EXPECTED.items():
 c=hard_case(name); assert c.fingerprint==fp
 s=initialize_persistent_gaussian_mixture(c.components)
 rebuilt=state_from_low(state_low(s))
 assert abs(rebuilt.rho-s.rho)<2e-13
 assert np.max(np.abs(rebuilt.means-s.means))<2e-12
 assert np.max(np.abs(rebuilt.covariances-s.covariances))<3e-12

left=initialize_persistent_gaussian_mixture(hard_case('dense_hot_extreme').components)
right=initialize_persistent_gaussian_mixture(hard_case('dilute_broad').components)
states=[left,left,right,right]
updated,balance=transport_candidate(states,1e-5,0.25,left,right,7)
assert np.isfinite(balance) and balance<1e-10
for s in updated:
 assert s.rho>0
 assert min(np.min(np.linalg.eigvalsh(c)) for c in s.covariances)>0
print(f"STAGE76_PREFLIGHT=PASS balance={balance:.3e} frozen_cases=2")

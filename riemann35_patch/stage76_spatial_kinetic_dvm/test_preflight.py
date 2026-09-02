#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from hyqmom_fp import SpatialGrid1D
from riemann35_patch.stage71_harder_unseen.hard_cases import hard_case
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import initialize_persistent_gaussian_mixture
from riemann35_patch.stage76_spatial_kinetic_dvm.run_stage76 import (
    common_grid, initialize_spatial_dvm, state_low, state_from_low, transport_candidate,
)

EXPECTED={
 "dense_hot_extreme":"11a27e07eac086371a0df7c6fd85f5ba717e6ff73841005a44cb78fdc6798a31",
 "dilute_broad":"011d6ae4b1e1ca76eca468881c58f26a66f3988f02f974d5a6e4784dff0684d7",
}
cases={}
for name,fp in EXPECTED.items():
 c=hard_case(name); assert c.fingerprint==fp; cases[name]=c
 s=initialize_persistent_gaussian_mixture(c.components)
 rebuilt=state_from_low(state_low(s))
 assert abs(rebuilt.rho-s.rho)<2e-13
 assert np.max(np.abs(rebuilt.means-s.means))<2e-12
 assert np.max(np.abs(rebuilt.covariances-s.covariances))<3e-12

left_case=cases['dense_hot_extreme']; right_case=cases['dilute_broad']
# Stage76 uses one common support for both boundary distributions.  Qualify the
# exact-moment DVM initializer on the production coarse/fine supports before
# any Slurm allocation is requested.  This is a reference-resolution check,
# not a closure/outcome gate.
xgrid=SpatialGrid1D(-1.0,1.0,4)
projection_residuals=[]
shapes=[]
for cells in (41,49):
 grid=common_grid(left_case.components,right_case.components,cells,7.0)
 _,_,_,lp,rp=initialize_spatial_dvm(xgrid,grid,left_case.components,right_case.components)
 assert lp is not None and rp is not None
 assert lp.relative_moment_residual < 1e-8
 assert rp.relative_moment_residual < 1e-8
 projection_residuals.extend([lp.relative_moment_residual,rp.relative_moment_residual])
 shapes.append(grid.shape)

left=initialize_persistent_gaussian_mixture(left_case.components)
right=initialize_persistent_gaussian_mixture(right_case.components)
states=[left,left,right,right]
updated,balance=transport_candidate(states,1e-5,0.25,left,right,7)
assert np.isfinite(balance) and balance<1e-10
for s in updated:
 assert s.rho>0
 assert min(np.min(np.linalg.eigvalsh(c)) for c in s.covariances)>0
print(
 f"STAGE76_PREFLIGHT=PASS balance={balance:.3e} frozen_cases=2 "
 f"dvm_shapes={shapes} max_init_projection={max(projection_residuals):.3e}"
)

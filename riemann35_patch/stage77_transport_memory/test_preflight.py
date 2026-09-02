#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from hyqmom_fp import SpatialGrid1D, dvm_upwind_transport_step
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import initialize_persistent_gaussian_mixture
from riemann35_patch.stage71_harder_unseen.hard_cases import hard_case
from riemann35_patch.stage76_spatial_kinetic_dvm.run_stage76 import common_grid, initialize_spatial_dvm, transport_candidate

EXPECTED={
 "dense_hot_extreme":"11a27e07eac086371a0df7c6fd85f5ba717e6ff73841005a44cb78fdc6798a31",
 "dilute_broad":"011d6ae4b1e1ca76eca468881c58f26a66f3988f02f974d5a6e4784dff0684d7",
}
left=hard_case("dense_hot_extreme"); right=hard_case("dilute_broad")
assert left.fingerprint==EXPECTED[left.name]
assert right.fingerprint==EXPECTED[right.name]

xgrid=SpatialGrid1D(-1.0,1.0,10)
vgrid=common_grid(left.components,right.components,49,7.0)
dvm,dl,dr,lp,rp=initialize_spatial_dvm(xgrid,vgrid,left.components,right.components)
assert lp is not None and rp is not None
assert lp.relative_moment_residual < 1e-8
assert rp.relative_moment_residual < 1e-8
vmax=float(np.max(np.abs(vgrid.centers()[:,0])))
dt=0.10*xgrid.width/vmax
dvm2,diag=dvm_upwind_transport_step(dvm,dt,dl,dr)
assert np.min(dvm2.masses)>0.0
assert max(diag.mass_balance_residual,diag.momentum_balance_residual,diag.energy_balance_residual)<1e-10

ls=initialize_persistent_gaussian_mixture(left.components)
rs=initialize_persistent_gaussian_mixture(right.components)
states=[ls if x<0 else rs for x in xgrid.centers]
new,balance=transport_candidate(states,dt,xgrid.width,ls,rs,11)
assert balance<1e-10
assert all(s.rho>0 for s in new)
assert min(np.min(np.linalg.eigvalsh(c)) for s in new for c in s.covariances)>0.0
print(f"STAGE77_PREFLIGHT=PASS dvm_shape={vgrid.shape} dt={dt:.9g} dvm_balance={max(diag.mass_balance_residual,diag.momentum_balance_residual,diag.energy_balance_residual):.3e} candidate_balance={balance:.3e}")

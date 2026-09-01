#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from riemann35_patch.stage71_harder_unseen.hard_cases import hard_case
CASES=("rare_beam_3d","dense_hot_extreme","dilute_broad")
for name in CASES:
    case=hard_case(name)
    covs=np.asarray([x[2] for x in case.components])
    assert np.allclose(covs,np.asarray([np.diag(np.diag(c)) for c in covs]),atol=1e-12)
    assert case.audit["minimum_covariance_eigenvalue"]>0
    print(name,case.fingerprint,case.configuration["density"])
print("STAGE74_PREFLIGHT=PASS")

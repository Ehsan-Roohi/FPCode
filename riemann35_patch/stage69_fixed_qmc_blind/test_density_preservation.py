import numpy as np
from riemann35_patch.stage69_fixed_qmc_blind.run_fixed_case import fix_density

def test_fix_density_restores_all_raw_moments_linearly():
    h=np.ones((3,35)); h[0]*=1.6; h[1]*=1.0; h[2]*=1.0
    r=fix_density({'histories':h},1.6)
    out=np.asarray(r['histories'])
    assert np.allclose(out[:,0],1.6)
    assert np.allclose(out[1],1.6*h[1])

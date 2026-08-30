import unittest
import numpy as np
from shock_physics import *
class ShockPhysicsTests(unittest.TestCase):
 def test_mach3_rankine_hugoniot(self):
  up,dn=normal_shock_states(3.); self.assertAlmostEqual(dn.rho,3.,places=13)
  np.testing.assert_allclose(analytic_fluxes(up),analytic_fluxes(dn),rtol=1e-14)
 def test_positive_conservative_nonequilibrium_mixture(self):
  up,dn=normal_shock_states(3.); vx,vr,w=axisymmetric_quadrature(-12,16,320,16,160)
  f=mott_smith_distribution(np.linspace(-7,7,31),vx,vr,up,dn,1.); self.assertGreaterEqual(float(f.min()),0.)
  m=moments_and_fluxes(f,vx,vr,w)
  for k in ("mass_flux","momentum_flux","energy_flux"): self.assertLess(np.ptp(m[k])/abs(np.mean(m[k])),2e-10)
  self.assertGreater(np.max(abs(m["pxx"]-m["pperp"])),.05); self.assertGreater(np.max(abs(m["qx"])),.05)
if __name__=="__main__": unittest.main()

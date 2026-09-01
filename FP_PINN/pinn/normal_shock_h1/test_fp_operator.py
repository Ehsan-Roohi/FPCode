import unittest
import numpy as np
from shock_physics import *
from fp_operator import divergence_axisymmetric,transformed_coordinates,simpson_weights

class FPOperatorTests(unittest.TestCase):
 def test_maxwellian_is_discrete_equilibrium(self):
  s=GasState(1.3,.7,1.8); vx=np.linspace(-10,10,401); vr=np.linspace(0,10,201)
  dvx=vx[1]-vx[0]; dvr=vr[1]-vr[0]; w=np.ones((len(vx),len(vr)))*2*np.pi*vr[None,:]*dvx*dvr; w[:,(0,-1)]*=.5; w[(0,-1),:]*=.5
  f=maxwellian(vx[None,:,None],vr[None,None,:],s); q,_=divergence_axisymmetric(f,vx,vr,w)
  scale=np.max(f); self.assertLess(float(np.sqrt(np.mean(q*q))/scale),3e-3)
 def test_transformed_boundary_states(self):
  up,dn=normal_shock_states(5.); x=np.array([-8.,8.]); vx=np.array([up.u,dn.u]); vr=np.array([0.,1.])
  y,cx,cr=transformed_coordinates(x,vx,vr,up,dn,8.); self.assertEqual(y.shape,(2,1,1)); self.assertAlmostEqual(cx[0,0,0],0.); self.assertAlmostEqual(cx[1,1,0],0.); self.assertTrue(np.all(cr>=0))
 def test_simpson_recovers_mach5_boundary_moments(self):
  up,dn=normal_shock_states(5.); th=np.sqrt(dn.temperature); vx=np.linspace(dn.u-8*th,up.u+8*th,129); vr=np.linspace(0,8*th,81)
  w=simpson_weights(len(vx),vx[1]-vx[0])[:,None]*simpson_weights(len(vr),vr[1]-vr[0])[None,:]*(2*np.pi*vr[None,:])
  f=np.stack([maxwellian(vx[:,None],vr[None,:],s) for s in (up,dn)]); m=moments_and_fluxes(f,vx,vr,w)
  self.assertLess(max(abs(m["rho"]/[up.rho,dn.rho]-1)),2e-4)
  for key,target in zip(("mass_flux","momentum_flux","energy_flux"),analytic_fluxes(up)):
   self.assertLess(np.ptp(m[key])/abs(target),2e-4)
   self.assertLess(np.max(abs(m[key]/target-1)),5e-4)
if __name__=="__main__": unittest.main()

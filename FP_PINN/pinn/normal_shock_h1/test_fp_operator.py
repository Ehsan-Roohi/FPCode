import unittest
import numpy as np
from shock_physics import *
from fp_operator import divergence_axisymmetric,transformed_coordinates

class FPOperatorTests(unittest.TestCase):
 def test_maxwellian_is_discrete_equilibrium(self):
  s=GasState(1.3,.7,1.8); vx=np.linspace(-10,10,401); vr=np.linspace(0,10,201)
  dvx=vx[1]-vx[0]; dvr=vr[1]-vr[0]; w=np.ones((len(vx),len(vr)))*2*np.pi*vr[None,:]*dvx*dvr; w[:,(0,-1)]*=.5; w[(0,-1),:]*=.5
  f=maxwellian(vx[None,:,None],vr[None,None,:],s); q,_=divergence_axisymmetric(f,vx,vr,w)
  scale=np.max(f); self.assertLess(float(np.sqrt(np.mean(q*q))/scale),3e-3)
 def test_transformed_boundary_states(self):
  up,dn=normal_shock_states(5.); x=np.array([-8.,8.]); vx=np.array([up.u,dn.u]); vr=np.array([0.,1.])
  y,cx,cr=transformed_coordinates(x,vx,vr,up,dn,8.); self.assertEqual(y.shape,(2,1,1)); self.assertAlmostEqual(cx[0,0,0],0.); self.assertAlmostEqual(cx[1,1,0],0.); self.assertTrue(np.all(cr>=0))
if __name__=="__main__": unittest.main()

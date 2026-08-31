"""Conservative axisymmetric Dougherty--Fokker--Planck discretization."""
import numpy as np
from shock_physics import moments_and_fluxes

def simpson_weights(n,h):
    """Composite Simpson weights for odd uniform grids."""
    if n < 3 or n % 2 != 1: raise ValueError("Simpson rule requires odd n >= 3")
    c=np.ones(n); c[1:-1:2]=4; c[2:-1:2]=2
    return c*(h/3)

def divergence_axisymmetric(f,vx,vr,w,nu=1.):
    """Return Q(f)=nu div_v[(v-u)f+T grad_v f] on a uniform grid."""
    m=moments_and_fluxes(f,vx,vr,w); u=m["u"][:,None,None]; T=m["temperature"][:,None,None]
    dvx=float(vx[1]-vx[0]); dvr=float(vr[1]-vr[0]); X=vx[None,:,None]; R=vr[None,None,:]
    jx=(X-u)*f+T*np.gradient(f,dvx,axis=1,edge_order=2)
    jr=R*f+T*np.gradient(f,dvr,axis=2,edge_order=2)
    q=np.gradient(jx,dvx,axis=1,edge_order=2)
    rjr=R*jr; q+=np.divide(np.gradient(rjr,dvr,axis=2,edge_order=2),R,out=np.zeros_like(f),where=R>0)
    q[:,:,0]=2*np.gradient(jr,dvr,axis=2,edge_order=2)[:,:,0]
    return nu*q,m

def steady_residual(f,x,vx,vr,w,nu=1.):
    q,m=divergence_axisymmetric(f,vx,vr,w,nu)
    return vx[None,:,None]*np.gradient(f,x,axis=0,edge_order=2)-q,m

def transformed_coordinates(x,vx,vr,up,dn,length):
    y=np.asarray(x)/length; a=.5*(1-y); u=a*up.u+(1-a)*dn.u; T=a*up.temperature+(1-a)*dn.temperature
    return y[:,None,None],(vx[None,:,None]-u[:,None,None])/np.sqrt(T[:,None,None]),vr[None,None,:]/np.sqrt(T[:,None,None])

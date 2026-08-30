"""Conservative Rankine--Hugoniot/Mott--Smith physics for H1 Gate 0."""
from dataclasses import asdict, dataclass
import numpy as np

@dataclass(frozen=True)
class GasState:
    rho: float; u: float; temperature: float
    def as_dict(self): return asdict(self)

def normal_shock_states(mach, gamma=5/3):
    if mach <= 1 or gamma <= 1: raise ValueError("supersonic mach and gamma>1 required")
    u1=mach*np.sqrt(gamma); rr=(gamma+1)*mach**2/((gamma-1)*mach**2+2)
    pr=1+2*gamma*(mach**2-1)/(gamma+1)
    return GasState(1.,u1,1.),GasState(rr,u1/rr,pr/rr)

def analytic_fluxes(s):
    p=s.rho*s.temperature
    return np.array([s.rho*s.u,s.rho*s.u**2+p,s.u*(.5*s.rho*s.u**2+2.5*p)])

def maxwellian(vx,vr,s):
    return s.rho*(2*np.pi*s.temperature)**-1.5*np.exp(-((vx-s.u)**2+vr**2)/(2*s.temperature))

def mott_smith_distribution(x,vx,vr,up,dn,width):
    if width <= 0: raise ValueError("width must be positive")
    a=.5*(1-np.tanh(np.asarray(x)/width))[:,None,None]
    return a*maxwellian(vx[None,:,None],vr[None,None,:],up)+(1-a)*maxwellian(vx[None,:,None],vr[None,None,:],dn)

def axisymmetric_quadrature(vxmin,vxmax,nvx,vrmax,nvr):
    gx,gwx=np.polynomial.legendre.leggauss(nvx); gs,gws=np.polynomial.legendre.leggauss(nvr)
    vx=.5*((vxmax-vxmin)*gx+vxmax+vxmin); wx=.5*(vxmax-vxmin)*gwx
    s=.5*vrmax**2*(gs+1); vr=np.sqrt(s); ws=.5*vrmax**2*gws
    return vx,vr,wx[:,None]*(np.pi*ws[None,:])

def moments_and_fluxes(f,vx,vr,w):
    X=vx[None,:,None]; R2=vr[None,None,:]**2; wf=f*w[None]
    rho=wf.sum((1,2)); mom=(wf*X).sum((1,2)); u=mom/rho
    c2=(X-u[:,None,None])**2+R2; T=(wf*c2).sum((1,2))/(3*rho)
    pxx=(wf*(X-u[:,None,None])**2).sum((1,2)); pp=.5*(wf*R2).sum((1,2))
    qx=.5*(wf*(X-u[:,None,None])*c2).sum((1,2))
    return dict(rho=rho,u=u,temperature=T,pxx=pxx,pperp=pp,qx=qx,mass_flux=mom,
      momentum_flux=(wf*X**2).sum((1,2)),energy_flux=.5*(wf*X*(X**2+R2)).sum((1,2)))

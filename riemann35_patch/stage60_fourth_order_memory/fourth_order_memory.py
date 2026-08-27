"""Measure within-population fourth-order memory erased by Gaussian recompression."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from hyqmom_fp import macroscopic_state
from hyqmom_fp.collision import coefficients_from_weighted_nodes
from hyqmom_fp.two_population import _gauss_hermite_mixture_nodes
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import PersistentGaussianMixtureState, persistent_gaussian_mixture_moments

@dataclass(frozen=True)
class FourthOrderMemory:
    total_excess: float
    radial_excess: float
    anisotropic_excess: float
    max_population: float
    affine_scale: float


def _gaussian_fourth(cov: np.ndarray) -> np.ndarray:
    return (np.einsum('ij,kl->ijkl',cov,cov)+np.einsum('ik,jl->ijkl',cov,cov)+np.einsum('il,jk->ijkl',cov,cov))


def fourth_order_memory_defect(populations: PersistentGaussianMixtureState, dt: float, tau: float, *, prandtl: float=2/3, quadrature_nodes: int=5) -> FourthOrderMemory:
    if dt<=0 or tau<=0: raise ValueError('dt and tau must be positive')
    incoming=persistent_gaussian_mixture_moments(populations); macro=macroscopic_state(incoming)
    weights,nodes=_gauss_hermite_mixture_nodes(populations.probabilities,populations.means,populations.covariances,populations.rho,quadrature_nodes)
    coeff=coefficients_from_weighted_nodes(nodes,weights,tau=tau,prandtl=prandtl)
    prob=weights/populations.rho; peculiar=nodes-macro.velocity; c2=np.einsum('ni,ni->n',peculiar,peculiar)
    nonlinear=peculiar@coeff.C.T
    nonlinear+=(c2-3*macro.theta)[:,None]*coeff.gamma
    nonlinear+=coeff.beta*(c2[:,None]*peculiar-2*macro.heat_flux[None,:]/populations.rho)
    mean_n2=float(np.dot(prob,np.einsum('ni,ni->n',nonlinear,nonlinear))); mean_cn=float(np.dot(prob,np.einsum('ni,ni->n',peculiar,nonlinear)))
    r=float(np.exp(-dt/tau)); r2=r*r
    alpha2=1+tau/(3*macro.theta)*(tau*(1-r)**2*mean_n2+2*(r-r2)*mean_cn)
    if alpha2<=1e-12: raise FloatingPointError('nonpositive alpha squared')
    alpha=float(np.sqrt(alpha2)); noise=macro.theta*(1-r2)/alpha**2
    mapped=macro.velocity+(r*peculiar+(1-r)*tau*nonlinear)/alpha
    npp=quadrature_nodes**3; means=[]; covs=[]; excess=[]
    for k,pcomp in enumerate(populations.probabilities):
        sl=slice(k*npp,(k+1)*npp); lp=prob[sl]/pcomp; x=mapped[sl]; mu=np.sum(lp[:,None]*x,axis=0); c=x-mu
        cov=np.einsum('n,ni,nj->ij',lp,c,c)+noise*np.eye(3)
        raw4=np.einsum('n,ni,nj,nk,nl->ijkl',lp,c,c,c,c)
        # convolution with isotropic Gaussian noise: add 6 second*noise + 3 noise^2 terms
        eye=np.eye(3)
        raw4 += noise*(np.einsum('ij,kl->ijkl',cov-noise*eye,eye)+np.einsum('ik,jl->ijkl',cov-noise*eye,eye)+np.einsum('il,jk->ijkl',cov-noise*eye,eye)+np.einsum('jk,il->ijkl',cov-noise*eye,eye)+np.einsum('jl,ik->ijkl',cov-noise*eye,eye)+np.einsum('kl,ij->ijkl',cov-noise*eye,eye))
        raw4 += noise**2*(np.einsum('ij,kl->ijkl',eye,eye)+np.einsum('ik,jl->ijkl',eye,eye)+np.einsum('il,jk->ijkl',eye,eye))
        means.append(mu); covs.append(cov); excess.append(raw4-_gaussian_fourth(cov))
    means=np.asarray(means); covs=np.asarray(covs); excess=np.asarray(excess)
    mixmu=np.sum(populations.probabilities[:,None]*means,axis=0); mixcov=np.zeros((3,3))
    for p,m,cov in zip(populations.probabilities,means,covs):
        d=m-mixmu; mixcov+=p*(cov+np.outer(d,d))
    mapped_theta=float(np.trace(mixcov)/3); scale=float(np.sqrt(macro.theta/mapped_theta))
    scaled=scale**4*excess; total=populations.rho*np.einsum('p,pijkl->ijkl',populations.probabilities,scaled)
    radial=float(np.einsum('iijj->',total)); iso=radial/15.0*(np.einsum('ij,kl->ijkl',np.eye(3),np.eye(3))+np.einsum('ik,jl->ijkl',np.eye(3),np.eye(3))+np.einsum('il,jk->ijkl',np.eye(3),np.eye(3)))
    natural=max(populations.rho*macro.theta**2,1e-14)
    popnorm=np.linalg.norm(scaled.reshape(len(scaled),-1),axis=1)
    return FourthOrderMemory(float(np.linalg.norm(total)/natural),abs(radial)/natural,float(np.linalg.norm(total-iso)/natural),float(np.max(popnorm)/max(macro.theta**2,1e-14)),scale)

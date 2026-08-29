from __future__ import annotations

import numpy as np

from hyqmom_fp import HYQMOM_35_INDICES
from hyqmom_fp.two_population import _gauss_hermite_mixture_nodes
from riemann35_patch.stage55_closure_source_audit.run_closure_method import _direct_node_source
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import (
    PersistentGaussianMixtureState,
    persistent_gaussian_mixture_fp_step,
    persistent_gaussian_mixture_moments,
)

ORDER = np.asarray([sum(i) for i in HYQMOM_35_INDICES])
ACTIVE = np.where((ORDER >= 1) & (ORDER <= 4))[0]
FOURTH = np.where(ORDER == 4)[0]
PAIRS = ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))


def _source(state: PersistentGaussianMixtureState, tau: float, prandtl: float, quadrature_nodes: int) -> np.ndarray:
    moments = persistent_gaussian_mixture_moments(state)
    weights, nodes = _gauss_hermite_mixture_nodes(
        state.probabilities, state.means, state.covariances, state.rho, quadrature_nodes
    )
    return _direct_node_source(moments, nodes, weights, tau=tau, prandtl=prandtl)


def _pack(state: PersistentGaussianMixtureState) -> np.ndarray:
    rows=[]
    for mean,cov in zip(state.means,state.covariances):
        rows.extend(mean.tolist())
        rows.extend([cov[i,j] for i,j in PAIRS])
    return np.asarray(rows,float)


def _unpack(template: PersistentGaussianMixtureState, x: np.ndarray) -> PersistentGaussianMixtureState:
    means=[]; covs=[]; p=0
    for _ in range(template.probabilities.size):
        means.append(x[p:p+3]); p+=3
        C=np.zeros((3,3))
        for i,j in PAIRS:
            C[i,j]=C[j,i]=x[p]; p+=1
        covs.append(C)
    return PersistentGaussianMixtureState(
        rho=template.rho,
        probabilities=template.probabilities.copy(),
        means=np.asarray(means),
        covariances=np.asarray(covs),
    )


def _valid(state: PersistentGaussianMixtureState) -> bool:
    return all(np.min(np.linalg.eigvalsh(C)) > 1e-12 for C in state.covariances)


def generator_consistent_step(
    state: PersistentGaussianMixtureState,
    dt: float,
    tau: float,
    *,
    prandtl: float = 2/3,
    quadrature_nodes: int = 5,
) -> tuple[PersistentGaussianMixtureState, np.ndarray, dict[str,float]]:
    """Stage-57 step plus a minimal linearized correction enforcing the exact order-4 generator.

    Orders 1-3 are constrained to remain at the Stage-57 values. Only the order-4
    increment is corrected to dt*S4 from the continuous cubic-FP generator evaluated
    at the incoming positive mixture. The correction acts on labelled Gaussian means
    and covariances with fixed positive weights and uses an SPD line search.
    """
    incoming = persistent_gaussian_mixture_moments(state)
    exact_source = _source(state, tau, prandtl, quadrature_nodes)
    base_state, base_moments, _ = persistent_gaussian_mixture_fp_step(
        state, dt, tau, prandtl=prandtl, quadrature_nodes=quadrature_nodes
    )
    target = base_moments.copy()
    target[FOURTH] = incoming[FOURTH] + dt * exact_source[FOURTH]
    rhs = target[ACTIVE] - base_moments[ACTIVE]

    x0 = _pack(base_state)
    n=x0.size
    J=np.zeros((ACTIVE.size,n))
    for j in range(n):
        h=1e-7*max(1.0,abs(x0[j]))
        xp=x0.copy(); xm=x0.copy(); xp[j]+=h; xm[j]-=h
        sp=_unpack(base_state,xp); sm=_unpack(base_state,xm)
        if not _valid(sp) or not _valid(sm):
            h*=0.1; xp=x0.copy(); xm=x0.copy(); xp[j]+=h; xm[j]-=h
            sp=_unpack(base_state,xp); sm=_unpack(base_state,xm)
        mp=persistent_gaussian_mixture_moments(sp)[ACTIVE]
        mm=persistent_gaussian_mixture_moments(sm)[ACTIVE]
        J[:,j]=(mp-mm)/(2*h)

    row_scale=np.maximum(np.linalg.norm(J,axis=1),1e-12)
    dx=np.linalg.lstsq(J/row_scale[:,None], rhs/row_scale, rcond=1e-11)[0]
    frac=1.0
    corrected=base_state
    for _ in range(50):
        trial=_unpack(base_state,x0+frac*dx)
        if _valid(trial):
            corrected=trial; break
        frac*=0.5
    corrected_moments=persistent_gaussian_mixture_moments(corrected)
    scale=max(np.linalg.norm(exact_source[FOURTH]),1e-14)
    achieved=(corrected_moments[FOURTH]-incoming[FOURTH])/dt
    fourth_source_error=float(np.linalg.norm(achieved-exact_source[FOURTH])/scale)
    lower_change=float(np.linalg.norm(corrected_moments[ACTIVE[ORDER[ACTIVE] <= 3]]-base_moments[ACTIVE[ORDER[ACTIVE] <= 3]])/max(np.linalg.norm(base_moments[ACTIVE[ORDER[ACTIVE] <= 3]]),1e-14))
    return corrected, corrected_moments, {
        'projection_fraction': float(frac),
        'fourth_source_error': fourth_source_error,
        'lower_order_change': lower_change,
        'correction_rms': float(np.linalg.norm(frac*dx)/np.sqrt(dx.size)),
    }

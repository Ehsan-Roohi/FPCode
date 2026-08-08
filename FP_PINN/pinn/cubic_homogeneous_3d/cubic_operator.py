"""Canonical homogeneous cubic Fokker--Planck closure.

The implementation follows the moment ordering used by the legacy FPCode
solver, but makes the cubic scalar coefficient explicit in the drift.  All
quantities are nondimensional.  The default choice ``nu=1`` gives
``tau=1`` and an equilibrium temperature ``theta=1`` for the supplied tests.

Unknown ordering in the 9 x 9 closure system is

    [C_xx, C_xy, C_xz, C_yy, C_yz, C_zz, Gamma_x, Gamma_y, Gamma_z].

The complete drift in peculiar velocity c is

    a_i = -nu*c_i + C_ij*c_j + Gamma_i*(|c|^2-DM2)
          + lambda*(c_i*|c|^2-Q_i).

Here Q=<c |c|^2>=2q/rho.  The last two centered terms preserve momentum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


CASE_NAMES: Tuple[str, ...] = ("equilibrium", "stress", "heat_flux")
SQRT_2PI = np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class Moments:
    mass: float
    mean: np.ndarray
    pij: np.ndarray
    q: np.ndarray
    m3: np.ndarray
    m4: np.ndarray
    m5: np.ndarray
    dm2: float
    dm4: float

    @property
    def theta(self) -> float:
        return self.dm2 / 3.0


@dataclass(frozen=True)
class Closure:
    matrix: np.ndarray
    gamma: np.ndarray
    cubic_lambda: float
    vector: np.ndarray
    condition_number: float
    linear_residual: float


def _normal_logpdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return -0.5 * (np.log(2.0 * np.pi * variance) + (x - mean) ** 2 / variance)


def _logaddexp_weighted(log_a: np.ndarray, weight_a: float,
                        log_b: np.ndarray, weight_b: float) -> np.ndarray:
    return np.logaddexp(np.log(weight_a) + log_a, np.log(weight_b) + log_b)


def validate_case(case: str) -> str:
    if case not in CASE_NAMES:
        raise ValueError(f"Unknown case {case!r}; choose one of {CASE_NAMES}")
    return case


def equilibrium_logpdf(c: np.ndarray) -> np.ndarray:
    """Unit-density, zero-mean, unit-temperature Maxwellian."""
    c = np.asarray(c, dtype=np.float64)
    return -1.5 * np.log(2.0 * np.pi) - 0.5 * np.sum(c * c, axis=-1)


def initial_logpdf(case: str, c: np.ndarray) -> np.ndarray:
    """Analytic positive initial density for each Stage-2 benchmark."""
    validate_case(case)
    c = np.asarray(c, dtype=np.float64)
    if c.shape[-1] != 3:
        raise ValueError("c must have final dimension three")

    if case == "equilibrium":
        return equilibrium_logpdf(c)

    if case == "stress":
        # Trace(covariance)=3, hence theta=1, but the stress is anisotropic.
        variances = np.array([1.6, 0.9, 0.5], dtype=np.float64)
        return np.sum(
            -0.5 * (np.log(2.0 * np.pi * variances) + c * c / variances),
            axis=-1,
        )

    # Positive skew two-Gaussian mixture in c_x.  Its exact moments are
    # <c_x>=0, <c_x^2>=1, <c_x^3>=1/4.  The transverse components are N(0,1).
    log_x_a = _normal_logpdf(c[..., 0], mean=1.0, variance=0.5)
    log_x_b = _normal_logpdf(c[..., 0], mean=-0.5, variance=0.5)
    log_x = _logaddexp_weighted(log_x_a, 1.0 / 3.0, log_x_b, 2.0 / 3.0)
    return log_x + _normal_logpdf(c[..., 1], 0.0, 1.0) + _normal_logpdf(
        c[..., 2], 0.0, 1.0
    )


def proposal_logpdf(case: str, c: np.ndarray) -> np.ndarray:
    """50:50 initial/equilibrium importance-sampling proposal."""
    return np.logaddexp(
        np.log(0.5) + initial_logpdf(case, c),
        np.log(0.5) + equilibrium_logpdf(c),
    )


def sample_initial(case: str, size: int, rng: np.random.Generator) -> np.ndarray:
    validate_case(case)
    if size <= 0:
        raise ValueError("size must be positive")
    if case == "equilibrium":
        return rng.normal(size=(size, 3))
    if case == "stress":
        return rng.normal(size=(size, 3)) * np.sqrt([1.6, 0.9, 0.5])

    component_a = rng.random(size) < (1.0 / 3.0)
    means = np.where(component_a, 1.0, -0.5)
    result = rng.normal(size=(size, 3))
    result[:, 0] = means + np.sqrt(0.5) * rng.normal(size=size)
    return result


def sample_proposal(case: str, size: int, rng: np.random.Generator) -> np.ndarray:
    choose_initial = rng.random(size) < 0.5
    values = rng.normal(size=(size, 3))
    n_initial = int(np.count_nonzero(choose_initial))
    if n_initial:
        values[choose_initial] = sample_initial(case, n_initial, rng)
    return values


def analytic_initial_summary(case: str) -> Dict[str, np.ndarray | float]:
    validate_case(case)
    if case == "equilibrium":
        pij = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        q = np.zeros(3)
    elif case == "stress":
        pij = np.array([1.6, 0.0, 0.0, 0.9, 0.0, 0.5])
        q = np.zeros(3)
    else:
        pij = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        q = np.array([0.25, 0.0, 0.0])
    return {"mass": 1.0, "mean": np.zeros(3), "dm2": 3.0, "pij": pij, "q": q}


def moments_from_samples(c: np.ndarray, weights: np.ndarray | None = None) -> Moments:
    """Return normalized peculiar-velocity moments through fifth order."""
    c = np.asarray(c, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("c must have shape (n, 3)")
    if not np.all(np.isfinite(c)):
        raise ValueError("c contains non-finite values")

    if weights is None:
        raw = np.full(c.shape[0], 1.0 / c.shape[0], dtype=np.float64)
        mass = 1.0
    else:
        raw = np.asarray(weights, dtype=np.float64).reshape(-1)
        if raw.shape[0] != c.shape[0] or np.any(raw < 0.0):
            raise ValueError("weights must be nonnegative with shape (n,)")
        mass = float(np.sum(raw))
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("weights must have positive finite mass")
        raw = raw / mass

    mean = np.sum(raw[:, None] * c, axis=0)
    v = c - mean
    x, y, z = v.T
    r2 = np.sum(v * v, axis=1)

    def avg(value: np.ndarray) -> float:
        return float(np.sum(raw * value))

    pij = np.array(
        [avg(x * x), avg(x * y), avg(x * z), avg(y * y), avg(y * z), avg(z * z)]
    )
    q = np.array([avg(x * r2), avg(y * r2), avg(z * r2)])
    m3 = np.array(
        [
            avg(x**3), avg(x * x * y), avg(x * x * z), avg(x * y * y),
            avg(x * y * z), avg(x * z * z), avg(y**3), avg(y * y * z),
            avg(y * z * z), avg(z**3),
        ]
    )
    # Important: every M4 component includes r^2, including the yz entry.
    m4 = np.array(
        [avg(x * x * r2), avg(x * y * r2), avg(x * z * r2),
         avg(y * y * r2), avg(y * z * r2), avg(z * z * r2)]
    )
    r4 = r2 * r2
    m5 = np.array([avg(x * r4), avg(y * r4), avg(z * r4)])
    dm2 = avg(r2)
    dm4 = avg(r4)
    return Moments(mass, mean, pij, q, m3, m4, m5, dm2, dm4)


def deviatoric_stress_norm_squared(moments: Moments) -> float:
    p = moments.pij
    third = moments.dm2 / 3.0
    return float(
        (p[0] - third) ** 2 + (p[3] - third) ** 2 + (p[5] - third) ** 2
        + 2.0 * (p[1] ** 2 + p[2] ** 2 + p[4] ** 2)
    )


def cubic_lambda(moments: Moments, nu: float = 1.0) -> float:
    """Analytic stabilizing cubic coefficient (non-positive sign convention)."""
    denominator = max(moments.dm2, np.finfo(np.float64).tiny) ** 3.5
    return -deviatoric_stress_norm_squared(moments) * float(nu) / denominator


def build_closure_system(
    moments: Moments, nu: float = 1.0, nubol: float | None = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build the canonical 9 x 9 C/Gamma moment-closure system."""
    if nubol is None:
        nubol = 2.0 * nu
    p = moments.pij
    q = moments.q
    m3 = moments.m3
    m4 = moments.m4
    m5 = moments.m5
    d2, d4 = moments.dm2, moments.dm4
    lam = cubic_lambda(moments, nu)
    lhs = np.zeros((9, 9), dtype=np.float64)

    lhs[0, [0, 1, 2]] = [2*p[0], 2*p[1], 2*p[2]]
    lhs[1, [0, 1, 2, 3, 4]] = [p[1], p[0]+p[3], p[4], p[1], p[2]]
    lhs[2, [0, 1, 2, 4, 5]] = [p[2], p[4], p[0]+p[5], p[1], p[2]]
    lhs[3, [1, 3, 4]] = [2*p[1], 2*p[3], 2*p[4]]
    lhs[4, [1, 2, 3, 4, 5]] = [p[2], p[1], p[4], p[3]+p[5], p[4]]
    lhs[5, [2, 4, 5]] = [2*p[2], 2*p[4], 2*p[5]]

    lhs[0, 6] = 2*q[0]
    lhs[1, [6, 7]] = [q[1], q[0]]
    lhs[2, [6, 8]] = [q[2], q[0]]
    lhs[3, 7] = 2*q[1]
    lhs[4, [7, 8]] = [q[2], q[1]]
    lhs[5, 8] = 2*q[2]

    lhs[6, 0], lhs[7, 0], lhs[8, 0] = q[0]+2*m3[0], 2*m3[1], 2*m3[2]
    lhs[6, 1], lhs[7, 1], lhs[8, 1] = q[1]+4*m3[1], q[0]+4*m3[3], 4*m3[4]
    lhs[6, 2], lhs[7, 2], lhs[8, 2] = q[2]+4*m3[2], 4*m3[4], q[0]+4*m3[5]
    lhs[6, 3], lhs[7, 3], lhs[8, 3] = 2*m3[3], q[1]+2*m3[6], 2*m3[7]
    lhs[6, 4], lhs[7, 4], lhs[8, 4] = 4*m3[4], q[2]+4*m3[7], q[1]+4*m3[8]
    lhs[6, 5], lhs[7, 5], lhs[8, 5] = 2*m3[5], 2*m3[8], q[2]+2*m3[9]

    d4_centered = d4 - d2*d2
    lhs[6:9, 6:9] = np.array(
        [
            [d4_centered+2*m4[0]-2*d2*p[0], 2*m4[1]-2*d2*p[1], 2*m4[2]-2*d2*p[2]],
            [2*m4[1]-2*d2*p[1], d4_centered+2*m4[3]-2*d2*p[3], 2*m4[4]-2*d2*p[4]],
            [2*m4[2]-2*d2*p[2], 2*m4[4]-2*d2*p[4], d4_centered+2*m4[5]-2*d2*p[5]],
        ]
    )

    rhs = np.zeros(9, dtype=np.float64)
    rhs[:6] = -2.0 * lam * m4
    rhs[6] = -lam * (3*m5[0]-d2*q[0]-2*(p[0]*q[0]+p[1]*q[1]+p[2]*q[2]))
    rhs[7] = -lam * (3*m5[1]-d2*q[1]-2*(p[1]*q[0]+p[3]*q[1]+p[4]*q[2]))
    rhs[8] = -lam * (3*m5[2]-d2*q[2]-2*(p[2]*q[0]+p[4]*q[1]+p[5]*q[2]))
    rhs[6:] += (3.0*nu - (2.0/3.0)*nubol) * q
    return lhs, rhs, lam


def solve_closure(
    moments: Moments,
    nu: float = 1.0,
    nubol: float | None = None,
    regularization: float = 1.0e-10,
) -> Closure:
    lhs, rhs, lam = build_closure_system(moments, nu=nu, nubol=nubol)
    scale = max(1.0, float(np.linalg.norm(lhs, ord="fro") / 9.0))
    system = lhs + (regularization * scale) * np.eye(9)
    try:
        vector = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        vector = np.linalg.lstsq(system, rhs, rcond=1.0e-12)[0]
    matrix = np.array(
        [[vector[0], vector[1], vector[2]],
         [vector[1], vector[3], vector[4]],
         [vector[2], vector[4], vector[5]]]
    )
    residual = float(np.linalg.norm(lhs @ vector - rhs))
    return Closure(
        matrix=matrix,
        gamma=vector[6:].copy(),
        cubic_lambda=float(lam),
        vector=vector,
        condition_number=float(np.linalg.cond(system)),
        linear_residual=residual,
    )


def nonlinear_drift(c: np.ndarray, moments: Moments, closure: Closure) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    peculiar = c - moments.mean
    r2 = np.sum(peculiar * peculiar, axis=-1)
    return (
        peculiar @ closure.matrix.T
        + (r2 - moments.dm2)[..., None] * closure.gamma
        + closure.cubic_lambda * (peculiar * r2[..., None] - moments.q)
    )


def full_drift(c: np.ndarray, moments: Moments, closure: Closure, nu: float = 1.0) -> np.ndarray:
    peculiar = np.asarray(c, dtype=np.float64) - moments.mean
    return -float(nu) * peculiar + nonlinear_drift(c, moments, closure)


def drift_divergence(c: np.ndarray, closure: Closure, nu: float = 1.0) -> np.ndarray:
    """Analytic velocity divergence of the full drift."""
    c = np.asarray(c, dtype=np.float64)
    r2 = np.sum(c * c, axis=-1)
    return (
        -3.0 * float(nu)
        + np.trace(closure.matrix)
        + 2.0 * np.sum(c * closure.gamma, axis=-1)
        + 5.0 * closure.cubic_lambda * r2
    )


def ou_cubic_step(
    c: np.ndarray,
    moments: Moments,
    closure: Closure,
    dt: float,
    rng: np.random.Generator,
    nu: float = 1.0,
    preserve_invariants: bool = True,
    target_dm2: float | None = None,
) -> np.ndarray:
    """One legacy-compatible OU-exact/cubic-explicit particle step."""
    if dt <= 0.0 or nu <= 0.0:
        raise ValueError("dt and nu must be positive")
    peculiar = np.asarray(c, dtype=np.float64) - moments.mean
    nonlinear = nonlinear_drift(c, moments, closure)
    decay = np.exp(-nu * dt)
    response = (1.0 - decay) / nu
    sigma = np.sqrt(moments.theta * (1.0 - np.exp(-2.0 * nu * dt)))
    updated = decay * peculiar + response * nonlinear + sigma * rng.normal(size=peculiar.shape)

    if preserve_invariants:
        updated -= np.mean(updated, axis=0)
        current_dm2 = float(np.mean(np.sum(updated * updated, axis=1)))
        desired = moments.dm2 if target_dm2 is None else float(target_dm2)
        updated *= np.sqrt(desired / current_dm2)
    return updated + moments.mean


def maxwellian_log_residual(c: np.ndarray, nu: float = 1.0, theta: float = 1.0) -> np.ndarray:
    """Relative PDE residual for the exact Maxwellian/OU pair (unit test helper)."""
    c = np.asarray(c, dtype=np.float64)
    r2 = np.sum(c * c, axis=-1)
    div_a = -3.0 * nu
    a_dot_grad_logf = nu * r2 / theta
    diffusion_term = nu * theta * (-3.0 / theta + r2 / theta**2)
    return div_a + a_dot_grad_logf - diffusion_term

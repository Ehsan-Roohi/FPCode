"""Projection of FP collision dynamics onto HyQMOM's 35 moments.

This module evaluates the continuous-time velocity-space generator associated
with the cubic drift used in FPCode.  The retained state is the fourth-order
35-moment vector used by HyQMOM.jl.  A pluggable closure supplies M5/M6 when
the quadratic and cubic drift corrections require them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb
from typing import Callable, Sequence

import numpy as np

from .moments import (
    HYQMOM_35_INDICES,
    GaussianTailClosure,
    MacroscopicState,
    MultiIndex,
    central_moment,
    macroscopic_state,
    maxwellian_moments_35,
    moment_value,
)

MomentClosure = Callable[[MultiIndex, Sequence[float], MacroscopicState], float]


@dataclass(frozen=True)
class CubicFPCoefficients:
    """Coefficients of the continuous-time cubic FP velocity drift.

    The peculiar-velocity drift is

        a(c) = -c/tau + C*c + gamma*(|c|^2 - 3 theta)
               + beta*(|c|^2*c - 2 q/rho),

    and the isotropic velocity diffusion is ``theta/tau``.  This is the
    unclipped continuous-time counterpart of FPCode's particle update.
    """

    tau: float
    C: np.ndarray
    gamma: np.ndarray
    beta: float
    theta: float | None = None

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        c_matrix = np.asarray(self.C, dtype=float)
        gamma_vector = np.asarray(self.gamma, dtype=float)
        if c_matrix.shape != (3, 3):
            raise ValueError("C must have shape (3, 3)")
        if gamma_vector.shape != (3,):
            raise ValueError("gamma must have shape (3,)")
        if not np.all(np.isfinite(c_matrix)) or not np.all(np.isfinite(gamma_vector)):
            raise ValueError("FP coefficients must be finite")
        object.__setattr__(self, "C", c_matrix)
        object.__setattr__(self, "gamma", gamma_vector)

    @classmethod
    def ornstein_uhlenbeck(
        cls, tau: float, theta: float | None = None
    ) -> "CubicFPCoefficients":
        """Construct the linear FP/BGK-limit coefficient set."""

        return cls(
            tau=tau,
            C=np.zeros((3, 3)),
            gamma=np.zeros(3),
            beta=0.0,
            theta=theta,
        )


def coefficients_from_moments(
    moments: Sequence[float],
    tau: float,
    prandtl: float = 2.0 / 3.0,
    gamma_scale: float = 0.05,
    closure: MomentClosure | None = None,
) -> CubicFPCoefficients:
    """Solve FPCode's physical 9-by-9 cubic-drift coefficient system.

    The retained state supplies all central moments through degree four.  The
    fifth-order contraction required by the Gorji--Torrilhon--Jenny system is
    evaluated with ``closure``; the Stage-0 default is ``GaussianTailClosure``.
    ``gamma_scale`` is retained only for API compatibility with the superseded
    heuristic coefficient map and has no effect.
    """

    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if not 0.0 < prandtl <= 1.0:
        raise ValueError("prandtl must lie in (0, 1]")

    del gamma_scale
    vector = np.asarray(moments, dtype=float)
    state = macroscopic_state(vector)
    rho = state.rho
    pij = np.asarray(
        [
            state.covariance[0, 0],
            state.covariance[0, 1],
            state.covariance[0, 2],
            state.covariance[1, 1],
            state.covariance[1, 2],
            state.covariance[2, 2],
        ]
    )
    q = 2.0 * state.heat_flux / rho
    m3_indices = (
        (3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1),
        (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3),
    )
    m3 = np.asarray([central_moment(vector, index) / rho for index in m3_indices])
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    m4 = np.zeros(6)
    for position, (left, right) in enumerate(pairs):
        for squared in range(3):
            index = [0, 0, 0]
            index[left] += 1
            index[right] += 1
            index[squared] += 2
            m4[position] += central_moment(vector, tuple(index)) / rho

    moment_closure = closure if closure is not None else GaussianTailClosure()
    m5 = np.zeros(3)
    for direction in range(3):
        for first_squared in range(3):
            for second_squared in range(3):
                central_power = [0, 0, 0]
                central_power[direction] += 1
                central_power[first_squared] += 2
                central_power[second_squared] += 2
                m5[direction] += _raw_times_central_moment(
                    (0, 0, 0),
                    tuple(central_power),
                    vector,
                    state,
                    moment_closure,
                ) / rho

    return _coefficients_from_central_statistics(
        pij=pij,
        q=q,
        m3=m3,
        m4=m4,
        m5=m5,
        tau=tau,
        prandtl=prandtl,
    )


def _coefficients_from_central_statistics(
    *,
    pij: Sequence[float],
    q: Sequence[float],
    m3: Sequence[float],
    m4: Sequence[float],
    m5: Sequence[float],
    tau: float,
    prandtl: float,
    equilibrium_tolerance: float = 1.0e-10,
    regularization: float = 1.0e-12,
) -> CubicFPCoefficients:
    """Port the physical coefficient solve used by FPCode's full solver."""

    if tau <= 0.0 or not 0.0 < prandtl <= 1.0:
        raise ValueError("invalid collision time or Prandtl number")
    pij = np.asarray(pij, dtype=float)
    q = np.asarray(q, dtype=float)
    m3 = np.asarray(m3, dtype=float)
    m4 = np.asarray(m4, dtype=float)
    m5 = np.asarray(m5, dtype=float)
    if pij.shape != (6,) or q.shape != (3,) or m3.shape != (10,):
        raise ValueError("invalid second- or third-order central statistics")
    if m4.shape != (6,) or m5.shape != (3,):
        raise ValueError("invalid fourth- or fifth-order central statistics")
    if not all(np.all(np.isfinite(item)) for item in (pij, q, m3, m4, m5)):
        raise ValueError("central statistics must be finite")

    dm2 = float(pij[0] + pij[3] + pij[5])
    if dm2 <= 0.0:
        raise ValueError("central second moment must be positive")
    theta = dm2 / 3.0
    nu = 1.0 / tau
    dev = np.asarray(
        [
            pij[0] - dm2 / 3.0,
            pij[1],
            pij[2],
            pij[3] - dm2 / 3.0,
            pij[4],
            pij[5] - dm2 / 3.0,
        ]
    )
    dev_norm_squared = float(
        dev[0] ** 2 + dev[3] ** 2 + dev[5] ** 2
        + 2.0 * (dev[1] ** 2 + dev[2] ** 2 + dev[4] ** 2)
    )
    beta = -dev_norm_squared * nu / dm2**3.5
    normalized_nonequilibrium = max(
        np.sqrt(dev_norm_squared) / dm2,
        float(np.linalg.norm(q)) / dm2**1.5,
    )
    if normalized_nonequilibrium <= equilibrium_tolerance:
        return CubicFPCoefficients.ornstein_uhlenbeck(tau=tau, theta=theta)

    lhs = np.zeros((9, 9))
    lhs[0, 0:3] = 2.0 * pij[0:3]
    lhs[1, [0, 1, 2, 3, 4]] = [pij[1], pij[0] + pij[3], pij[4], pij[1], pij[2]]
    lhs[2, [0, 1, 2, 4, 5]] = [pij[2], pij[4], pij[0] + pij[5], pij[1], pij[2]]
    lhs[3, [1, 3, 4]] = [2.0 * pij[1], 2.0 * pij[3], 2.0 * pij[4]]
    lhs[4, [1, 2, 3, 4, 5]] = [pij[2], pij[1], pij[4], pij[3] + pij[5], pij[4]]
    lhs[5, [2, 4, 5]] = [2.0 * pij[2], 2.0 * pij[4], 2.0 * pij[5]]

    lhs[0, 6] = 2.0 * q[0]
    lhs[1, 6:8] = [q[1], q[0]]
    lhs[2, [6, 8]] = [q[2], q[0]]
    lhs[3, 7] = 2.0 * q[1]
    lhs[4, 7:9] = [q[2], q[1]]
    lhs[5, 8] = 2.0 * q[2]

    lhs[6, 0:6] = [q[0] + 2*m3[0], q[1] + 4*m3[1], q[2] + 4*m3[2], 2*m3[3], 4*m3[4], 2*m3[5]]
    lhs[7, 0:6] = [2*m3[1], q[0] + 4*m3[3], 4*m3[4], q[1] + 2*m3[6], q[2] + 4*m3[7], 2*m3[8]]
    lhs[8, 0:6] = [2*m3[2], 4*m3[4], q[0] + 4*m3[5], 2*m3[7], q[1] + 4*m3[8], q[2] + 2*m3[9]]

    dm4 = float(m4[0] + m4[3] + m4[5])
    dm4_term = dm4 - dm2**2
    lower = np.asarray(
        [
            [dm4_term + 2*m4[0] - 2*dm2*pij[0], 2*m4[1] - 2*dm2*pij[1], 2*m4[2] - 2*dm2*pij[2]],
            [2*m4[1] - 2*dm2*pij[1], dm4_term + 2*m4[3] - 2*dm2*pij[3], 2*m4[4] - 2*dm2*pij[4]],
            [2*m4[2] - 2*dm2*pij[2], 2*m4[4] - 2*dm2*pij[4], dm4_term + 2*m4[5] - 2*dm2*pij[5]],
        ]
    )
    lhs[6:9, 6:9] = lower

    rhs = np.zeros(9)
    rhs[:6] = -2.0 * beta * m4
    rhs[6] = -beta * (3*m5[0] - dm2*q[0] - 2*(pij[0]*q[0] + pij[1]*q[1] + pij[2]*q[2]))
    rhs[7] = -beta * (3*m5[1] - dm2*q[1] - 2*(pij[1]*q[0] + pij[3]*q[1] + pij[4]*q[2]))
    rhs[8] = -beta * (3*m5[2] - dm2*q[2] - 2*(pij[2]*q[0] + pij[4]*q[1] + pij[5]*q[2]))
    rhs[6:9] += (3.0 - 2.0 * prandtl) * nu * q

    matrix_scale = max(float(np.linalg.norm(lhs, ord=np.inf)), 1.0)
    regularized = lhs + regularization * matrix_scale * np.eye(9)
    try:
        solution = np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(regularized, rhs, rcond=1.0e-12)[0]
    if not np.all(np.isfinite(solution)):
        raise FloatingPointError("cubic FP coefficient solve produced NaN or infinity")
    C = np.asarray(
        [
            [solution[0], solution[1], solution[2]],
            [solution[1], solution[3], solution[4]],
            [solution[2], solution[4], solution[5]],
        ]
    )
    return CubicFPCoefficients(
        tau=tau,
        C=C,
        gamma=solution[6:9],
        beta=float(beta),
        theta=theta,
    )


def coefficients_from_weighted_nodes(
    velocities: Sequence[Sequence[float]],
    weights: Sequence[float] | None,
    tau: float,
    prandtl: float = 2.0 / 3.0,
) -> CubicFPCoefficients:
    """Solve the physical cubic-FP coefficients from a weighted velocity set."""

    nodes = np.asarray(velocities, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] == 0:
        raise ValueError("velocities must have shape (n, 3)")
    if weights is None:
        probabilities = np.full(nodes.shape[0], 1.0 / nodes.shape[0])
    else:
        probabilities = np.asarray(weights, dtype=float)
        if probabilities.shape != (nodes.shape[0],) or np.any(probabilities < 0.0):
            raise ValueError("invalid velocity weights")
        probabilities = probabilities / np.sum(probabilities)
    mean = np.sum(probabilities[:, None] * nodes, axis=0)
    c = nodes - mean
    c2 = np.einsum("ni,ni->n", c, c)
    covariance = np.einsum("n,ni,nj->ij", probabilities, c, c)
    pij = np.asarray([covariance[0, 0], covariance[0, 1], covariance[0, 2], covariance[1, 1], covariance[1, 2], covariance[2, 2]])
    q = np.einsum("n,ni,n->i", probabilities, c, c2)
    m3 = np.asarray([
        np.dot(probabilities, c[:, 0]**3), np.dot(probabilities, c[:, 0]**2*c[:, 1]),
        np.dot(probabilities, c[:, 0]**2*c[:, 2]), np.dot(probabilities, c[:, 0]*c[:, 1]**2),
        np.dot(probabilities, c[:, 0]*c[:, 1]*c[:, 2]), np.dot(probabilities, c[:, 0]*c[:, 2]**2),
        np.dot(probabilities, c[:, 1]**3), np.dot(probabilities, c[:, 1]**2*c[:, 2]),
        np.dot(probabilities, c[:, 1]*c[:, 2]**2), np.dot(probabilities, c[:, 2]**3),
    ])
    m4 = np.asarray([
        np.dot(probabilities, c[:, 0]*c[:, 0]*c2), np.dot(probabilities, c[:, 0]*c[:, 1]*c2),
        np.dot(probabilities, c[:, 0]*c[:, 2]*c2), np.dot(probabilities, c[:, 1]*c[:, 1]*c2),
        np.dot(probabilities, c[:, 1]*c[:, 2]*c2), np.dot(probabilities, c[:, 2]*c[:, 2]*c2),
    ])
    m5 = np.einsum("n,ni,n->i", probabilities, c, c2**2)
    return _coefficients_from_central_statistics(
        pij=pij, q=q, m3=m3, m4=m4, m5=m5, tau=tau, prandtl=prandtl
    )


def _shift(index: MultiIndex, direction: int, amount: int) -> MultiIndex:
    shifted = list(index)
    shifted[direction] += amount
    if shifted[direction] < 0:
        raise ValueError("negative moment exponent")
    return tuple(shifted)


def _raw_times_central_moment(
    raw_power: MultiIndex,
    central_power: MultiIndex,
    moments: Sequence[float],
    state: MacroscopicState,
    closure: MomentClosure,
) -> float:
    """Evaluate integral(v**raw_power * c**central_power * f dv)."""

    total = 0.0
    for px, py, pz in product(
        range(central_power[0] + 1),
        range(central_power[1] + 1),
        range(central_power[2] + 1),
    ):
        expanded = (px, py, pz)
        index = tuple(raw_power[d] + expanded[d] for d in range(3))
        coefficient = 1.0
        for direction in range(3):
            coefficient *= (
                comb(central_power[direction], expanded[direction])
                * (-state.velocity[direction])
                ** (central_power[direction] - expanded[direction])
            )
        total += coefficient * closure(index, moments, state)
    return float(total)


def _project_collision_invariants(source: np.ndarray) -> np.ndarray:
    """Remove roundoff/closure leakage from collision invariants."""

    projected = np.asarray(source, dtype=float).copy()
    index_to_position = {
        index: position for position, index in enumerate(HYQMOM_35_INDICES)
    }
    projected[index_to_position[(0, 0, 0)]] = 0.0
    projected[index_to_position[(1, 0, 0)]] = 0.0
    projected[index_to_position[(0, 1, 0)]] = 0.0
    projected[index_to_position[(0, 0, 1)]] = 0.0

    energy_positions = [
        index_to_position[(2, 0, 0)],
        index_to_position[(0, 2, 0)],
        index_to_position[(0, 0, 2)],
    ]
    energy_leak = float(sum(projected[position] for position in energy_positions))
    for position in energy_positions:
        projected[position] -= energy_leak / 3.0
    return projected


def projected_fp_collision_source(
    moments: Sequence[float],
    coefficients: CubicFPCoefficients,
    closure: MomentClosure | None = None,
    enforce_invariants: bool = True,
) -> np.ndarray:
    """Evaluate the cubic FP collision source for all 35 retained moments.

    ``GaussianTailClosure`` is the explicit Stage-0 default for M5/M6.  Pass a
    different callable to test a Grad-HyQMOM, quadrature, neural, or tabulated
    high-order closure without changing the collision projection.
    """

    vector = np.asarray(moments, dtype=float)
    if vector.shape != (35,):
        raise ValueError(f"expected a 35-moment vector, got {vector.shape}")
    state = macroscopic_state(vector)
    moment_closure = closure if closure is not None else GaussianTailClosure()

    source = np.asarray(
        [
            fp_collision_moment_source(
                alpha,
                vector,
                coefficients,
                closure=moment_closure,
                state=state,
            )
            for alpha in HYQMOM_35_INDICES
        ],
        dtype=float,
    )

    if enforce_invariants:
        source = _project_collision_invariants(source)
    return source


def fp_collision_moment_source(
    alpha: MultiIndex,
    moments: Sequence[float],
    coefficients: CubicFPCoefficients,
    *,
    closure: MomentClosure,
    state: MacroscopicState | None = None,
) -> float:
    """Evaluate the continuous cubic-FP source of one raw moment.

    Unlike :func:`projected_fp_collision_source`, ``alpha`` may have order
    above four.  The supplied closure must provide every moment requested by
    the cubic drift, up to ``sum(alpha)+2``.  This supports source-local
    dynamic M5/M6 experiments without changing the 35 transported moments.
    """

    vector = np.asarray(moments, dtype=float)
    local_state = state if state is not None else macroscopic_state(vector)
    theta = local_state.theta if coefficients.theta is None else coefficients.theta
    if theta <= 0.0:
        raise ValueError("FP diffusion temperature must be positive")
    diffusion = theta / coefficients.tau
    q_over_rho = local_state.heat_flux / local_state.rho
    zero = (0, 0, 0)
    value = 0.0
    for direction in range(3):
        exponent = alpha[direction]
        if exponent == 0:
            continue
        raw_power = _shift(alpha, direction, -1)
        central_direction = list(zero)
        central_direction[direction] = 1
        drift_expectation = -_raw_times_central_moment(
            raw_power,
            tuple(central_direction),
            vector,
            local_state,
            closure,
        ) / coefficients.tau

        for coupled_direction in range(3):
            central_linear = list(zero)
            central_linear[coupled_direction] = 1
            drift_expectation += coefficients.C[
                direction, coupled_direction
            ] * _raw_times_central_moment(
                raw_power,
                tuple(central_linear),
                vector,
                local_state,
                closure,
            )

        if coefficients.gamma[direction] != 0.0:
            quadratic = 0.0
            for squared_direction in range(3):
                central_quadratic = list(zero)
                central_quadratic[squared_direction] = 2
                quadratic += _raw_times_central_moment(
                    raw_power,
                    tuple(central_quadratic),
                    vector,
                    local_state,
                    closure,
                )
            drift_expectation += coefficients.gamma[direction] * (
                quadratic - 3.0 * theta * closure(raw_power, vector, local_state)
            )

        if coefficients.beta != 0.0:
            cubic = 0.0
            for squared_direction in range(3):
                central_cubic = list(zero)
                central_cubic[squared_direction] += 2
                central_cubic[direction] += 1
                cubic += _raw_times_central_moment(
                    raw_power,
                    tuple(central_cubic),
                    vector,
                    local_state,
                    closure,
                )
            drift_expectation += coefficients.beta * (
                cubic
                - 2.0
                * q_over_rho[direction]
                * closure(raw_power, vector, local_state)
            )

        value += exponent * drift_expectation
        if exponent >= 2:
            diffusion_index = _shift(alpha, direction, -2)
            value += (
                diffusion
                * exponent
                * (exponent - 1)
                * closure(diffusion_index, vector, local_state)
            )
    return float(value)


def bgk_collision_source(moments: Sequence[float], tau: float) -> np.ndarray:
    """Reference BGK source in the same 35-moment ordering."""

    if tau <= 0.0:
        raise ValueError("tau must be positive")
    vector = np.asarray(moments, dtype=float)
    state = macroscopic_state(vector)
    equilibrium = maxwellian_moments_35(
        state.rho, state.velocity, state.theta
    )
    return (equilibrium - vector) / tau

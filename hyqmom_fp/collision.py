"""Projection of FP collision dynamics onto HyQMOM's 35 moments.

This module evaluates the continuous-time velocity-space generator associated
with the cubic drift used in FPCode.  The retained state is the fourth-order
35-moment vector used by HyQMOM.jl.  A pluggable closure supplies M5/M6 when
the quadratic and cubic drift corrections require them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb, sqrt
from typing import Callable, Sequence

import numpy as np

from .moments import (
    HYQMOM_35_INDICES,
    GaussianTailClosure,
    MacroscopicState,
    MultiIndex,
    macroscopic_state,
    maxwellian_moments_35,
    moment_value,
)

MomentClosure = Callable[[MultiIndex, Sequence[float], MacroscopicState], float]


@dataclass(frozen=True)
class CubicFPCoefficients:
    """Coefficients of the continuous-time cubic FP velocity drift.

    The peculiar-velocity drift is

        a(c) = -c/tau + C*c + gamma*(|c|^2 - 5 theta)
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
) -> CubicFPCoefficients:
    """Reproduce FPCode's physics coefficient map from a 35-moment state.

    Nondimensional units ``mass = k_B = 1`` are used, so ``RT = theta`` and
    ``p = rho*theta``.  The stress sign convention matches
    ``FP_PINN/legacy_source/147CylFP.py``.
    """

    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if not 0.0 < prandtl <= 1.0:
        raise ValueError("prandtl must lie in (0, 1]")

    state = macroscopic_state(moments)
    inverse_tau = 1.0 / tau
    prandtl_factor = (1.0 - prandtl) / prandtl
    C = inverse_tau * state.stress / state.pressure
    beta = inverse_tau * prandtl_factor / (10.0 * state.theta**2)
    q_scale = state.rho * state.theta * sqrt(state.theta)
    gamma = (
        gamma_scale
        * inverse_tau
        * prandtl_factor
        * (state.heat_flux / q_scale)
        / state.theta
    )
    return CubicFPCoefficients(
        tau=tau,
        C=C,
        gamma=gamma,
        beta=beta,
        theta=state.theta,
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
    theta = state.theta if coefficients.theta is None else coefficients.theta
    if theta <= 0.0:
        raise ValueError("FP diffusion temperature must be positive")
    diffusion = theta / coefficients.tau
    q_over_rho = state.heat_flux / state.rho

    source = np.zeros(35, dtype=float)
    zero = (0, 0, 0)

    for position, alpha in enumerate(HYQMOM_35_INDICES):
        value = 0.0
        for direction in range(3):
            exponent = alpha[direction]
            if exponent == 0:
                continue
            raw_power = _shift(alpha, direction, -1)

            central_direction = list(zero)
            central_direction[direction] = 1
            drift_expectation = (
                -_raw_times_central_moment(
                    raw_power,
                    tuple(central_direction),
                    vector,
                    state,
                    moment_closure,
                )
                / coefficients.tau
            )

            for coupled_direction in range(3):
                central_linear = list(zero)
                central_linear[coupled_direction] = 1
                drift_expectation += coefficients.C[
                    direction, coupled_direction
                ] * _raw_times_central_moment(
                    raw_power,
                    tuple(central_linear),
                    vector,
                    state,
                    moment_closure,
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
                        state,
                        moment_closure,
                    )
                drift_expectation += coefficients.gamma[direction] * (
                    quadratic
                    - 5.0
                    * theta
                    * moment_closure(raw_power, vector, state)
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
                        state,
                        moment_closure,
                    )
                drift_expectation += coefficients.beta * (
                    cubic
                    - 2.0
                    * q_over_rho[direction]
                    * moment_closure(raw_power, vector, state)
                )

            value += exponent * drift_expectation

            if exponent >= 2:
                diffusion_index = _shift(alpha, direction, -2)
                value += (
                    diffusion
                    * exponent
                    * (exponent - 1)
                    * moment_closure(diffusion_index, vector, state)
                )

        source[position] = value

    if enforce_invariants:
        source = _project_collision_invariants(source)
    return source


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

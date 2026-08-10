"""Deterministic Hermite-moment reference for homogeneous cubic FP dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .collision import (
    coefficients_from_moments,
    fp_collision_moment_source,
)
from .mixture_closure import realizability_margin_35
from .moments import (
    HYQMOM_35_INDICES,
    MacroscopicState,
    macroscopic_state,
    multivariate_gaussian_raw_moment,
)


def total_degree_indices(maximum_order: int) -> tuple[tuple[int, int, int], ...]:
    if maximum_order < 0:
        raise ValueError("maximum order must be nonnegative")
    return tuple(
        (i, j, order - i - j)
        for order in range(maximum_order + 1)
        for i in range(order + 1)
        for j in range(order - i + 1)
    )


def _subindices(index: tuple[int, int, int]):
    for i in range(index[0] + 1):
        for j in range(index[1] + 1):
            for k in range(index[2] + 1):
                yield (i, j, k)


def _binomial_multi(
    upper: tuple[int, int, int], lower: tuple[int, int, int]
) -> int:
    from math import comb

    return int(np.prod([comb(upper[d], lower[d]) for d in range(3)]))


@dataclass(frozen=True)
class HermiteMomentState:
    """Raw moments through a chosen total degree."""

    maximum_order: int
    indices: tuple[tuple[int, int, int], ...]
    values: np.ndarray


@dataclass(frozen=True)
class HermiteStepDiagnostics:
    """Diagnostics for one SSP-RK2 Hermite reference step."""

    realizability_margin: float
    source_norm: float


class HermiteGalerkinTailClosure:
    """Set Hermite coefficients above the retained spectral order to zero."""

    def __init__(
        self,
        dynamic: HermiteMomentState,
        first_35: np.ndarray,
        state: MacroscopicState | None = None,
    ) -> None:
        self.dynamic = dynamic
        self.position = {index: position for position, index in enumerate(dynamic.indices)}
        self.first_35 = np.asarray(first_35, dtype=float)
        self.state = state if state is not None else macroscopic_state(first_35)
        self._central_cache: dict[tuple[int, int, int], float] = {}
        self._hermite_cache: dict[tuple[int, int, int], float] = {}
        self._standard_cache: dict[tuple[int, int, int], float] = {}
        self._raw_cache: dict[tuple[int, int, int], float] = {}

    def _stored_raw(self, index: tuple[int, int, int]) -> float:
        return float(self.dynamic.values[self.position[index]])

    def _central_standardized(self, index: tuple[int, int, int]) -> float:
        if index in self._central_cache:
            return self._central_cache[index]
        value = 0.0
        for beta in _subindices(index):
            coefficient = _binomial_multi(index, beta)
            mean_factor = np.prod(
                [
                    (-self.state.velocity[d]) ** (index[d] - beta[d])
                    for d in range(3)
                ]
            )
            value += coefficient * mean_factor * self._stored_raw(beta)
        value /= self.state.rho * self.state.theta ** (sum(index) / 2.0)
        self._central_cache[index] = float(value)
        return float(value)

    def _hermite_coefficient(self, index: tuple[int, int, int]) -> float:
        if sum(index) > self.dynamic.maximum_order:
            return 0.0
        if index in self._hermite_cache:
            return self._hermite_cache[index]
        from math import factorial

        value = 0.0
        for mx in range(index[0] // 2 + 1):
            for my in range(index[1] // 2 + 1):
                for mz in range(index[2] // 2 + 1):
                    m = (mx, my, mz)
                    reduced = tuple(index[d] - 2 * m[d] for d in range(3))
                    coefficient = 1.0
                    for direction in range(3):
                        coefficient *= (
                            factorial(index[direction])
                            * (-1.0) ** m[direction]
                            / (
                                2.0 ** m[direction]
                                * factorial(m[direction])
                                * factorial(reduced[direction])
                            )
                        )
                    value += coefficient * self._central_standardized(reduced)
        self._hermite_cache[index] = float(value)
        return float(value)

    def _standardized_moment(self, index: tuple[int, int, int]) -> float:
        if index in self._standard_cache:
            return self._standard_cache[index]
        from math import factorial

        value = 0.0
        for mx in range(index[0] // 2 + 1):
            for my in range(index[1] // 2 + 1):
                for mz in range(index[2] // 2 + 1):
                    m = (mx, my, mz)
                    reduced = tuple(index[d] - 2 * m[d] for d in range(3))
                    coefficient = 1.0
                    for direction in range(3):
                        coefficient *= factorial(index[direction]) / (
                            2.0 ** m[direction]
                            * factorial(m[direction])
                            * factorial(reduced[direction])
                        )
                    value += coefficient * self._hermite_coefficient(reduced)
        self._standard_cache[index] = float(value)
        return float(value)

    def _tail_raw(self, index: tuple[int, int, int]) -> float:
        if index in self._raw_cache:
            return self._raw_cache[index]
        value = 0.0
        for beta in _subindices(index):
            coefficient = _binomial_multi(index, beta)
            mean_factor = np.prod(
                [
                    self.state.velocity[d] ** (index[d] - beta[d])
                    for d in range(3)
                ]
            )
            value += (
                coefficient
                * mean_factor
                * self.state.theta ** (sum(beta) / 2.0)
                * self._standardized_moment(beta)
            )
        value *= self.state.rho
        self._raw_cache[index] = float(value)
        return float(value)

    def __call__(
        self,
        index: tuple[int, int, int],
        moments,
        state: MacroscopicState | None = None,
    ) -> float:
        del moments, state
        if sum(index) <= self.dynamic.maximum_order:
            return self._stored_raw(index)
        if sum(index) > self.dynamic.maximum_order + 2:
            raise ValueError("Hermite closure only supplies two tail orders")
        return self._tail_raw(index)


def initialize_hermite_moment_state(
    components,
    maximum_order: int,
) -> HermiteMomentState:
    """Initialize the spectral moments analytically from Gaussian components."""

    indices = total_degree_indices(maximum_order)
    values = np.asarray(
        [
            sum(
                weight * multivariate_gaussian_raw_moment(index, mean, covariance)
                for weight, mean, covariance in components
            )
            for index in indices
        ],
        dtype=float,
    )
    return HermiteMomentState(maximum_order, indices, values)


def first_35_from_hermite_state(dynamic: HermiteMomentState) -> np.ndarray:
    position = {index: number for number, index in enumerate(dynamic.indices)}
    return np.asarray([dynamic.values[position[index]] for index in HYQMOM_35_INDICES])


def _rhs(
    dynamic: HermiteMomentState,
    tau: float,
    prandtl: float,
) -> np.ndarray:
    first_35 = first_35_from_hermite_state(dynamic)
    state = macroscopic_state(first_35)
    closure = HermiteGalerkinTailClosure(dynamic, first_35, state)
    coefficients = coefficients_from_moments(
        first_35,
        tau=tau,
        prandtl=prandtl,
        closure=closure,
    )
    source = np.asarray(
        [
            fp_collision_moment_source(
                index,
                first_35,
                coefficients,
                closure=closure,
                state=state,
            )
            for index in dynamic.indices
        ],
        dtype=float,
    )
    position = {index: number for number, index in enumerate(dynamic.indices)}
    for index in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        source[position[index]] = 0.0
    energy = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    leak = sum(source[position[index]] for index in energy)
    for index in energy:
        source[position[index]] -= leak / 3.0
    return source


def hermite_ssprk2_step(
    dynamic: HermiteMomentState,
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
) -> tuple[HermiteMomentState, HermiteStepDiagnostics]:
    """Advance the deterministic Hermite hierarchy with SSP-RK2."""

    if dt <= 0.0 or tau <= 0.0:
        raise ValueError("dt and tau must be positive")
    initial = dynamic.values
    source0 = _rhs(dynamic, tau, prandtl)
    stage1 = HermiteMomentState(
        dynamic.maximum_order,
        dynamic.indices,
        initial + dt * source0,
    )
    source1 = _rhs(stage1, tau, prandtl)
    values = 0.5 * initial + 0.5 * (stage1.values + dt * source1)
    position = {index: number for number, index in enumerate(dynamic.indices)}
    for index in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        values[position[index]] = initial[position[index]]
    energy = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    energy_error = sum(initial[position[index]] for index in energy) - sum(
        values[position[index]] for index in energy
    )
    for index in energy:
        values[position[index]] += energy_error / 3.0
    updated = HermiteMomentState(dynamic.maximum_order, dynamic.indices, values)
    first_35 = first_35_from_hermite_state(updated)
    margin = realizability_margin_35(first_35)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Hermite reference produced NaN or infinity")
    return updated, HermiteStepDiagnostics(
        realizability_margin=float(margin),
        source_norm=float(np.linalg.norm(source0)),
    )

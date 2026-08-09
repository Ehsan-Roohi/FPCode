"""Moment conventions shared by HyQMOM.jl and the FP prototype.

The ordering below is copied from the public ``comp-physics/HyQMOM.jl``
35-moment solver.  All moments are *raw integrated moments*,

    M_ijk = integral(vx**i * vy**j * vz**k * f(v) dv).

The module deliberately keeps this conversion layer independent of either
solver so arrays can be exchanged without silent index permutations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import comb
from typing import Iterable, Mapping, Sequence

import numpy as np

MultiIndex = tuple[int, int, int]


HYQMOM_35_INDICES: tuple[MultiIndex, ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (3, 0, 0),
    (4, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (2, 1, 0),
    (3, 1, 0),
    (0, 2, 0),
    (1, 2, 0),
    (2, 2, 0),
    (0, 3, 0),
    (1, 3, 0),
    (0, 4, 0),
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 1),
    (3, 0, 1),
    (0, 0, 2),
    (1, 0, 2),
    (2, 0, 2),
    (0, 0, 3),
    (1, 0, 3),
    (0, 0, 4),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 1),
    (0, 2, 1),
    (1, 2, 1),
    (0, 3, 1),
    (0, 1, 2),
    (1, 1, 2),
    (0, 1, 3),
    (0, 2, 2),
)

HYQMOM_35_NAMES: tuple[str, ...] = tuple(
    f"M{i}{j}{k}" for i, j, k in HYQMOM_35_INDICES
)

_POSITION: dict[MultiIndex, int] = {
    index: position for position, index in enumerate(HYQMOM_35_INDICES)
}


def _as_moment_vector(moments: Sequence[float]) -> np.ndarray:
    vector = np.asarray(moments, dtype=float)
    if vector.shape != (35,):
        raise ValueError(f"expected a 35-moment vector, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("moment vector contains NaN or infinity")
    return vector


def moment_value(moments: Sequence[float], index: MultiIndex) -> float:
    """Return a retained raw moment from the HyQMOM 35-vector."""

    try:
        position = _POSITION[index]
    except KeyError as exc:
        raise KeyError(f"moment M{index[0]}{index[1]}{index[2]} is not retained") from exc
    return float(_as_moment_vector(moments)[position])


def multivariate_gaussian_raw_moment(
    index: MultiIndex,
    mean: Sequence[float],
    covariance: Sequence[Sequence[float]],
) -> float:
    """Evaluate a raw moment of a three-dimensional Gaussian.

    The recursion follows Gaussian integration by parts and supports the fifth-
    and sixth-order moments required by the cubic FP projection without an
    explicit velocity grid.
    """

    mean_array = np.asarray(mean, dtype=float)
    covariance_array = np.asarray(covariance, dtype=float)
    if mean_array.shape != (3,) or covariance_array.shape != (3, 3):
        raise ValueError("mean and covariance must have shapes (3,) and (3, 3)")

    @lru_cache(maxsize=None)
    def recurse(alpha: MultiIndex) -> float:
        if alpha == (0, 0, 0):
            return 1.0

        direction = next(d for d, power in enumerate(alpha) if power > 0)
        beta = list(alpha)
        beta[direction] -= 1
        beta_tuple = tuple(beta)
        value = mean_array[direction] * recurse(beta_tuple)

        for other_direction, other_power in enumerate(beta_tuple):
            if other_power == 0:
                continue
            reduced = list(beta_tuple)
            reduced[other_direction] -= 1
            value += (
                other_power
                * covariance_array[direction, other_direction]
                * recurse(tuple(reduced))
            )
        return float(value)

    return recurse(index)


def gaussian_moments_35(
    rho: float,
    mean: Sequence[float],
    covariance: Sequence[Sequence[float]],
) -> np.ndarray:
    """Pack the first 35 raw moments of a Gaussian in HyQMOM order."""

    if rho <= 0.0:
        raise ValueError("rho must be positive")
    return np.asarray(
        [
            rho * multivariate_gaussian_raw_moment(index, mean, covariance)
            for index in HYQMOM_35_INDICES
        ],
        dtype=float,
    )


def maxwellian_moments_35(
    rho: float,
    velocity: Sequence[float],
    theta: float,
) -> np.ndarray:
    """Return a Maxwellian 35-moment vector with variance ``theta``."""

    if theta <= 0.0:
        raise ValueError("theta must be positive")
    return gaussian_moments_35(rho, velocity, theta * np.eye(3))


def mixture_of_gaussians_moments_35(
    components: Iterable[
        tuple[float, Sequence[float], Sequence[Sequence[float]]]
    ],
) -> np.ndarray:
    """Build a non-equilibrium 35-vector from weighted Gaussian components.

    Each component is ``(partial_density, mean, covariance)``.  This is useful
    for reproducible crossing-stream and shock-like homogeneous tests.
    """

    total = np.zeros(35, dtype=float)
    number_of_components = 0
    for partial_density, mean, covariance in components:
        total += gaussian_moments_35(partial_density, mean, covariance)
        number_of_components += 1
    if number_of_components == 0:
        raise ValueError("at least one Gaussian component is required")
    return total


def central_moment(moments: Sequence[float], index: MultiIndex) -> float:
    """Convert a retained raw moment to a central moment."""

    vector = _as_moment_vector(moments)
    rho = vector[_POSITION[(0, 0, 0)]]
    if rho <= 0.0:
        raise ValueError("M000 must be positive")
    mean = np.asarray(
        [
            vector[_POSITION[(1, 0, 0)]],
            vector[_POSITION[(0, 1, 0)]],
            vector[_POSITION[(0, 0, 1)]],
        ]
    ) / rho

    value = 0.0
    for px, py, pz in product(
        range(index[0] + 1), range(index[1] + 1), range(index[2] + 1)
    ):
        raw_index = (px, py, pz)
        coefficient = (
            comb(index[0], px)
            * comb(index[1], py)
            * comb(index[2], pz)
            * (-mean[0]) ** (index[0] - px)
            * (-mean[1]) ** (index[1] - py)
            * (-mean[2]) ** (index[2] - pz)
        )
        value += coefficient * moment_value(vector, raw_index)
    return float(value)


@dataclass(frozen=True)
class MacroscopicState:
    """Hydrodynamic quantities extracted from the retained moments."""

    rho: float
    velocity: np.ndarray
    theta: float
    covariance: np.ndarray
    pressure: float
    stress: np.ndarray
    heat_flux: np.ndarray


def macroscopic_state(moments: Sequence[float]) -> MacroscopicState:
    """Extract state variables using the conventions in FPCode's cubic solver."""

    vector = _as_moment_vector(moments)
    rho = moment_value(vector, (0, 0, 0))
    if rho <= 0.0:
        raise ValueError("M000 must be positive")

    velocity = np.asarray(
        [
            moment_value(vector, (1, 0, 0)),
            moment_value(vector, (0, 1, 0)),
            moment_value(vector, (0, 0, 1)),
        ]
    ) / rho

    covariance_integral = np.asarray(
        [
            [
                central_moment(vector, (2, 0, 0)),
                central_moment(vector, (1, 1, 0)),
                central_moment(vector, (1, 0, 1)),
            ],
            [
                central_moment(vector, (1, 1, 0)),
                central_moment(vector, (0, 2, 0)),
                central_moment(vector, (0, 1, 1)),
            ],
            [
                central_moment(vector, (1, 0, 1)),
                central_moment(vector, (0, 1, 1)),
                central_moment(vector, (0, 0, 2)),
            ],
        ],
        dtype=float,
    )
    covariance = covariance_integral / rho
    theta = float(np.trace(covariance) / 3.0)
    if theta <= 0.0:
        raise ValueError("temperature-like variance theta must be positive")

    pressure = rho * theta
    stress = -(covariance_integral - pressure * np.eye(3))
    heat_flux = 0.5 * np.asarray(
        [
            central_moment(vector, (3, 0, 0))
            + central_moment(vector, (1, 2, 0))
            + central_moment(vector, (1, 0, 2)),
            central_moment(vector, (2, 1, 0))
            + central_moment(vector, (0, 3, 0))
            + central_moment(vector, (0, 1, 2)),
            central_moment(vector, (2, 0, 1))
            + central_moment(vector, (0, 2, 1))
            + central_moment(vector, (0, 0, 3)),
        ],
        dtype=float,
    )

    return MacroscopicState(
        rho=rho,
        velocity=velocity,
        theta=theta,
        covariance=covariance,
        pressure=pressure,
        stress=stress,
        heat_flux=heat_flux,
    )


class GaussianTailClosure:
    """Close unretained fifth/sixth moments with a local Gaussian tail.

    Retained moments through order four are always returned exactly.  Only the
    M5/M6 values required by the cubic FP drift are reconstructed.  Replacing
    this class with a Grad-HyQMOM or quadrature reconstruction is the main
    physics upgrade planned after this first coupling milestone.
    """

    maximum_order = 6

    def __call__(
        self,
        index: MultiIndex,
        moments: Sequence[float],
        state: MacroscopicState | None = None,
    ) -> float:
        if sum(index) <= 4:
            return moment_value(moments, index)
        if sum(index) > self.maximum_order:
            raise ValueError(f"Gaussian tail closure supports up to M{sum(index)}")
        local_state = state if state is not None else macroscopic_state(moments)
        return local_state.rho * multivariate_gaussian_raw_moment(
            index, local_state.velocity, local_state.covariance
        )

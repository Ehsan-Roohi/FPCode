"""Regularized four-delta initial data for extreme nonequilibrium audits.

Rodney Fox suggested a homogeneous two-dimensional velocity distribution made
from four delta functions with unit mass, zero momentum, unit energy, and
nonzero third-order moments.  The cubic-FP implementation is three-dimensional
and its algebraic closures require a positive-definite covariance.  This module
therefore embeds the four centers in the ``vx-vy`` plane and replaces each
delta by the same narrow isotropic Gaussian.  The regularization fraction is
reported explicitly and can be reduced in a refinement study.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .moments import (
    HYQMOM_35_INDICES,
    central_moment,
    mixture_of_gaussians_moments_35,
)


THIRD_ORDER_INDICES = tuple(
    (i, j, 3 - i - j)
    for i in range(4)
    for j in range(4 - i)
)


@dataclass(frozen=True)
class FourDeltaInitialState:
    """Analytic four-center Gaussian regularization and constraint audit."""

    weights: np.ndarray
    centers: np.ndarray
    component_variance: float
    energy_trace: float
    components: tuple[tuple[float, np.ndarray, np.ndarray], ...]
    moments: np.ndarray
    mass_error: float
    momentum_norm: float
    energy_trace_error: float
    central_third_norm: float


def regularized_four_delta_state(
    *,
    energy_trace: float = 1.0,
    regularization_fraction: float = 0.03,
    rotation_degrees: float = 17.0,
) -> FourDeltaInitialState:
    """Return a reproducible skew four-delta state embedded in 3-D velocity.

    ``energy_trace`` means ``sum_i M_2e_i / M000``.  A fraction
    ``regularization_fraction`` of this trace is assigned to the common
    isotropic Gaussian width; the remaining fraction is carried by the four
    centered velocities.  Rotation changes neither the constraints nor the
    amount of nonequilibrium and avoids alignment with a coordinate axis.
    """

    if not np.isfinite(energy_trace) or energy_trace <= 0.0:
        raise ValueError("energy_trace must be finite and positive")
    if (
        not np.isfinite(regularization_fraction)
        or not 0.0 < regularization_fraction < 1.0
    ):
        raise ValueError("regularization_fraction must lie strictly between 0 and 1")
    if not np.isfinite(rotation_degrees):
        raise ValueError("rotation_degrees must be finite")

    weights = np.asarray([0.45, 0.25, 0.20, 0.10], dtype=float)
    raw_xy = np.asarray(
        [
            [-1.15, -0.25],
            [0.10, 1.35],
            [1.55, -0.45],
            [-0.35, -1.70],
        ],
        dtype=float,
    )
    centered_xy = raw_xy - weights @ raw_xy
    angle = np.deg2rad(rotation_degrees)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    centered_xy = centered_xy @ rotation.T
    center_trace = float(np.sum(weights * np.sum(centered_xy**2, axis=1)))
    target_center_trace = (1.0 - regularization_fraction) * energy_trace
    centered_xy *= np.sqrt(target_center_trace / center_trace)
    centers = np.column_stack([centered_xy, np.zeros(weights.size)])

    component_variance = regularization_fraction * energy_trace / 3.0
    covariance = component_variance * np.eye(3)
    components = tuple(
        (float(weight), center.copy(), covariance.copy())
        for weight, center in zip(weights, centers)
    )
    moments = mixture_of_gaussians_moments_35(components)
    position = {
        index: offset for offset, index in enumerate(HYQMOM_35_INDICES)
    }
    momentum = np.asarray(
        [
            moments[position[(1, 0, 0)]],
            moments[position[(0, 1, 0)]],
            moments[position[(0, 0, 1)]],
        ]
    )
    measured_energy_trace = float(
        sum(
            moments[position[index]]
            for index in ((2, 0, 0), (0, 2, 0), (0, 0, 2))
        )
        / moments[position[(0, 0, 0)]]
    )
    central_third = np.asarray(
        [central_moment(moments, index) for index in THIRD_ORDER_INDICES]
    ) / moments[position[(0, 0, 0)]]
    return FourDeltaInitialState(
        weights=weights,
        centers=centers,
        component_variance=float(component_variance),
        energy_trace=float(energy_trace),
        components=components,
        moments=moments,
        mass_error=abs(float(moments[position[(0, 0, 0)]]) - 1.0),
        momentum_norm=float(np.linalg.norm(momentum)),
        energy_trace_error=abs(measured_energy_trace - energy_trace),
        central_third_norm=float(np.linalg.norm(central_third)),
    )


def weighted_raw_moments(
    nodes: Sequence[Sequence[float]],
    weights: Sequence[float],
    indices: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    """Evaluate arbitrary raw moments of a weighted velocity measure."""

    velocities = np.asarray(nodes, dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError("nodes must have shape (n, 3)")
    if probabilities.shape != (velocities.shape[0],):
        raise ValueError("weights must have one entry per node")
    if np.any(probabilities < 0.0) or float(np.sum(probabilities)) <= 0.0:
        raise ValueError("weights must be nonnegative with positive mass")
    return np.asarray(
        [
            np.dot(
                probabilities,
                np.prod(velocities ** np.asarray(index)[None, :], axis=1),
            )
            for index in indices
        ],
        dtype=float,
    )


def all_third_order_indices() -> tuple[tuple[int, int, int], ...]:
    """Return the ten total-degree-three raw-moment indices."""

    return tuple(
        index
        for index in product(range(4), repeat=3)
        if sum(index) == 3
    )

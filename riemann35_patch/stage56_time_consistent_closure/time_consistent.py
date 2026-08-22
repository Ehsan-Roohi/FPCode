"""Time-consistent degree-six moment update used only by Stage 56.

The qualified state contains every raw moment through total degree six:
the repository's 35 moments through degree four and the 49 M5/M6 values.
The linear Ornstein--Uhlenbeck part is mapped exactly for all 84 moments.
The remaining cubic-FP source is advanced with an H3-limited SSPRK2 step,
and the positive two-population tail relaxation is Strang split.

H3 positive semidefiniteness is a necessary truncated-moment realizability
condition.  It is intentionally not described as a sufficient degree-six
realizability proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb, factorial
from typing import Sequence

import numpy as np

from hyqmom_fp import (
    HYQMOM_35_INDICES,
    coefficients_from_moments,
    reconstruct_two_population_quadrature,
)
from hyqmom_fp.collision import CubicFPCoefficients, fp_collision_moment_source
from hyqmom_fp.grad_hyqmom import WeightedNodeTailClosure
from hyqmom_fp.moments import macroscopic_state
from riemann35_patch.stage55_closure_source_audit.run_closure_method import (
    TAIL_INDICES,
)


PACKED_INDICES = tuple(HYQMOM_35_INDICES) + tuple(TAIL_INDICES)
PACKED_POSITION = {index: position for position, index in enumerate(PACKED_INDICES)}
MOMENT_MATRIX_BASIS_3 = tuple(
    index
    for order in range(4)
    for index in product(range(order + 1), repeat=3)
    if sum(index) == order
)
INVARIANT_INDICES = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)
ENERGY_INDICES = ((2, 0, 0), (0, 2, 0), (0, 0, 2))


@dataclass(frozen=True)
class StepDiagnostics:
    """Diagnostics accumulated across one symmetric Stage-56 step."""

    minimum_h3_margin: float
    nonlinear_limiter: float
    projection_limiter: float
    minimum_projection_weight: float
    maximum_projection_moment_residual: float
    maximum_source_quadrature_residual: float
    source_norm: float


class DegreeSixTailClosure:
    """Use transported M5/M6 and positive-node M7/M8 values."""

    def __init__(
        self,
        packed: Sequence[float],
        high_order_nodes: np.ndarray,
        high_order_weights: np.ndarray,
    ) -> None:
        vector = np.asarray(packed, dtype=float)
        if vector.shape != (len(PACKED_INDICES),):
            raise ValueError("degree-six state must contain 84 moments")
        self.values = {
            index: float(vector[position])
            for position, index in enumerate(PACKED_INDICES)
        }
        self.high = WeightedNodeTailClosure(
            high_order_nodes,
            high_order_weights,
            maximum_order=8,
        )

    def __call__(self, index, moments, state=None) -> float:
        del state
        order = sum(index)
        if order <= 6:
            return self.values[index]
        return self.high(index, moments)


def pack_degree_six(moments: Sequence[float], tail: Sequence[float]) -> np.ndarray:
    """Pack the repository 35-vector and all M5/M6 values."""

    low = np.asarray(moments, dtype=float)
    high = np.asarray(tail, dtype=float)
    if low.shape != (35,) or high.shape != (49,):
        raise ValueError("expected 35 retained and 49 tail moments")
    packed = np.concatenate((low, high))
    if not np.all(np.isfinite(packed)):
        raise ValueError("degree-six state contains NaN or infinity")
    return packed


def unpack_degree_six(packed: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(packed, dtype=float)
    if vector.shape != (84,):
        raise ValueError("expected 84 degree-six moments")
    return vector[:35].copy(), vector[35:].copy()


def _multi_indices_below(alpha: tuple[int, int, int]):
    for i in range(alpha[0] + 1):
        for j in range(alpha[1] + 1):
            for k in range(alpha[2] + 1):
                yield (i, j, k)


def _binomial_multi(alpha, beta) -> int:
    return int(np.prod([comb(alpha[d], beta[d]) for d in range(3)]))


def _centered_gaussian_moment(index, variance: float) -> float:
    value = 1.0
    for exponent in index:
        if exponent % 2:
            return 0.0
        if exponent:
            value *= factorial(exponent) / (
                2 ** (exponent // 2) * factorial(exponent // 2)
            ) * variance ** (exponent // 2)
    return float(value)


def exact_ou_degree_six_map(
    packed: Sequence[float], relaxation: float, theta: float | None = None
) -> np.ndarray:
    """Apply the exact isotropic OU map to every raw moment through M6."""

    vector = np.asarray(packed, dtype=float)
    moments, _ = unpack_degree_six(vector)
    state = macroscopic_state(moments)
    local_theta = state.theta if theta is None else float(theta)
    if not 0.0 <= relaxation <= 1.0 or local_theta <= 0.0:
        raise ValueError("invalid OU relaxation or temperature")

    central_before: dict[tuple[int, int, int], float] = {}
    for alpha in PACKED_INDICES:
        value = 0.0
        for beta in _multi_indices_below(alpha):
            mean_factor = np.prod(
                [
                    (-state.velocity[d]) ** (alpha[d] - beta[d])
                    for d in range(3)
                ]
            )
            value += (
                _binomial_multi(alpha, beta)
                * mean_factor
                * vector[PACKED_POSITION[beta]]
            )
        central_before[alpha] = float(value)

    noise_variance = local_theta * (1.0 - relaxation**2)
    central_after: dict[tuple[int, int, int], float] = {}
    for alpha in PACKED_INDICES:
        value = 0.0
        for beta in _multi_indices_below(alpha):
            noise_index = tuple(alpha[d] - beta[d] for d in range(3))
            value += (
                _binomial_multi(alpha, beta)
                * relaxation ** sum(beta)
                * central_before[beta]
                * _centered_gaussian_moment(noise_index, noise_variance)
            )
        central_after[alpha] = float(value)

    mapped = np.empty_like(vector)
    for position, alpha in enumerate(PACKED_INDICES):
        value = 0.0
        for beta in _multi_indices_below(alpha):
            mean_factor = np.prod(
                [
                    state.velocity[d] ** (alpha[d] - beta[d])
                    for d in range(3)
                ]
            )
            value += (
                _binomial_multi(alpha, beta)
                * mean_factor
                * central_after[beta]
            )
        mapped[position] = value
    return mapped


def h3_moment_matrix(packed: Sequence[float]) -> np.ndarray:
    """Return the 20x20 degree-three moment matrix using moments through M6."""

    vector = np.asarray(packed, dtype=float)
    if vector.shape != (84,):
        raise ValueError("expected 84 degree-six moments")
    matrix = np.empty((len(MOMENT_MATRIX_BASIS_3),) * 2)
    for row, left in enumerate(MOMENT_MATRIX_BASIS_3):
        for column, right in enumerate(MOMENT_MATRIX_BASIS_3):
            index = tuple(left[d] + right[d] for d in range(3))
            matrix[row, column] = vector[PACKED_POSITION[index]]
    return 0.5 * (matrix + matrix.T)


def h3_margin(packed: Sequence[float]) -> float:
    """Return the minimum H3 eigenvalue normalized by the mean diagonal."""

    matrix = h3_moment_matrix(packed)
    scale = max(float(np.trace(matrix)) / matrix.shape[0], 1.0e-15)
    return float(np.linalg.eigvalsh(matrix)[0] / scale)


def _limited_h3_update(
    base: np.ndarray,
    candidate: np.ndarray,
    *,
    margin_floor: float,
) -> tuple[np.ndarray, float]:
    if np.all(np.isfinite(candidate)) and h3_margin(candidate) >= margin_floor:
        return candidate, 1.0
    if h3_margin(base) < margin_floor:
        raise FloatingPointError("incoming degree-six state violates the H3 floor")
    lower = 0.0
    upper = 1.0
    increment = candidate - base
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        trial = base + midpoint * increment
        if np.all(np.isfinite(trial)) and h3_margin(trial) >= margin_floor:
            lower = midpoint
        else:
            upper = midpoint
    return base + lower * increment, float(lower)


def _restore_collision_invariants(source: np.ndarray) -> np.ndarray:
    projected = np.asarray(source, dtype=float).copy()
    for index in INVARIANT_INDICES:
        projected[PACKED_POSITION[index]] = 0.0
    leak = sum(projected[PACKED_POSITION[index]] for index in ENERGY_INDICES)
    for index in ENERGY_INDICES:
        projected[PACKED_POSITION[index]] -= leak / 3.0
    return projected


def nonlinear_degree_six_source(
    packed: Sequence[float],
    *,
    tau: float,
    prandtl: float,
    quadrature_nodes: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate the cubic-FP source with its OU generator removed."""

    vector = np.asarray(packed, dtype=float)
    moments, _ = unpack_degree_six(vector)
    state = macroscopic_state(moments)
    quadrature = reconstruct_two_population_quadrature(
        moments,
        quadrature_nodes=quadrature_nodes,
        minimum_skewness_norm=0.05,
        residual_correction=False,
    )
    closure = DegreeSixTailClosure(vector, quadrature.nodes, quadrature.weights)
    coefficients = coefficients_from_moments(
        moments,
        tau=tau,
        prandtl=prandtl,
        closure=closure,
    )
    ou_coefficients = CubicFPCoefficients.ornstein_uhlenbeck(
        tau=tau,
        theta=state.theta,
    )
    full = np.asarray(
        [
            fp_collision_moment_source(
                index,
                moments,
                coefficients,
                closure=closure,
                state=state,
            )
            for index in PACKED_INDICES
        ]
    )
    ou = np.asarray(
        [
            fp_collision_moment_source(
                index,
                moments,
                ou_coefficients,
                closure=closure,
                state=state,
            )
            for index in PACKED_INDICES
        ]
    )
    source = _restore_collision_invariants(full - ou)
    return source, {
        "minimum_weight": float(np.min(quadrature.weights)),
        "moment_residual": float(quadrature.base_relative_moment_residual),
        "source_norm": float(np.linalg.norm(source)),
    }


def positive_projection_target(
    packed: Sequence[float], *, quadrature_nodes: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a positive-node M5/M6 target using the requested node count."""

    vector = np.asarray(packed, dtype=float)
    moments, _ = unpack_degree_six(vector)
    quadrature = reconstruct_two_population_quadrature(
        moments,
        quadrature_nodes=quadrature_nodes,
        minimum_skewness_norm=0.05,
        residual_correction=False,
    )
    closure = WeightedNodeTailClosure(
        quadrature.nodes,
        quadrature.weights,
        maximum_order=6,
    )
    tail = np.asarray([closure(index, moments) for index in TAIL_INDICES])
    return tail, {
        "minimum_weight": float(np.min(quadrature.weights)),
        "moment_residual": float(quadrature.base_relative_moment_residual),
        "node_count": int(quadrature.weights.size),
        "quadrature_nodes": int(quadrature_nodes),
    }


def _projection_relaxation(
    packed: np.ndarray,
    *,
    duration: float,
    relaxation_time: float,
    quadrature_nodes: int,
    margin_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    target, diagnostics = positive_projection_target(
        packed,
        quadrature_nodes=quadrature_nodes,
    )
    retention = float(np.exp(-duration / relaxation_time))
    candidate = packed.copy()
    candidate[35:] = target + retention * (packed[35:] - target)
    limited, limiter = _limited_h3_update(
        packed,
        candidate,
        margin_floor=margin_floor,
    )
    return limited, {
        **diagnostics,
        "limiter": limiter,
        "h3_margin": h3_margin(limited),
    }


def _ssprk2_nonlinear_step(
    packed: np.ndarray,
    *,
    dt: float,
    tau: float,
    prandtl: float,
    quadrature_nodes: int,
    margin_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    first, first_diagnostics = nonlinear_degree_six_source(
        packed,
        tau=tau,
        prandtl=prandtl,
        quadrature_nodes=quadrature_nodes,
    )
    euler, limiter_euler = _limited_h3_update(
        packed,
        packed + dt * first,
        margin_floor=margin_floor,
    )
    second, second_diagnostics = nonlinear_degree_six_source(
        euler,
        tau=tau,
        prandtl=prandtl,
        quadrature_nodes=quadrature_nodes,
    )
    candidate = packed + 0.5 * dt * (first + second)
    updated, limiter_final = _limited_h3_update(
        packed,
        candidate,
        margin_floor=margin_floor,
    )
    return updated, {
        "limiter": min(limiter_euler, limiter_final),
        "minimum_weight": min(
            first_diagnostics["minimum_weight"],
            second_diagnostics["minimum_weight"],
        ),
        "maximum_moment_residual": max(
            first_diagnostics["moment_residual"],
            second_diagnostics["moment_residual"],
        ),
        "source_norm": max(
            first_diagnostics["source_norm"],
            second_diagnostics["source_norm"],
        ),
        "h3_margin": h3_margin(updated),
    }


def time_consistent_degree_six_step(
    packed: Sequence[float],
    *,
    dt: float,
    tau: float,
    prandtl: float = 2.0 / 3.0,
    tail_relaxation_time: float = 0.01,
    quadrature_nodes: int = 5,
    h3_floor: float = -1.0e-12,
) -> tuple[np.ndarray, StepDiagnostics]:
    """Advance one P/OU/N/OU/P Strang step."""

    if min(dt, tau, tail_relaxation_time) <= 0.0:
        raise ValueError("all time scales must be positive")
    incoming = np.asarray(packed, dtype=float)
    if incoming.shape != (84,):
        raise ValueError("expected 84 degree-six moments")
    margins = [h3_margin(incoming)]

    state, first_projection = _projection_relaxation(
        incoming,
        duration=0.5 * dt,
        relaxation_time=tail_relaxation_time,
        quadrature_nodes=quadrature_nodes,
        margin_floor=h3_floor,
    )
    margins.append(first_projection["h3_margin"])
    theta = macroscopic_state(state[:35]).theta
    state = exact_ou_degree_six_map(
        state,
        relaxation=float(np.exp(-0.5 * dt / tau)),
        theta=theta,
    )
    margins.append(h3_margin(state))
    state, nonlinear = _ssprk2_nonlinear_step(
        state,
        dt=dt,
        tau=tau,
        prandtl=prandtl,
        quadrature_nodes=quadrature_nodes,
        margin_floor=h3_floor,
    )
    margins.append(nonlinear["h3_margin"])
    theta = macroscopic_state(state[:35]).theta
    state = exact_ou_degree_six_map(
        state,
        relaxation=float(np.exp(-0.5 * dt / tau)),
        theta=theta,
    )
    margins.append(h3_margin(state))
    state, second_projection = _projection_relaxation(
        state,
        duration=0.5 * dt,
        relaxation_time=tail_relaxation_time,
        quadrature_nodes=quadrature_nodes,
        margin_floor=h3_floor,
    )
    margins.append(second_projection["h3_margin"])
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("Stage-56 step produced NaN or infinity")
    return state, StepDiagnostics(
        minimum_h3_margin=float(min(margins)),
        nonlinear_limiter=float(nonlinear["limiter"]),
        projection_limiter=float(
            min(first_projection["limiter"], second_projection["limiter"])
        ),
        minimum_projection_weight=float(
            min(
                first_projection["minimum_weight"],
                second_projection["minimum_weight"],
            )
        ),
        maximum_projection_moment_residual=float(
            max(
                first_projection["moment_residual"],
                second_projection["moment_residual"],
            )
        ),
        maximum_source_quadrature_residual=float(
            nonlinear["maximum_moment_residual"]
        ),
        source_norm=float(nonlinear["source_norm"]),
    )

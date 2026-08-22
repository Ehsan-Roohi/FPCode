"""Time integrators for the positive projected M5/M6 memory.

Stage 55 applied the algebraic tail projection after the dynamic source step.
That first-order Lie split changed by more than the qualification tolerance
when the step was halved.  This module keeps the same 35+49 state and positive
two-population target, but applies the stiff tail relaxation analytically in a
symmetric Strang split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hyqmom_fp import (
    DYNAMIC_TAIL_INDICES,
    DynamicHighOrderState,
    dynamic_high_order_fp_step,
    reconstruct_two_population_quadrature,
)
from hyqmom_fp.grad_hyqmom import WeightedNodeTailClosure


@dataclass(frozen=True)
class TailProjectionDiagnostics:
    """Diagnostics for one exact relaxation toward a positive tail target."""

    retention: float
    minimum_weight: float
    negative_mass_fraction: float
    target_relative_residual: float
    relative_distance_before: float
    relative_distance_after: float


@dataclass(frozen=True)
class ProjectedStepDiagnostics:
    """Combined source-step and tail-projection diagnostics."""

    limiter_fraction: float
    realizability_margin: float
    minimum_weight: float
    maximum_negative_mass_fraction: float
    maximum_target_relative_residual: float
    maximum_projection_distance: float
    tail_increment_norm: float


def positive_tail_target(
    moments: np.ndarray,
    *,
    maximum_order: int = 6,
    quadrature_nodes: int = 5,
    minimum_skewness_norm: float = 0.05,
) -> tuple[np.ndarray, float, float, float]:
    """Return the positive two-population M5/M6 target and diagnostics."""

    if maximum_order != 6:
        raise ValueError("Stage 56 is qualified only for maximum_order=6")
    quadrature = reconstruct_two_population_quadrature(
        moments,
        quadrature_nodes=quadrature_nodes,
        minimum_skewness_norm=minimum_skewness_norm,
        residual_correction=False,
    )
    closure = WeightedNodeTailClosure(
        quadrature.nodes,
        quadrature.weights,
        maximum_order=maximum_order,
    )
    target = np.asarray(
        [closure(index, moments) for index in DYNAMIC_TAIL_INDICES],
        dtype=float,
    )
    return (
        target,
        float(np.min(quadrature.weights)),
        float(quadrature.negative_mass_fraction),
        float(quadrature.base_relative_moment_residual),
    )


def exact_tail_relaxation(
    state: DynamicHighOrderState,
    duration: float,
    relaxation_time: float,
    *,
    quadrature_nodes: int = 5,
    minimum_skewness_norm: float = 0.05,
) -> tuple[DynamicHighOrderState, TailProjectionDiagnostics]:
    """Integrate ``tail'=(target-tail)/relaxation_time`` exactly.

    The 35 moments are held fixed over this substep.  Consequently, repeated
    substeps satisfy the exponential semigroup to round-off.
    """

    if duration < 0.0 or relaxation_time <= 0.0:
        raise ValueError("duration must be nonnegative and relaxation_time positive")
    target, minimum_weight, negative_mass, residual = positive_tail_target(
        np.asarray(state.moments, dtype=float),
        maximum_order=state.maximum_order,
        quadrature_nodes=quadrature_nodes,
        minimum_skewness_norm=minimum_skewness_norm,
    )
    tail = np.asarray(state.tail_moments, dtype=float)
    scale = max(float(np.linalg.norm(target)), 1.0e-14)
    before = float(np.linalg.norm(tail - target) / scale)
    retention = float(np.exp(-duration / relaxation_time))
    updated_tail = target + retention * (tail - target)
    after = float(np.linalg.norm(updated_tail - target) / scale)
    return (
        DynamicHighOrderState(
            moments=np.asarray(state.moments, dtype=float).copy(),
            tail_moments=updated_tail,
            maximum_order=state.maximum_order,
        ),
        TailProjectionDiagnostics(
            retention=retention,
            minimum_weight=minimum_weight,
            negative_mass_fraction=negative_mass,
            target_relative_residual=residual,
            relative_distance_before=before,
            relative_distance_after=after,
        ),
    )


def _combined_diagnostics(dynamic, projections) -> ProjectedStepDiagnostics:
    return ProjectedStepDiagnostics(
        limiter_fraction=float(dynamic.limiter_fraction),
        realizability_margin=float(dynamic.realizability_margin),
        minimum_weight=float(min(item.minimum_weight for item in projections)),
        maximum_negative_mass_fraction=float(
            max(item.negative_mass_fraction for item in projections)
        ),
        maximum_target_relative_residual=float(
            max(item.target_relative_residual for item in projections)
        ),
        maximum_projection_distance=float(
            max(item.relative_distance_before for item in projections)
        ),
        tail_increment_norm=float(dynamic.tail_increment_norm),
    )


def legacy_lie_projected_step(
    state: DynamicHighOrderState,
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
    tail_relaxation_time: float = 1.0e-2,
    quadrature_nodes: int = 5,
) -> tuple[DynamicHighOrderState, ProjectedStepDiagnostics]:
    """Use the Stage-55-style post-step Lie projection as a control."""

    dynamic_state, dynamic = dynamic_high_order_fp_step(
        state,
        dt,
        tau,
        prandtl=prandtl,
        minimum_skewness_norm=0.05,
        high_order_quadrature_nodes=quadrature_nodes,
    )
    projected, projection = exact_tail_relaxation(
        dynamic_state,
        dt,
        tail_relaxation_time,
        quadrature_nodes=quadrature_nodes,
    )
    return projected, _combined_diagnostics(dynamic, (projection,))


def strang_exact_projected_step(
    state: DynamicHighOrderState,
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
    tail_relaxation_time: float = 1.0e-2,
    quadrature_nodes: int = 5,
) -> tuple[DynamicHighOrderState, ProjectedStepDiagnostics]:
    """Advance with exact half relaxation, dynamic source, exact half relaxation."""

    half_state, first = exact_tail_relaxation(
        state,
        0.5 * dt,
        tail_relaxation_time,
        quadrature_nodes=quadrature_nodes,
    )
    dynamic_state, dynamic = dynamic_high_order_fp_step(
        half_state,
        dt,
        tau,
        prandtl=prandtl,
        minimum_skewness_norm=0.05,
        high_order_quadrature_nodes=quadrature_nodes,
    )
    projected, second = exact_tail_relaxation(
        dynamic_state,
        0.5 * dt,
        tail_relaxation_time,
        quadrature_nodes=quadrature_nodes,
    )
    return projected, _combined_diagnostics(dynamic, (first, second))

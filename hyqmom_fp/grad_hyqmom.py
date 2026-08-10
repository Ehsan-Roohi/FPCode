"""Grad--HyQMOM / Gaussian--GQMOM source closure for the 35-moment state.

This module implements Appendix C of Bryngelson, Fox & Laurent (JCP 566,
2026, 115242) for the fourth-order (``n=2``) 3-D moment set.  The retained
moments are first standardized componentwise.  Gaussian--GQMOM supplies a
strictly realizable univariate quadrature in each velocity direction, while
the Grad--HyQMOM polynomial correction restores every cross moment through
total degree four.  The resulting (generally signed) tensor quadrature then
defines the fifth- and sixth-order moments required by the cubic FP source.

The implementation intentionally keeps the signed-weight diagnostic.  The
Grad approximation is a source quadrature, not a positive VDF
reconstruction; negative weights therefore do not invalidate its moment
closure, but they are an important robustness diagnostic near the boundary of
moment space.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .moments import (
    HYQMOM_35_INDICES,
    MacroscopicState,
    MultiIndex,
    central_moment,
    macroscopic_state,
    moment_value,
)


POSITION = {index: position for position, index in enumerate(HYQMOM_35_INDICES)}


@dataclass(frozen=True)
class GaussianGQMOMMarginal:
    """Univariate Gaussian--GQMOM quadrature and recurrence data."""

    nodes: np.ndarray
    weights: np.ndarray
    recurrence_a: np.ndarray
    recurrence_b: np.ndarray
    hankel_margin: float


@dataclass(frozen=True)
class GradHyQMOMQuadrature:
    """Signed tensor quadrature representing the Appendix-C approximation."""

    weights: np.ndarray
    nodes: np.ndarray
    reconstructed_moments: np.ndarray
    relative_moment_residual: float
    negative_mass_fraction: float
    minimum_hankel_margin: float
    marginal_quadratures: tuple[
        GaussianGQMOMMarginal, GaussianGQMOMMarginal, GaussianGQMOMMarginal
    ]


@dataclass(frozen=True)
class GradHyQMOMStepDiagnostics:
    """Diagnostics for the OU-split realizability-limited collision step."""

    limiter_fraction: float
    realizability_margin: float
    negative_mass_fraction: float
    minimum_hankel_margin: float
    source_norm: float
    nonlinear_source_norm: float


class WeightedNodeTailClosure:
    """Return exact retained moments and node-quadrature M5/M6 values."""

    def __init__(
        self,
        nodes: Sequence[Sequence[float]],
        weights: Sequence[float],
        *,
        maximum_order: int = 6,
    ) -> None:
        self.nodes = np.asarray(nodes, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.maximum_order = int(maximum_order)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must have shape (n, 3)")
        if self.weights.shape != (self.nodes.shape[0],):
            raise ValueError("weights must have shape (n,)")
        if not np.all(np.isfinite(self.nodes)) or not np.all(
            np.isfinite(self.weights)
        ):
            raise ValueError("nodes and weights must be finite")

    def __call__(
        self,
        index: MultiIndex,
        moments: Sequence[float],
        state: MacroscopicState | None = None,
    ) -> float:
        del state
        order = sum(index)
        if order <= 4:
            return moment_value(moments, index)
        if order > self.maximum_order:
            raise ValueError(f"weighted-node closure supports up to M{order}")
        values = np.prod(self.nodes ** np.asarray(index)[None, :], axis=1)
        return float(np.dot(self.weights, values))


def gaussian_gqmom_marginal(
    third_standardized: float,
    fourth_standardized: float,
    *,
    quadrature_nodes: int = 6,
    boundary_tolerance: float = 1.0e-12,
) -> GaussianGQMOMMarginal:
    """Construct the ``n=2`` Gaussian--GQMOM standardized quadrature.

    With ``s0=1``, ``s1=0`` and ``s2=1``, the known recurrence coefficients
    are ``a0=0``, ``a1=s3``, ``b1=1`` and
    ``b2=s4-s3^2-1``.  HyQMOM supplies ``a2=(a0+a1)/2``.  The Gaussian--GQMOM
    continuation uses ``a_k=a2`` and ``b_k=(k/2)b2`` for ``k>2``; for a
    Maxwellian this reduces exactly to the Hermite recurrence.
    """

    if quadrature_nodes < 3:
        raise ValueError("Gaussian-GQMOM requires at least three nodes for n=2")
    if not np.all(np.isfinite([third_standardized, fourth_standardized])):
        raise ValueError("standardized moments must be finite")
    hankel = float(fourth_standardized - third_standardized**2 - 1.0)
    if hankel <= boundary_tolerance:
        raise ValueError(
            "univariate fourth-order moment is on or too near the Hankel boundary: "
            f"b2={hankel:.6e}"
        )

    a = np.empty(quadrature_nodes)
    a[0] = 0.0
    a[1] = float(third_standardized)
    a[2:] = 0.5 * float(third_standardized)
    b = np.zeros(quadrature_nodes)
    b[1] = 1.0
    for order in range(2, quadrature_nodes):
        b[order] = (order / 2.0) * hankel

    jacobi = np.diag(a)
    off_diagonal = np.sqrt(b[1:])
    jacobi += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    nodes, eigenvectors = np.linalg.eigh(jacobi)
    weights = eigenvectors[0, :] ** 2
    weights /= np.sum(weights)
    return GaussianGQMOMMarginal(
        nodes=nodes,
        weights=weights,
        recurrence_a=a,
        recurrence_b=b,
        hankel_margin=hankel,
    )


def _orthogonal_polynomials(
    marginal: GaussianGQMOMMarginal, maximum_degree: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return monomial coefficients for Q_0...Q_degree and their norms."""

    polynomials = [np.asarray([1.0])]
    norms = np.ones(maximum_degree + 1)
    if maximum_degree == 0:
        return polynomials, norms
    previous = np.asarray([0.0])
    current = polynomials[0]
    for degree in range(maximum_degree):
        shifted = np.pad(current, (1, 0))
        same = np.pad(current, (0, 1))
        prior = np.pad(previous, (0, shifted.size - previous.size))
        following = (
            shifted
            - marginal.recurrence_a[degree] * same
            - marginal.recurrence_b[degree] * prior
        )
        polynomials.append(following)
        previous, current = current, following
        if degree + 1 <= maximum_degree:
            norms[degree + 1] = norms[degree] * marginal.recurrence_b[degree + 1]
    return polynomials, norms


def _standardized_moments_through_four(
    moments: Sequence[float], state: MacroscopicState
) -> dict[MultiIndex, float]:
    scales = np.sqrt(np.diag(state.covariance))
    if np.min(scales) <= 0.0:
        raise ValueError("all coordinate variances must be positive")
    standardized: dict[MultiIndex, float] = {}
    for index in HYQMOM_35_INDICES:
        denominator = state.rho * np.prod(scales ** np.asarray(index))
        standardized[index] = float(central_moment(moments, index) / denominator)
    return standardized


def _polynomial_cross_expectation(
    left: np.ndarray,
    middle: np.ndarray,
    right: np.ndarray,
    standardized: dict[MultiIndex, float],
) -> float:
    value = 0.0
    for i, coefficient_i in enumerate(left):
        for j, coefficient_j in enumerate(middle):
            for k, coefficient_k in enumerate(right):
                value += (
                    coefficient_i
                    * coefficient_j
                    * coefficient_k
                    * standardized[(i, j, k)]
                )
    return float(value)


def _moments_from_weighted_nodes(weights: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.dot(
                weights,
                nodes[:, 0] ** i * nodes[:, 1] ** j * nodes[:, 2] ** k,
            )
            for i, j, k in HYQMOM_35_INDICES
        ]
    )


def reconstruct_grad_hyqmom_quadrature(
    moments: Sequence[float],
    *,
    quadrature_nodes: int = 6,
    boundary_tolerance: float = 1.0e-12,
) -> GradHyQMOMQuadrature:
    """Construct the Appendix-C Grad--HyQMOM source quadrature.

    Six nodes per direction integrate the product of the fourth-degree Grad
    correction and every monomial through degree six exactly for the implicit
    Gaussian--GQMOM measure (maximum one-dimensional degree ten).
    """

    vector = np.asarray(moments, dtype=float)
    if vector.shape != (35,) or not np.all(np.isfinite(vector)):
        raise ValueError("expected a finite 35-moment vector")
    if quadrature_nodes < 6:
        raise ValueError("at least six nodes per direction are required for M6")
    state = macroscopic_state(vector)
    coordinate_scales = np.sqrt(np.diag(state.covariance))
    standardized = _standardized_moments_through_four(vector, state)

    marginals = tuple(
        gaussian_gqmom_marginal(
            standardized[tuple(3 if direction == axis else 0 for direction in range(3))],
            standardized[tuple(4 if direction == axis else 0 for direction in range(3))],
            quadrature_nodes=quadrature_nodes,
            boundary_tolerance=boundary_tolerance,
        )
        for axis in range(3)
    )
    polynomial_data = tuple(_orthogonal_polynomials(item, 4) for item in marginals)

    kappas: dict[MultiIndex, float] = {}
    for i in range(5):
        for j in range(5 - i):
            for k in range(5 - i - j):
                numerator = _polynomial_cross_expectation(
                    polynomial_data[0][0][i],
                    polynomial_data[1][0][j],
                    polynomial_data[2][0][k],
                    standardized,
                )
                denominator = (
                    polynomial_data[0][1][i]
                    * polynomial_data[1][1][j]
                    * polynomial_data[2][1][k]
                )
                if denominator <= 0.0:
                    raise ValueError("orthogonal-polynomial norm is nonpositive")
                kappas[(i, j, k)] = numerator / denominator

    grid_indices = np.asarray(
        list(product(range(quadrature_nodes), repeat=3)), dtype=int
    )
    standardized_nodes = np.column_stack(
        [marginals[axis].nodes[grid_indices[:, axis]] for axis in range(3)]
    )
    base_weights = np.prod(
        np.column_stack(
            [marginals[axis].weights[grid_indices[:, axis]] for axis in range(3)]
        ),
        axis=1,
    )
    polynomial_values = tuple(
        np.asarray(
            [
                np.polynomial.polynomial.polyval(
                    marginals[axis].nodes, polynomial_data[axis][0][degree]
                )
                for degree in range(5)
            ]
        )
        for axis in range(3)
    )
    correction = np.zeros(grid_indices.shape[0])
    for (i, j, k), coefficient in kappas.items():
        correction += coefficient * (
            polynomial_values[0][i, grid_indices[:, 0]]
            * polynomial_values[1][j, grid_indices[:, 1]]
            * polynomial_values[2][k, grid_indices[:, 2]]
        )

    weights = state.rho * base_weights * correction
    nodes = state.velocity + standardized_nodes * coordinate_scales
    reconstructed = _moments_from_weighted_nodes(weights, nodes)
    residual = float(
        np.linalg.norm(reconstructed - vector) / max(np.linalg.norm(vector), 1.0e-15)
    )
    negative_mass = float(-np.sum(np.minimum(weights, 0.0)) / state.rho)
    return GradHyQMOMQuadrature(
        weights=weights,
        nodes=nodes,
        reconstructed_moments=reconstructed,
        relative_moment_residual=residual,
        negative_mass_fraction=negative_mass,
        minimum_hankel_margin=float(min(item.hankel_margin for item in marginals)),
        marginal_quadratures=marginals,
    )


def grad_hyqmom_fp_step(
    moments: Sequence[float],
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
    margin_floor: float = 1.0e-12,
    quadrature_nodes: int = 6,
) -> tuple[np.ndarray, GradHyQMOMStepDiagnostics]:
    """Advance an OU-split Grad--HyQMOM source with a realizability guard.

    The linear Ornstein--Uhlenbeck part is propagated exactly.  The remaining
    cubic-FP increment is evaluated with the Appendix-C M5/M6 closure and
    multiplied by the largest ``0 <= lambda <= 1`` for which the complete
    10-by-10 moment matrix generated by all monomials through degree two stays
    inside the fourth-order realizability cone.  This is a collision-side
    operator-splitting guard for coupling to HyQMOM transport; ``lambda=1``
    means the limiter is inactive.  No personal-communication attribution is
    implied by this implementation.
    """

    if dt <= 0.0 or tau <= 0.0:
        raise ValueError("dt and tau must be positive")
    if not 0.0 < prandtl <= 1.0 or margin_floor < 0.0:
        raise ValueError("invalid Prandtl number or margin floor")
    vector = np.asarray(moments, dtype=float)
    state = macroscopic_state(vector)
    quadrature = reconstruct_grad_hyqmom_quadrature(
        vector, quadrature_nodes=quadrature_nodes
    )
    closure = WeightedNodeTailClosure(quadrature.nodes, quadrature.weights)

    # Local imports avoid a module-level cycle: mixture_closure imports the
    # physical coefficient solve, while this source step needs its OU map and
    # canonical moment-matrix predicate.
    from .collision import (
        CubicFPCoefficients,
        coefficients_from_moments,
        projected_fp_collision_source,
    )
    from .mixture_closure import exact_ou_moment_map, realizability_margin_35

    coefficients = coefficients_from_moments(
        vector, tau=tau, prandtl=prandtl, closure=closure
    )
    source = projected_fp_collision_source(vector, coefficients, closure=closure)
    ou_coefficients = CubicFPCoefficients.ornstein_uhlenbeck(
        tau=tau, theta=state.theta
    )
    ou_source = projected_fp_collision_source(vector, ou_coefficients)
    nonlinear_source = source - ou_source
    relaxation = float(np.exp(-dt / tau))
    ou_mapped = exact_ou_moment_map(vector, relaxation, state.theta)

    def candidate(limiter: float) -> np.ndarray:
        return ou_mapped + limiter * dt * nonlinear_source

    limiter = 1.0
    updated = candidate(limiter)
    margin = realizability_margin_35(updated)
    if not np.all(np.isfinite(updated)) or margin < margin_floor:
        lower = 0.0
        upper = 1.0
        base_margin = realizability_margin_35(ou_mapped)
        if base_margin < margin_floor:
            raise FloatingPointError(
                "exact OU base state did not enter the realizability interior"
            )
        for _ in range(60):
            midpoint = 0.5 * (lower + upper)
            trial = candidate(midpoint)
            trial_margin = realizability_margin_35(trial)
            if np.all(np.isfinite(trial)) and trial_margin >= margin_floor:
                lower = midpoint
            else:
                upper = midpoint
        limiter = lower
        updated = candidate(limiter)
        margin = realizability_margin_35(updated)

    # Pin collision invariants after the split update.  The corrections are
    # roundoff-size because both the exact OU map and projected nonlinear
    # source conserve them analytically.
    for index in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        updated[POSITION[index]] = vector[POSITION[index]]
    energy_indices = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    energy_error = sum(vector[POSITION[index]] for index in energy_indices) - sum(
        updated[POSITION[index]] for index in energy_indices
    )
    for index in energy_indices:
        updated[POSITION[index]] += energy_error / 3.0
    margin = realizability_margin_35(updated)
    if margin < -5.0e-13 or not np.all(np.isfinite(updated)):
        raise FloatingPointError("limited Grad-HyQMOM step is not realizable")
    return updated, GradHyQMOMStepDiagnostics(
        limiter_fraction=float(limiter),
        realizability_margin=float(margin),
        negative_mass_fraction=quadrature.negative_mass_fraction,
        minimum_hankel_margin=quadrature.minimum_hankel_margin,
        source_norm=float(np.linalg.norm(source)),
        nonlinear_source_norm=float(np.linalg.norm(nonlinear_source)),
    )

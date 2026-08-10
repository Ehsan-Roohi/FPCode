"""High-skew two-population source closure for cubic Fokker--Planck moments.

The retained HyQMOM state contains all moments through degree four, whereas
the cubic drift requires degree-five and degree-six moments.  This module
targets the failure mode isolated in Stage 11: a light, energetic beam riding
on a dominant background population.

The state is first whitened.  The dominant direction of the standardized
third-moment tensor defines a rank-one two-Gaussian population split.  The
one-dimensional Pearson fit is exact for a two-population equal-covariance
beam.  A minimum-weight residual correction on a positive Gauss--Hermite
quadrature then restores every retained moment through degree four before the
quadrature is used to evaluate M5/M6.  The collision step retains the exact
Ornstein--Uhlenbeck map and the full realizability guard used by Stage 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .grad_hyqmom import WeightedNodeTailClosure
from .mixture_closure import (
    exact_ou_moment_map,
    fit_equal_variance_marginal,
    fit_location_scale_marginal,
    realizability_margin_35,
)
from .moments import (
    HYQMOM_35_INDICES,
    MacroscopicState,
    central_moment,
    macroscopic_state,
    moment_value,
)


POSITION = {index: position for position, index in enumerate(HYQMOM_35_INDICES)}
DYNAMIC_TAIL_INDICES = tuple(
    (i, j, order - i - j)
    for order in (5, 6)
    for i in range(order + 1)
    for j in range(order - i + 1)
)


def _dynamic_indices(maximum_order: int) -> tuple[tuple[int, int, int], ...]:
    if maximum_order < 6:
        raise ValueError("dynamic hierarchy must include moments through degree six")
    return tuple(
        (i, j, order - i - j)
        for order in range(5, maximum_order + 1)
        for i in range(order + 1)
        for j in range(order - i + 1)
    )


@dataclass(frozen=True)
class TwoPopulationQuadrature:
    """Residual-corrected quadrature for the high-skew source tail."""

    weights: np.ndarray
    nodes: np.ndarray
    reconstructed_moments: np.ndarray
    component_weights: np.ndarray
    component_means: np.ndarray
    component_covariances: np.ndarray
    dominant_direction_standardized: np.ndarray
    standardized_skewness: float
    skewness_tensor_norm: float
    marginal_branch: str
    base_relative_moment_residual: float
    corrected_relative_moment_residual: float
    residual_correction_norm: float
    negative_mass_fraction: float


@dataclass(frozen=True)
class TwoPopulationStepDiagnostics:
    """Diagnostics for one guarded high-skew collision step."""

    limiter_fraction: float
    realizability_margin: float
    skewness_tensor_norm: float
    standardized_skewness: float
    marginal_branch: str
    base_relative_moment_residual: float
    corrected_relative_moment_residual: float
    residual_correction_norm: float
    negative_mass_fraction: float
    source_norm: float
    nonlinear_source_norm: float


@dataclass(frozen=True)
class PersistentTwoPopulationState:
    """Two Gaussian populations carried between collision intervals."""

    rho: float
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray


@dataclass(frozen=True)
class PersistentTwoPopulationDiagnostics:
    """Diagnostics for an assumed-density two-population finite map."""

    alpha: float
    affine_scale: float
    realizability_margin: float
    maximum_c2_over_theta: float


@dataclass(frozen=True)
class DynamicHighOrderState:
    """The 35 transported moments plus source-local dynamic M5/M6 memory."""

    moments: np.ndarray
    tail_moments: np.ndarray
    maximum_order: int = 6


@dataclass(frozen=True)
class DynamicHighOrderDiagnostics:
    """Diagnostics for the source-local 35+49 moment update."""

    limiter_fraction: float
    realizability_margin: float
    tail_increment_norm: float
    algebraic_tail_relative_difference: float
    source_norm: float


class _DynamicTailClosure:
    def __init__(
        self,
        tail_moments: np.ndarray,
        high_order_quadrature: TwoPopulationQuadrature,
        dynamic_indices: tuple[tuple[int, int, int], ...],
        maximum_order: int,
    ) -> None:
        self.tail = {
            index: float(value)
            for index, value in zip(dynamic_indices, tail_moments)
        }
        self.maximum_order = int(maximum_order)
        self.high = WeightedNodeTailClosure(
            high_order_quadrature.nodes,
            high_order_quadrature.weights,
            maximum_order=maximum_order + 2,
        )

    def __call__(
        self,
        index: tuple[int, int, int],
        moments: Sequence[float],
        state: MacroscopicState | None = None,
    ) -> float:
        order = sum(index)
        if order <= 4:
            return moment_value(moments, index)
        if order <= self.maximum_order:
            return self.tail[index]
        return self.high(index, moments, state)


def _central_tensor(moments: np.ndarray, order: int) -> np.ndarray:
    state = macroscopic_state(moments)
    tensor = np.empty((3,) * order, dtype=float)
    for directions in product(range(3), repeat=order):
        index = [0, 0, 0]
        for direction in directions:
            index[direction] += 1
        tensor[directions] = central_moment(moments, tuple(index)) / state.rho
    return tensor


def _whitened_statistics(
    moments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = macroscopic_state(moments)
    eigenvalues, eigenvectors = np.linalg.eigh(state.covariance)
    scale = max(float(np.max(eigenvalues)), state.theta, 1.0)
    if np.min(eigenvalues) <= 1.0e-13 * scale:
        raise ValueError("two-population whitening covariance is near singular")
    square_root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_square_root = (
        eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    )
    third = _central_tensor(moments, 3)
    fourth = _central_tensor(moments, 4)
    third_standardized = np.einsum(
        "ia,jb,kc,abc->ijk",
        inverse_square_root,
        inverse_square_root,
        inverse_square_root,
        third,
        optimize=True,
    )
    fourth_standardized = np.einsum(
        "ia,jb,kc,ld,abcd->ijkl",
        inverse_square_root,
        inverse_square_root,
        inverse_square_root,
        inverse_square_root,
        fourth,
        optimize=True,
    )
    return square_root, inverse_square_root, third_standardized, fourth_standardized


def _dominant_skew_direction(third: np.ndarray) -> tuple[np.ndarray, float]:
    unfolding = third.reshape(3, 9)
    left_vectors = np.linalg.svd(unfolding, full_matrices=False)[0]
    starts = [np.eye(3)[axis] for axis in range(3)]
    starts.extend(left_vectors[:, axis] for axis in range(3))
    starts.extend(
        (
            np.asarray([1.0, 1.0, 1.0]),
            np.asarray([1.0, -1.0, 0.5]),
            np.asarray([0.5, 1.0, -1.0]),
        )
    )
    best_direction: np.ndarray | None = None
    best_value = 0.0
    for start in starts:
        direction = np.asarray(start, dtype=float)
        direction /= np.linalg.norm(direction)
        for _ in range(100):
            gradient = np.einsum(
                "ijk,j,k->i", third, direction, direction, optimize=True
            )
            norm = float(np.linalg.norm(gradient))
            if norm <= 1.0e-15:
                break
            candidate = gradient / norm
            if abs(float(np.dot(candidate, direction))) >= 1.0 - 1.0e-14:
                direction = candidate
                break
            direction = candidate
        value = float(
            np.einsum(
                "ijk,i,j,k", third, direction, direction, direction, optimize=True
            )
        )
        if abs(value) > abs(best_value):
            best_direction = direction.copy()
            best_value = value
    if best_direction is None or abs(best_value) <= 1.0e-14:
        raise ValueError("standardized third-moment tensor has no resolved direction")
    if best_value < 0.0:
        best_direction *= -1.0
        best_value *= -1.0
    return best_direction, best_value


def _moments_from_weighted_nodes(weights: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.dot(weights, nodes[:, 0] ** i * nodes[:, 1] ** j * nodes[:, 2] ** k)
            for i, j, k in HYQMOM_35_INDICES
        ],
        dtype=float,
    )


def _standardized_central_targets(
    rho: float, third: np.ndarray, fourth: np.ndarray
) -> np.ndarray:
    targets = []
    for index in HYQMOM_35_INDICES:
        order = sum(index)
        directions = tuple(
            direction
            for direction, exponent in enumerate(index)
            for _ in range(exponent)
        )
        if order == 0:
            value = rho
        elif order == 1:
            value = 0.0
        elif order == 2:
            value = rho if directions[0] == directions[1] else 0.0
        elif order == 3:
            value = rho * third[directions]
        elif order == 4:
            value = rho * fourth[directions]
        else:  # pragma: no cover - the retained set stops at degree four
            raise ValueError("unexpected retained moment order")
        targets.append(value)
    return np.asarray(targets, dtype=float)


def _monomial_matrix(nodes: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            nodes[:, 0] ** i * nodes[:, 1] ** j * nodes[:, 2] ** k
            for i, j, k in HYQMOM_35_INDICES
        ],
        dtype=float,
    )


def _gauss_hermite_mixture_nodes(
    component_weights: np.ndarray,
    component_means: np.ndarray,
    component_covariances: np.ndarray,
    rho: float,
    quadrature_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    abscissae, one_dimensional_weights = np.polynomial.hermite.hermgauss(
        quadrature_nodes
    )
    abscissae = np.sqrt(2.0) * abscissae
    one_dimensional_weights = one_dimensional_weights / np.sqrt(np.pi)
    grid_indices = np.asarray(
        list(product(range(quadrature_nodes), repeat=3)), dtype=int
    )
    standard_nodes = np.column_stack(
        [abscissae[grid_indices[:, axis]] for axis in range(3)]
    )
    standard_weights = np.prod(
        np.column_stack(
            [one_dimensional_weights[grid_indices[:, axis]] for axis in range(3)]
        ),
        axis=1,
    )
    all_nodes = []
    all_weights = []
    for weight, mean, covariance in zip(
        component_weights, component_means, component_covariances
    ):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance_scale = max(float(np.max(eigenvalues)), 1.0)
        if np.min(eigenvalues) < -1.0e-11 * covariance_scale:
            raise ValueError("two-population component covariance is not positive")
        covariance_root = (
            eigenvectors
            @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
            @ eigenvectors.T
        )
        all_nodes.append(mean + standard_nodes @ covariance_root.T)
        all_weights.append(rho * weight * standard_weights)
    return np.concatenate(all_weights), np.vstack(all_nodes)


def _residual_correct_weights(
    base_weights: np.ndarray,
    standardized_nodes: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    matrix = _monomial_matrix(standardized_nodes)
    base_values = matrix @ base_weights
    scale = np.sqrt(np.maximum((matrix**2) @ base_weights, 1.0e-24))
    scaled_matrix = matrix / scale[:, None]
    right_hand_side = (targets - base_values) / scale
    gram = (scaled_matrix * base_weights[None, :]) @ scaled_matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    cutoff = max(float(np.max(eigenvalues)), 1.0) * 1.0e-12
    inverse = np.zeros_like(eigenvalues)
    inverse[eigenvalues > cutoff] = 1.0 / eigenvalues[eigenvalues > cutoff]
    coefficients = eigenvectors @ (
        inverse * (eigenvectors.T @ right_hand_side)
    )
    multiplicative_correction = scaled_matrix.T @ coefficients
    corrected = base_weights * (1.0 + multiplicative_correction)
    residual = float(
        np.linalg.norm((matrix @ corrected - targets) / scale) / np.sqrt(targets.size)
    )
    correction_norm = float(
        np.sqrt(np.dot(base_weights, multiplicative_correction**2) / np.sum(base_weights))
    )
    return corrected, residual, correction_norm


def reconstruct_two_population_quadrature(
    moments: Sequence[float],
    *,
    quadrature_nodes: int = 4,
    minimum_skewness_norm: float = 0.5,
    residual_correction: bool = True,
) -> TwoPopulationQuadrature:
    """Reconstruct a high-skew two-population quadrature from 35 moments."""

    vector = np.asarray(moments, dtype=float)
    if vector.shape != (35,) or not np.all(np.isfinite(vector)):
        raise ValueError("expected a finite 35-moment vector")
    if quadrature_nodes < 4:
        raise ValueError("four Gauss--Hermite nodes per direction are required")
    state = macroscopic_state(vector)
    square_root, inverse_square_root, third, fourth = _whitened_statistics(vector)
    skewness_tensor_norm = float(np.linalg.norm(third))
    if skewness_tensor_norm < minimum_skewness_norm:
        raise ValueError(
            "two-population closure is reserved for high-skew states: "
            f"||S3||={skewness_tensor_norm:.6e}"
        )
    direction, skewness = _dominant_skew_direction(third)
    projected_fourth = float(
        np.einsum(
            "ijkl,i,j,k,l",
            fourth,
            direction,
            direction,
            direction,
            direction,
            optimize=True,
        )
    )
    try:
        marginal = fit_equal_variance_marginal(1.0, skewness, projected_fourth)
    except ValueError:
        marginal = fit_location_scale_marginal(1.0, skewness, projected_fourth)

    component_weights = np.asarray(marginal.weights, dtype=float)
    standardized_means = marginal.means[:, None] * direction[None, :]
    standardized_covariances = np.asarray(
        [
            np.eye(3) + (variance - 1.0) * np.outer(direction, direction)
            for variance in marginal.component_variances
        ]
    )
    component_means = state.velocity + standardized_means @ square_root.T
    component_covariances = np.asarray(
        [square_root @ covariance @ square_root.T for covariance in standardized_covariances]
    )
    base_weights, nodes = _gauss_hermite_mixture_nodes(
        component_weights,
        component_means,
        component_covariances,
        state.rho,
        quadrature_nodes,
    )
    standardized_nodes = (nodes - state.velocity) @ inverse_square_root.T
    targets = _standardized_central_targets(state.rho, third, fourth)
    base_matrix = _monomial_matrix(standardized_nodes)
    target_scale = np.maximum(np.abs(targets), 1.0)
    base_relative_residual = float(
        np.linalg.norm((base_matrix @ base_weights - targets) / target_scale)
        / np.sqrt(targets.size)
    )
    if residual_correction:
        weights, corrected_residual, correction_norm = _residual_correct_weights(
            base_weights, standardized_nodes, targets
        )
    else:
        weights = base_weights
        corrected_residual = base_relative_residual
        correction_norm = 0.0
    reconstructed = _moments_from_weighted_nodes(weights, nodes)
    raw_residual = float(
        np.linalg.norm(reconstructed - vector) / max(np.linalg.norm(vector), 1.0e-15)
    )
    if residual_correction and (
        corrected_residual > 2.0e-8 or raw_residual > 2.0e-8
    ):
        raise FloatingPointError(
            "two-population residual quadrature did not reproduce retained moments: "
            f"standardized={corrected_residual:.3e}, raw={raw_residual:.3e}"
        )
    negative_mass = float(-np.sum(np.minimum(weights, 0.0)) / state.rho)
    return TwoPopulationQuadrature(
        weights=weights,
        nodes=nodes,
        reconstructed_moments=reconstructed,
        component_weights=component_weights,
        component_means=component_means,
        component_covariances=component_covariances,
        dominant_direction_standardized=direction,
        standardized_skewness=float(skewness),
        skewness_tensor_norm=skewness_tensor_norm,
        marginal_branch=marginal.branch,
        base_relative_moment_residual=base_relative_residual,
        corrected_relative_moment_residual=raw_residual,
        residual_correction_norm=correction_norm,
        negative_mass_fraction=negative_mass,
    )


def initialize_persistent_two_population(
    moments: Sequence[float],
    *,
    minimum_skewness_norm: float = 0.5,
) -> PersistentTwoPopulationState:
    """Initialize persistent Gaussian populations from a high-skew state."""

    vector = np.asarray(moments, dtype=float)
    state = macroscopic_state(vector)
    reconstruction = reconstruct_two_population_quadrature(
        vector,
        minimum_skewness_norm=minimum_skewness_norm,
        residual_correction=False,
    )
    return PersistentTwoPopulationState(
        rho=state.rho,
        weights=reconstruction.component_weights.copy(),
        means=reconstruction.component_means.copy(),
        covariances=reconstruction.component_covariances.copy(),
    )


def persistent_two_population_moments(
    populations: PersistentTwoPopulationState,
) -> np.ndarray:
    """Return the retained moments of the persistent Gaussian populations."""

    from .moments import mixture_of_gaussians_moments_35

    return mixture_of_gaussians_moments_35(
        [
            (
                populations.rho * float(weight),
                mean,
                covariance,
            )
            for weight, mean, covariance in zip(
                populations.weights,
                populations.means,
                populations.covariances,
            )
        ]
    )


def persistent_two_population_fp_step(
    populations: PersistentTwoPopulationState,
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
    quadrature_nodes: int = 4,
) -> tuple[
    PersistentTwoPopulationState,
    np.ndarray,
    PersistentTwoPopulationDiagnostics,
]:
    """Advance two labelled populations with a Gaussian assumed-density map.

    Each population is integrated with its own Gauss--Hermite nodes, but the
    physical FP coefficients are obtained from the combined distribution.
    After the nonlinear drift and exact OU noise, each labelled population is
    compressed back to its mean and full covariance.  Retaining the labels is
    the key difference from an algebraic reconstruction at every time step.
    """

    if dt <= 0.0 or tau <= 0.0:
        raise ValueError("dt and tau must be positive")
    if populations.weights.shape != (2,):
        raise ValueError("the persistent map currently requires two populations")
    incoming = persistent_two_population_moments(populations)
    state = macroscopic_state(incoming)
    quadrature_weights, nodes = _gauss_hermite_mixture_nodes(
        populations.weights,
        populations.means,
        populations.covariances,
        populations.rho,
        quadrature_nodes,
    )
    from .collision import coefficients_from_weighted_nodes

    coefficients = coefficients_from_weighted_nodes(
        nodes,
        quadrature_weights,
        tau=tau,
        prandtl=prandtl,
    )
    probabilities = quadrature_weights / populations.rho
    peculiar = nodes - state.velocity
    c2 = np.einsum("ni,ni->n", peculiar, peculiar)
    nonlinear = peculiar @ coefficients.C.T
    nonlinear += (c2 - 3.0 * state.theta)[:, None] * coefficients.gamma
    nonlinear += coefficients.beta * (
        c2[:, None] * peculiar
        - 2.0 * state.heat_flux[None, :] / state.rho
    )
    mean_n2 = float(
        np.dot(probabilities, np.einsum("ni,ni->n", nonlinear, nonlinear))
    )
    mean_cn = float(
        np.dot(probabilities, np.einsum("ni,ni->n", peculiar, nonlinear))
    )
    relaxation = float(np.exp(-dt / tau))
    relaxation_squared = relaxation**2
    alpha_squared = 1.0 + tau / (3.0 * state.theta) * (
        tau * (1.0 - relaxation) ** 2 * mean_n2
        + 2.0 * (relaxation - relaxation_squared) * mean_cn
    )
    alpha = float(np.sqrt(alpha_squared)) if alpha_squared > 1.0e-6 else 1.0
    noise_variance = state.theta * (1.0 - relaxation_squared) / alpha**2
    mapped_nodes = state.velocity + (
        relaxation * peculiar + (1.0 - relaxation) * tau * nonlinear
    ) / alpha

    nodes_per_population = quadrature_nodes**3
    mapped_means = []
    mapped_covariances = []
    for component, component_weight in enumerate(populations.weights):
        block = slice(
            component * nodes_per_population,
            (component + 1) * nodes_per_population,
        )
        local_probabilities = probabilities[block] / component_weight
        local_nodes = mapped_nodes[block]
        local_mean = np.sum(local_probabilities[:, None] * local_nodes, axis=0)
        centered = local_nodes - local_mean
        local_covariance = np.einsum(
            "n,ni,nj->ij", local_probabilities, centered, centered
        ) + noise_variance * np.eye(3)
        mapped_means.append(local_mean)
        mapped_covariances.append(local_covariance)
    mapped_means_array = np.asarray(mapped_means)
    mapped_covariances_array = np.asarray(mapped_covariances)

    mixture_mean = np.sum(
        populations.weights[:, None] * mapped_means_array, axis=0
    )
    mixture_covariance = np.zeros((3, 3))
    for weight, mean, covariance in zip(
        populations.weights, mapped_means_array, mapped_covariances_array
    ):
        offset = mean - mixture_mean
        mixture_covariance += weight * (covariance + np.outer(offset, offset))
    mapped_theta = float(np.trace(mixture_covariance) / 3.0)
    if mapped_theta <= 0.0:
        raise FloatingPointError("persistent map produced nonpositive temperature")
    affine_scale = float(np.sqrt(state.theta / mapped_theta))
    corrected_means = state.velocity + affine_scale * (
        mapped_means_array - mixture_mean
    )
    corrected_covariances = affine_scale**2 * mapped_covariances_array
    updated_populations = PersistentTwoPopulationState(
        rho=populations.rho,
        weights=populations.weights.copy(),
        means=corrected_means,
        covariances=corrected_covariances,
    )
    updated = persistent_two_population_moments(updated_populations)
    margin = realizability_margin_35(updated)
    if margin < -5.0e-13 or not np.all(np.isfinite(updated)):
        raise FloatingPointError("persistent two-population state is not realizable")
    return (
        updated_populations,
        updated,
        PersistentTwoPopulationDiagnostics(
            alpha=alpha,
            affine_scale=affine_scale,
            realizability_margin=float(margin),
            maximum_c2_over_theta=float(np.max(c2) / state.theta),
        ),
    )


def initialize_dynamic_high_order_state(
    moments: Sequence[float],
    *,
    minimum_skewness_norm: float = 0.5,
    maximum_order: int = 6,
) -> DynamicHighOrderState:
    """Initialize source-local M5/M6 from the two-population quadrature."""

    vector = np.asarray(moments, dtype=float)
    dynamic_indices = _dynamic_indices(maximum_order)
    quadrature = reconstruct_two_population_quadrature(
        vector,
        quadrature_nodes=max(4, (maximum_order + 2) // 2),
        minimum_skewness_norm=minimum_skewness_norm,
        residual_correction=False,
    )
    closure = WeightedNodeTailClosure(
        quadrature.nodes,
        quadrature.weights,
        maximum_order=maximum_order,
    )
    state = macroscopic_state(vector)
    tail = np.asarray(
        [closure(index, vector, state) for index in dynamic_indices],
        dtype=float,
    )
    return DynamicHighOrderState(
        moments=vector.copy(),
        tail_moments=tail,
        maximum_order=maximum_order,
    )


def dynamic_high_order_fp_step(
    dynamic_state: DynamicHighOrderState,
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
    margin_floor: float = 1.0e-12,
    minimum_skewness_norm: float = 0.05,
    high_order_quadrature_nodes: int | None = None,
) -> tuple[DynamicHighOrderState, DynamicHighOrderDiagnostics]:
    """Advance the 35 moments and dynamic M5/M6 with an M7/M8 tail closure."""

    if dt <= 0.0 or tau <= 0.0:
        raise ValueError("dt and tau must be positive")
    vector = np.asarray(dynamic_state.moments, dtype=float)
    tail = np.asarray(dynamic_state.tail_moments, dtype=float)
    dynamic_indices = _dynamic_indices(dynamic_state.maximum_order)
    if vector.shape != (35,) or tail.shape != (len(dynamic_indices),):
        raise ValueError("invalid dynamic high-order state")
    state = macroscopic_state(vector)
    if high_order_quadrature_nodes is None:
        high_order_quadrature_nodes = max(
            4, (dynamic_state.maximum_order + 4) // 2
        )
    algebraic_quadrature = reconstruct_two_population_quadrature(
        vector,
        quadrature_nodes=high_order_quadrature_nodes,
        minimum_skewness_norm=minimum_skewness_norm,
        residual_correction=False,
    )
    closure = _DynamicTailClosure(
        tail,
        algebraic_quadrature,
        dynamic_indices,
        dynamic_state.maximum_order,
    )

    from .collision import (
        CubicFPCoefficients,
        coefficients_from_moments,
        fp_collision_moment_source,
        projected_fp_collision_source,
    )

    coefficients = coefficients_from_moments(
        vector, tau=tau, prandtl=prandtl, closure=closure
    )
    source = projected_fp_collision_source(vector, coefficients, closure=closure)
    ou_coefficients = CubicFPCoefficients.ornstein_uhlenbeck(
        tau=tau, theta=state.theta
    )
    ou_source = projected_fp_collision_source(vector, ou_coefficients, closure=closure)
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
        if realizability_margin_35(ou_mapped) < margin_floor:
            raise FloatingPointError("dynamic exact OU base state is not realizable")
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
        raise FloatingPointError("dynamic high-order step is not realizable")

    full_tail_source = np.asarray(
        [
            fp_collision_moment_source(
                index, vector, coefficients, closure=closure, state=state
            )
            for index in dynamic_indices
        ]
    )
    ou_tail_source = np.asarray(
        [
            fp_collision_moment_source(
                index, vector, ou_coefficients, closure=closure, state=state
            )
            for index in dynamic_indices
        ]
    )
    updated_tail = tail + dt * (
        ou_tail_source + limiter * (full_tail_source - ou_tail_source)
    )
    if not np.all(np.isfinite(updated_tail)):
        raise FloatingPointError("dynamic M5/M6 update produced NaN or infinity")
    algebraic_closure = WeightedNodeTailClosure(
        algebraic_quadrature.nodes,
        algebraic_quadrature.weights,
        maximum_order=dynamic_state.maximum_order,
    )
    algebraic_tail = np.asarray(
        [algebraic_closure(index, vector, state) for index in dynamic_indices]
    )
    relative_difference = float(
        np.linalg.norm(tail - algebraic_tail)
        / max(np.linalg.norm(algebraic_tail), 1.0e-14)
    )
    next_state = DynamicHighOrderState(
        moments=updated,
        tail_moments=updated_tail,
        maximum_order=dynamic_state.maximum_order,
    )
    return next_state, DynamicHighOrderDiagnostics(
        limiter_fraction=float(limiter),
        realizability_margin=float(margin),
        tail_increment_norm=float(np.linalg.norm(updated_tail - tail)),
        algebraic_tail_relative_difference=relative_difference,
        source_norm=float(np.linalg.norm(source)),
    )


def two_population_fp_step(
    moments: Sequence[float],
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
    margin_floor: float = 1.0e-12,
    quadrature_nodes: int = 4,
    minimum_skewness_norm: float = 0.5,
    residual_correction: bool = True,
) -> tuple[np.ndarray, TwoPopulationStepDiagnostics]:
    """Advance the exact-OU/high-skew source with a realizability guard."""

    if dt <= 0.0 or tau <= 0.0:
        raise ValueError("dt and tau must be positive")
    vector = np.asarray(moments, dtype=float)
    state = macroscopic_state(vector)
    quadrature = reconstruct_two_population_quadrature(
        vector,
        quadrature_nodes=quadrature_nodes,
        minimum_skewness_norm=minimum_skewness_norm,
        residual_correction=residual_correction,
    )
    closure = WeightedNodeTailClosure(quadrature.nodes, quadrature.weights)

    from .collision import (
        CubicFPCoefficients,
        coefficients_from_moments,
        projected_fp_collision_source,
    )

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
        if realizability_margin_35(ou_mapped) < margin_floor:
            raise FloatingPointError("exact OU base state is not realizable")
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
        raise FloatingPointError("limited two-population step is not realizable")
    return updated, TwoPopulationStepDiagnostics(
        limiter_fraction=float(limiter),
        realizability_margin=float(margin),
        skewness_tensor_norm=quadrature.skewness_tensor_norm,
        standardized_skewness=quadrature.standardized_skewness,
        marginal_branch=quadrature.marginal_branch,
        base_relative_moment_residual=quadrature.base_relative_moment_residual,
        corrected_relative_moment_residual=quadrature.corrected_relative_moment_residual,
        residual_correction_norm=quadrature.residual_correction_norm,
        negative_mass_fraction=quadrature.negative_mass_fraction,
        source_norm=float(np.linalg.norm(source)),
        nonlinear_source_norm=float(np.linalg.norm(nonlinear_source)),
    )

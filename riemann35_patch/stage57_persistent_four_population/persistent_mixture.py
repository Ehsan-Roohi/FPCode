"""Positive persistent Gaussian-population collision closure.

Stage 56 proved that exact time integration converges, but that repeated
relaxation toward an algebraic two-population tail converges to the wrong
trace-free third-order dynamics.  This module retains the labelled positive
Gaussian populations instead.  The state contains no velocity microstate:
only density, population probabilities, means, and symmetric covariances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from hyqmom_fp import HYQMOM_35_INDICES, macroscopic_state, mixture_of_gaussians_moments_35
from hyqmom_fp.collision import coefficients_from_weighted_nodes
from hyqmom_fp.mixture_closure import realizability_margin_35
from hyqmom_fp.moments import central_moment
from hyqmom_fp.two_population import _gauss_hermite_mixture_nodes


THIRD_INDICES = tuple(index for index in HYQMOM_35_INDICES if sum(index) == 3)
SYMMETRIC_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


@dataclass(frozen=True)
class PersistentGaussianMixtureState:
    """Compact positive state for labelled Gaussian populations."""

    rho: float
    probabilities: np.ndarray
    means: np.ndarray
    covariances: np.ndarray


@dataclass(frozen=True)
class PersistentMixtureDiagnostics:
    """Diagnostics for one assumed-density collision step."""

    alpha: float
    affine_scale: float
    realizability_margin: float
    minimum_quadrature_weight: float
    minimum_covariance_eigenvalue: float
    maximum_c2_over_theta: float
    stored_scalar_count: int
    heat_flux_projection_fraction: float
    heat_flux_projection_residual: float


def _validate_state(state: PersistentGaussianMixtureState) -> None:
    count = state.probabilities.size
    if not np.isfinite(state.rho) or state.rho <= 0.0:
        raise ValueError("mixture density must be finite and positive")
    if count < 2:
        raise ValueError("at least two persistent populations are required")
    if state.means.shape != (count, 3):
        raise ValueError("population means must have shape (count, 3)")
    if state.covariances.shape != (count, 3, 3):
        raise ValueError("population covariances must have shape (count, 3, 3)")
    if not np.all(np.isfinite(state.probabilities)) or np.any(state.probabilities <= 0.0):
        raise ValueError("population probabilities must be finite and positive")
    if not np.isclose(np.sum(state.probabilities), 1.0, rtol=0.0, atol=2.0e-13):
        raise ValueError("population probabilities must sum to one")
    if not np.all(np.isfinite(state.means)) or not np.all(np.isfinite(state.covariances)):
        raise ValueError("population parameters contain NaN or infinity")
    for covariance in state.covariances:
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=2.0e-13):
            raise ValueError("population covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
            raise ValueError("population covariance must be positive definite")


def initialize_persistent_gaussian_mixture(
    components: Iterable[
        tuple[float, Sequence[float], Sequence[Sequence[float]]]
    ],
) -> PersistentGaussianMixtureState:
    """Initialize the compact state from positive Gaussian components."""

    materialized = tuple(components)
    if len(materialized) < 2:
        raise ValueError("at least two Gaussian components are required")
    partial_densities = np.asarray([item[0] for item in materialized], dtype=float)
    if np.any(~np.isfinite(partial_densities)) or np.any(partial_densities <= 0.0):
        raise ValueError("component densities must be finite and positive")
    rho = float(np.sum(partial_densities))
    state = PersistentGaussianMixtureState(
        rho=rho,
        probabilities=partial_densities / rho,
        means=np.asarray([item[1] for item in materialized], dtype=float),
        covariances=np.asarray([item[2] for item in materialized], dtype=float),
    )
    _validate_state(state)
    return state


def persistent_gaussian_mixture_moments(
    state: PersistentGaussianMixtureState,
) -> np.ndarray:
    """Return the retained 35 moments of the compact positive state."""

    _validate_state(state)
    return mixture_of_gaussians_moments_35(
        [
            (state.rho * float(probability), mean, covariance)
            for probability, mean, covariance in zip(
                state.probabilities, state.means, state.covariances
            )
        ]
    )


def stored_scalar_count(state: PersistentGaussianMixtureState) -> int:
    """Count density, weights, means, and unique covariance entries."""

    count = state.probabilities.size
    return 1 + count + 3 * count + 6 * count


def _third_components(moments: np.ndarray) -> np.ndarray:
    return np.asarray([central_moment(moments, index) for index in THIRD_INDICES])


def _symmetric_third_tensor(components: np.ndarray) -> np.ndarray:
    tensor = np.zeros((3, 3, 3), dtype=float)
    for position, powers in enumerate(THIRD_INDICES):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    counts = (
                        int(i == 0) + int(j == 0) + int(k == 0),
                        int(i == 1) + int(j == 1) + int(k == 1),
                        int(i == 2) + int(j == 2) + int(k == 2),
                    )
                    if counts == powers:
                        tensor[i, j, k] = components[position]
    return tensor


def _unique_third_components(tensor: np.ndarray) -> np.ndarray:
    values = []
    for powers in THIRD_INDICES:
        directions = tuple(
            direction
            for direction, power in enumerate(powers)
            for _ in range(power)
        )
        values.append(tensor[directions[0], directions[1], directions[2]])
    return np.asarray(values)


def _trace_carrying_tensor(heat_flux: np.ndarray) -> np.ndarray:
    trace = 2.0 * np.asarray(heat_flux, dtype=float)
    identity = np.eye(3)
    return (
        np.einsum("ij,k->ijk", identity, trace)
        + np.einsum("ik,j->ijk", identity, trace)
        + np.einsum("jk,i->ijk", identity, trace)
    ) / 5.0


def _heat_flux_trace_projection(
    populations: PersistentGaussianMixtureState,
    desired_heat_flux: np.ndarray,
) -> tuple[PersistentGaussianMixtureState, float, float]:
    """Match the physical heat-flux trace while preserving lower moments.

    A four-population state has 36 mean/covariance degrees of freedom.  A
    small constrained Newton projection uses them to preserve the mixture
    mean, the full covariance, and the trace-free third tensor while imposing
    the analytic cubic-FP heat-flux relaxation.  Positivity follows from fixed
    positive weights and an SPD line search on every covariance.
    """

    baseline_moments = persistent_gaussian_mixture_moments(populations)
    baseline_macro = macroscopic_state(baseline_moments)
    baseline_tensor = _symmetric_third_tensor(_third_components(baseline_moments))
    baseline_heat_flux = 0.5 * np.einsum("ijj->i", baseline_tensor)
    target_tensor = (
        baseline_tensor
        - _trace_carrying_tensor(baseline_heat_flux)
        + _trace_carrying_tensor(desired_heat_flux)
    )
    target_third = _unique_third_components(target_tensor)
    target_covariance = baseline_macro.covariance
    target_mean = baseline_macro.velocity
    count = populations.probabilities.size
    projected = populations
    minimum_fraction = 1.0

    # The third central moments are raw density-weighted moments, whereas the
    # mean and covariance constraints are normalized per unit mass.  Their
    # Newton rows therefore carry an explicit rho factor.  Omitting it leaves
    # unit-density cases unchanged but mis-scales the projection for rho != 1.
    for _ in range(20):
        current_moments = persistent_gaussian_mixture_moments(projected)
        current_macro = macroscopic_state(current_moments)
        current_third = _third_components(current_moments)
        mean_delta = target_mean - current_macro.velocity
        covariance_delta_target = target_covariance - current_macro.covariance
        covariance_right = np.asarray(
            [covariance_delta_target[left, right] for left, right in SYMMETRIC_PAIRS]
        )
        third_right = target_third - current_third
        right_hand_side = np.concatenate([mean_delta, covariance_right, third_right])
        if np.linalg.norm(right_hand_side) <= 2.0e-13:
            break

        matrix = np.zeros((3 + 6 + 10, count * 9))
        offsets = projected.means - current_macro.velocity
        for component, (probability, offset, covariance) in enumerate(
            zip(projected.probabilities, offsets, projected.covariances)
        ):
            for direction in range(3):
                column = component * 9 + direction
                basis_mean = np.zeros(3)
                basis_mean[direction] = 1.0
                matrix[direction, column] = probability
                covariance_derivative = probability * (
                    np.outer(offset, basis_mean) + np.outer(basis_mean, offset)
                )
                for pair_position, (left, right) in enumerate(SYMMETRIC_PAIRS):
                    matrix[3 + pair_position, column] = covariance_derivative[left, right]
                for third_position, powers in enumerate(THIRD_INDICES):
                    directions = tuple(
                        axis for axis, power in enumerate(powers) for _ in range(power)
                    )
                    i, j, k = directions
                    matrix[9 + third_position, column] = populations.rho * probability * (
                        basis_mean[i] * (offset[j] * offset[k] + covariance[j, k])
                        + basis_mean[j] * (offset[i] * offset[k] + covariance[i, k])
                        + basis_mean[k] * (offset[i] * offset[j] + covariance[i, j])
                    )
            for pair_position, (left, right) in enumerate(SYMMETRIC_PAIRS):
                column = component * 9 + 3 + pair_position
                basis_covariance = np.zeros((3, 3))
                basis_covariance[left, right] = 1.0
                basis_covariance[right, left] = 1.0
                for covariance_position, (a, b) in enumerate(SYMMETRIC_PAIRS):
                    matrix[3 + covariance_position, column] = (
                        probability * basis_covariance[a, b]
                    )
                for third_position, powers in enumerate(THIRD_INDICES):
                    directions = tuple(
                        axis for axis, power in enumerate(powers) for _ in range(power)
                    )
                    i, j, k = directions
                    matrix[9 + third_position, column] = populations.rho * probability * (
                        offset[i] * basis_covariance[j, k]
                        + offset[j] * basis_covariance[i, k]
                        + offset[k] * basis_covariance[i, j]
                    )

        row_scale = np.maximum(np.linalg.norm(matrix, axis=1), 1.0e-12)
        scaled_matrix = matrix / row_scale[:, None]
        scaled_right = right_hand_side / row_scale
        correction = np.linalg.lstsq(
            scaled_matrix, scaled_right, rcond=1.0e-11
        )[0]
        mean_correction = correction.reshape(count, 9)[:, :3]
        covariance_correction = np.zeros_like(projected.covariances)
        for component in range(count):
            for pair_position, (left, right) in enumerate(SYMMETRIC_PAIRS):
                value = correction[component * 9 + 3 + pair_position]
                covariance_correction[component, left, right] = value
                covariance_correction[component, right, left] = value
        fraction = 1.0
        for _ in range(60):
            trial_covariances = projected.covariances + fraction * covariance_correction
            if min(np.min(np.linalg.eigvalsh(item)) for item in trial_covariances) > 1.0e-12:
                break
            fraction *= 0.5
        else:
            raise FloatingPointError("heat-flux trace projection could not preserve SPD")
        minimum_fraction = min(minimum_fraction, fraction)
        projected = PersistentGaussianMixtureState(
            rho=projected.rho,
            probabilities=projected.probabilities.copy(),
            means=projected.means + fraction * mean_correction,
            covariances=projected.covariances + fraction * covariance_correction,
        )
        _validate_state(projected)

    final_moments = persistent_gaussian_mixture_moments(projected)
    final_macro = macroscopic_state(final_moments)
    final_tensor = _symmetric_third_tensor(_third_components(final_moments))
    lower_scale = max(np.linalg.norm(target_covariance), 1.0e-14)
    # Near equilibrium the target third tensor approaches zero, so dividing
    # only by its norm turns round-off-level absolute errors into a misleading
    # large relative residual.  rho*theta^(3/2) is the natural third-moment
    # scale and remains finite throughout relaxation.
    third_scale = max(
        np.linalg.norm(target_tensor),
        populations.rho * max(baseline_macro.theta, 1.0e-14) ** 1.5,
        1.0e-14,
    )
    residual = max(
        float(np.linalg.norm(final_macro.velocity - target_mean)),
        float(np.linalg.norm(final_macro.covariance - target_covariance) / lower_scale),
        float(np.linalg.norm(final_tensor - target_tensor) / third_scale),
    )
    return projected, minimum_fraction, residual


def persistent_gaussian_mixture_fp_step(
    populations: PersistentGaussianMixtureState,
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
    quadrature_nodes: int = 5,
    enforce_heat_flux_rate: bool = True,
) -> tuple[PersistentGaussianMixtureState, np.ndarray, PersistentMixtureDiagnostics]:
    """Advance labelled positive populations with an assumed-density map.

    The cubic drift is evaluated on a positive tensor Gauss--Hermite rule.
    Each labelled population is mapped separately and compressed to its mean
    and full covariance.  A single affine correction restores the conserved
    mixture momentum and energy trace without changing positive weights or
    covariance definiteness.
    """

    _validate_state(populations)
    if dt <= 0.0 or tau <= 0.0:
        raise ValueError("dt and tau must be positive")
    if quadrature_nodes < 4:
        raise ValueError("at least four quadrature nodes per direction are required")
    incoming = persistent_gaussian_mixture_moments(populations)
    macro = macroscopic_state(incoming)
    quadrature_weights, nodes = _gauss_hermite_mixture_nodes(
        populations.probabilities,
        populations.means,
        populations.covariances,
        populations.rho,
        quadrature_nodes,
    )
    coefficients = coefficients_from_weighted_nodes(
        nodes,
        quadrature_weights,
        tau=tau,
        prandtl=prandtl,
    )
    probabilities = quadrature_weights / populations.rho
    peculiar = nodes - macro.velocity
    c2 = np.einsum("ni,ni->n", peculiar, peculiar)
    nonlinear = peculiar @ coefficients.C.T
    nonlinear += (c2 - 3.0 * macro.theta)[:, None] * coefficients.gamma
    nonlinear += coefficients.beta * (
        c2[:, None] * peculiar
        - 2.0 * macro.heat_flux[None, :] / populations.rho
    )
    mean_n2 = float(
        np.dot(probabilities, np.einsum("ni,ni->n", nonlinear, nonlinear))
    )
    mean_cn = float(
        np.dot(probabilities, np.einsum("ni,ni->n", peculiar, nonlinear))
    )
    relaxation = float(np.exp(-dt / tau))
    relaxation_squared = relaxation**2
    alpha_squared = 1.0 + tau / (3.0 * macro.theta) * (
        tau * (1.0 - relaxation) ** 2 * mean_n2
        + 2.0 * (relaxation - relaxation_squared) * mean_cn
    )
    if alpha_squared <= 1.0e-12:
        raise FloatingPointError("persistent mixture produced nonpositive alpha squared")
    alpha = float(np.sqrt(alpha_squared))
    noise_variance = macro.theta * (1.0 - relaxation_squared) / alpha**2
    mapped_nodes = macro.velocity + (
        relaxation * peculiar + (1.0 - relaxation) * tau * nonlinear
    ) / alpha

    nodes_per_population = quadrature_nodes**3
    mapped_means = []
    mapped_covariances = []
    for component, component_probability in enumerate(populations.probabilities):
        block = slice(
            component * nodes_per_population,
            (component + 1) * nodes_per_population,
        )
        local_probabilities = probabilities[block] / component_probability
        local_nodes = mapped_nodes[block]
        local_mean = np.sum(local_probabilities[:, None] * local_nodes, axis=0)
        centered = local_nodes - local_mean
        local_covariance = np.einsum(
            "n,ni,nj->ij", local_probabilities, centered, centered
        ) + noise_variance * np.eye(3)
        mapped_means.append(local_mean)
        mapped_covariances.append(0.5 * (local_covariance + local_covariance.T))
    mapped_means_array = np.asarray(mapped_means)
    mapped_covariances_array = np.asarray(mapped_covariances)

    mixture_mean = np.sum(
        populations.probabilities[:, None] * mapped_means_array, axis=0
    )
    mixture_covariance = np.zeros((3, 3))
    for probability, mean, covariance in zip(
        populations.probabilities, mapped_means_array, mapped_covariances_array
    ):
        offset = mean - mixture_mean
        mixture_covariance += probability * (covariance + np.outer(offset, offset))
    mapped_theta = float(np.trace(mixture_covariance) / 3.0)
    if mapped_theta <= 0.0:
        raise FloatingPointError("persistent mixture produced nonpositive temperature")
    affine_scale = float(np.sqrt(macro.theta / mapped_theta))
    corrected_means = macro.velocity + affine_scale * (
        mapped_means_array - mixture_mean
    )
    corrected_covariances = affine_scale**2 * mapped_covariances_array
    updated_populations = PersistentGaussianMixtureState(
        rho=populations.rho,
        probabilities=populations.probabilities.copy(),
        means=corrected_means,
        covariances=corrected_covariances,
    )
    _validate_state(updated_populations)
    projection_fraction = 1.0
    projection_residual = 0.0
    if enforce_heat_flux_rate:
        desired_heat_flux = np.exp(-2.0 * prandtl * dt / tau) * macro.heat_flux
        updated_populations, projection_fraction, projection_residual = (
            _heat_flux_trace_projection(updated_populations, desired_heat_flux)
        )
    updated = persistent_gaussian_mixture_moments(updated_populations)
    margin = float(realizability_margin_35(updated))
    if margin < -5.0e-13 or not np.all(np.isfinite(updated)):
        raise FloatingPointError("persistent mixture moment state is not realizable")
    minimum_covariance = float(
        min(np.min(np.linalg.eigvalsh(item)) for item in corrected_covariances)
    )
    return (
        updated_populations,
        updated,
        PersistentMixtureDiagnostics(
            alpha=alpha,
            affine_scale=affine_scale,
            realizability_margin=margin,
            minimum_quadrature_weight=float(np.min(quadrature_weights)),
            minimum_covariance_eigenvalue=minimum_covariance,
            maximum_c2_over_theta=float(np.max(c2) / macro.theta),
            stored_scalar_count=stored_scalar_count(updated_populations),
            heat_flux_projection_fraction=projection_fraction,
            heat_flux_projection_residual=projection_residual,
        ),
    )
"""Positive discrete maximum-entropy closure for the 35-moment state."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from scipy.special import logsumexp

from .collision import (
    CubicFPCoefficients,
    coefficients_from_moments,
    projected_fp_collision_source,
)
from .grad_hyqmom import WeightedNodeTailClosure
from .mixture_closure import exact_ou_moment_map, realizability_margin_35
from .moments import (
    HYQMOM_35_INDICES,
    central_moment,
    macroscopic_state,
    moment_value,
)


@dataclass(frozen=True)
class MaximumEntropyQuadrature:
    """Positive velocity quadrature matching all moments through degree four."""

    nodes: np.ndarray
    weights: np.ndarray
    reconstructed_moments: np.ndarray
    relative_moment_residual: float
    scaled_constraint_residual: float
    iterations: int
    dual_parameters: np.ndarray
    minimum_probability: float


@dataclass(frozen=True)
class MaximumEntropyStepDiagnostics:
    """Diagnostics for one guarded maximum-entropy FP source step."""

    limiter_fraction: float
    realizability_margin: float
    relative_moment_residual: float
    scaled_constraint_residual: float
    iterations: int
    minimum_probability: float
    source_norm: float
    nonlinear_source_norm: float
    dual_parameters: np.ndarray


def _central_tensor(moments: np.ndarray, order: int) -> np.ndarray:
    state = macroscopic_state(moments)
    tensor = np.empty((3,) * order, dtype=float)
    for directions in product(range(3), repeat=order):
        index = [0, 0, 0]
        for direction in directions:
            index[direction] += 1
        tensor[directions] = central_moment(moments, tuple(index)) / state.rho
    return tensor


def _standardized_targets(
    moments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = macroscopic_state(moments)
    eigenvalues, eigenvectors = np.linalg.eigh(state.covariance)
    if np.min(eigenvalues) <= 1.0e-13 * max(np.max(eigenvalues), 1.0):
        raise ValueError("maximum-entropy whitening covariance is near singular")
    square_root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_root = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    tensors = {}
    for order in (3, 4):
        central = _central_tensor(moments, order)
        labels = "ia,jb,kc" if order == 3 else "ia,jb,kc,ld"
        output = "ijk" if order == 3 else "ijkl"
        tensors[order] = np.einsum(
            f"{labels},{'abc' if order == 3 else 'abcd'}->{output}",
            *([inverse_root] * order),
            central,
            optimize=True,
        )
    targets = []
    for index in HYQMOM_35_INDICES:
        order = sum(index)
        if order == 0:
            value = 1.0
        elif order == 1:
            value = 0.0
        elif order == 2:
            value = 1.0 if 2 in index else 0.0
        else:
            directions = tuple(
                direction
                for direction, exponent in enumerate(index)
                for _ in range(exponent)
            )
            value = float(tensors[order][directions])
        targets.append(value)
    return np.asarray(targets), square_root, state.velocity


def _standard_normal_tensor_rule(nodes_per_dimension: int) -> tuple[np.ndarray, np.ndarray]:
    if nodes_per_dimension < 8:
        raise ValueError("maximum-entropy closure needs at least eight nodes per direction")
    nodes_1d, weights_1d = np.polynomial.hermite_e.hermegauss(nodes_per_dimension)
    weights_1d = weights_1d / np.sqrt(2.0 * np.pi)
    indices = np.asarray(
        list(product(range(nodes_per_dimension), repeat=3)), dtype=int
    )
    nodes = np.column_stack([nodes_1d[indices[:, axis]] for axis in range(3)])
    weights = np.prod(
        np.column_stack([weights_1d[indices[:, axis]] for axis in range(3)]),
        axis=1,
    )
    return nodes, weights


def _features(nodes: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.prod(nodes ** np.asarray(index)[None, :], axis=1)
            for index in HYQMOM_35_INDICES[1:]
        ]
    )


def reconstruct_maximum_entropy_quadrature(
    moments: Sequence[float],
    *,
    nodes_per_dimension: int = 6,
    initial_parameters: Sequence[float] | None = None,
    tolerance: float = 1.0e-7,
    maximum_iterations: int = 80,
) -> MaximumEntropyQuadrature:
    """Solve the positive discrete entropy dual on a whitened GH grid."""

    vector = np.asarray(moments, dtype=float)
    if vector.shape != (35,) or not np.all(np.isfinite(vector)):
        raise ValueError("expected a finite 35-moment vector")
    state = macroscopic_state(vector)
    target, square_root, velocity = _standardized_targets(vector)
    # An isotropic global Gauss--Hermite grid does not contain near-boundary
    # rare-beam moments in the interior of its discrete convex hull.  Use the
    # positive high-skew reconstruction only as an adaptive proposal/support;
    # the entropy solve below is still responsible for all 34 constraints.
    from .two_population import reconstruct_two_population_quadrature

    proposal = reconstruct_two_population_quadrature(
        vector,
        quadrature_nodes=nodes_per_dimension,
        minimum_skewness_norm=0.05,
        residual_correction=False,
    )
    inverse_root = np.linalg.inv(square_root)
    standardized_nodes = (proposal.nodes - velocity) @ inverse_root.T
    prior = proposal.weights / state.rho
    prior = prior / np.sum(prior)
    raw_features = _features(standardized_nodes)
    raw_target = target[1:]
    feature_scale = np.sqrt(
        np.maximum(np.sum(prior[:, None] * raw_features**2, axis=0), 1.0e-24)
    )
    features = raw_features / feature_scale[None, :]
    target_scaled = raw_target / feature_scale
    if initial_parameters is None:
        parameters = np.zeros(features.shape[1])
    else:
        parameters = np.asarray(initial_parameters, dtype=float).copy()
        if parameters.shape != (features.shape[1],):
            raise ValueError("initial maximum-entropy parameter shape mismatch")
    log_prior = np.log(prior)

    def evaluate(candidate: np.ndarray):
        log_weight = log_prior + features @ candidate
        log_normalization = logsumexp(log_weight)
        probabilities = np.exp(log_weight - log_normalization)
        mean = probabilities @ features
        residual = mean - target_scaled
        objective = float(log_normalization - np.dot(candidate, target_scaled))
        return objective, probabilities, mean, residual

    objective, probabilities, mean, residual = evaluate(parameters)
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tolerance:
            break
        centered = features - mean[None, :]
        hessian = centered.T @ (probabilities[:, None] * centered)
        regularization = 1.0e-12 * max(float(np.linalg.norm(hessian, ord=np.inf)), 1.0)
        try:
            direction = np.linalg.solve(
                hessian + regularization * np.eye(hessian.shape[0]), residual
            )
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(
                hessian + regularization * np.eye(hessian.shape[0]),
                residual,
                rcond=1.0e-12,
            )[0]
        directional_derivative = float(np.dot(residual, direction))
        step = 1.0
        accepted = False
        for _ in range(30):
            candidate = parameters - step * direction
            trial = evaluate(candidate)
            if np.isfinite(trial[0]) and trial[0] <= objective - 1.0e-4 * step * directional_derivative:
                parameters = candidate
                objective, probabilities, mean, residual = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            if residual_norm <= 10.0 * tolerance:
                break
            raise FloatingPointError("maximum-entropy Newton line search stalled")
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm > 10.0 * tolerance:
        raise FloatingPointError(
            f"maximum-entropy constraints did not converge: {residual_norm:.3e}"
        )
    physical_nodes = velocity + standardized_nodes @ square_root.T
    weights = state.rho * probabilities
    reconstructed = np.asarray(
        [
            np.dot(
                weights,
                physical_nodes[:, 0] ** index[0]
                * physical_nodes[:, 1] ** index[1]
                * physical_nodes[:, 2] ** index[2],
            )
            for index in HYQMOM_35_INDICES
        ]
    )
    relative_residual = float(
        np.linalg.norm(reconstructed - vector) / max(np.linalg.norm(vector), 1.0e-15)
    )
    return MaximumEntropyQuadrature(
        nodes=physical_nodes,
        weights=weights,
        reconstructed_moments=reconstructed,
        relative_moment_residual=relative_residual,
        scaled_constraint_residual=residual_norm,
        iterations=iterations,
        dual_parameters=parameters,
        minimum_probability=float(np.min(probabilities)),
    )


def maximum_entropy_fp_step(
    moments: Sequence[float],
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
    nodes_per_dimension: int = 6,
    initial_parameters: Sequence[float] | None = None,
    margin_floor: float = 1.0e-12,
) -> tuple[np.ndarray, MaximumEntropyStepDiagnostics]:
    """Advance an exact-OU/guarded source using the positive MaxEnt tail."""

    vector = np.asarray(moments, dtype=float)
    state = macroscopic_state(vector)
    quadrature = reconstruct_maximum_entropy_quadrature(
        vector,
        nodes_per_dimension=nodes_per_dimension,
        initial_parameters=initial_parameters,
    )
    closure = WeightedNodeTailClosure(quadrature.nodes, quadrature.weights)
    coefficients = coefficients_from_moments(
        vector, tau=tau, prandtl=prandtl, closure=closure
    )
    full_source = projected_fp_collision_source(vector, coefficients, closure)
    ou_coefficients = CubicFPCoefficients.ornstein_uhlenbeck(
        tau=tau, theta=state.theta
    )
    ou_source = projected_fp_collision_source(vector, ou_coefficients, closure)
    nonlinear_source = full_source - ou_source
    relaxation = float(np.exp(-dt / tau))
    base = exact_ou_moment_map(vector, relaxation, state.theta)
    increment = dt * nonlinear_source
    limiter = 1.0
    candidate = base + increment
    margin = realizability_margin_35(candidate)
    if not np.isfinite(margin) or margin < margin_floor:
        lower, upper = 0.0, 1.0
        for _ in range(60):
            middle = 0.5 * (lower + upper)
            trial_margin = realizability_margin_35(base + middle * increment)
            if np.isfinite(trial_margin) and trial_margin >= margin_floor:
                lower = middle
            else:
                upper = middle
        limiter = lower
        candidate = base + limiter * increment
        margin = realizability_margin_35(candidate)
    # Exact collision invariants are retained after the split/guard.
    position = {index: number for number, index in enumerate(HYQMOM_35_INDICES)}
    for index in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        candidate[position[index]] = vector[position[index]]
    energy_indices = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    error = sum(vector[position[index]] for index in energy_indices) - sum(
        candidate[position[index]] for index in energy_indices
    )
    for index in energy_indices:
        candidate[position[index]] += error / 3.0
    return candidate, MaximumEntropyStepDiagnostics(
        limiter_fraction=float(limiter),
        realizability_margin=float(realizability_margin_35(candidate)),
        relative_moment_residual=quadrature.relative_moment_residual,
        scaled_constraint_residual=quadrature.scaled_constraint_residual,
        iterations=quadrature.iterations,
        minimum_probability=quadrature.minimum_probability,
        source_norm=float(np.linalg.norm(full_source)),
        nonlinear_source_norm=float(np.linalg.norm(nonlinear_source)),
        dual_parameters=quadrature.dual_parameters,
    )

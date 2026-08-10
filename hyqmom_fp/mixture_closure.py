"""Finite-width Gaussian-mixture closure for the cubic FP collision map.

The 35-moment Riemann state contains every raw moment through total degree
four.  A cubic velocity drift needs moments through degree six when its source
is projected analytically.  This module instead reconstructs a smooth local
Gaussian mixture from the retained second-, third-, and fourth-order moments
and applies the same finite-step map used by the particle reference.

The reconstruction is performed in the principal-axis frame of the retained
covariance.  On the platykurtic branch, an equal-variance two-Gaussian Pearson
fit solves

    2 v**3 + kappa4*v - kappa3**2 = 0,

where ``v`` is the between-component variance.  On the symmetric leptokurtic
branch, where that ansatz is structurally incomplete, a two-Gaussian
location--scale fit with unequal component variances is used instead.  This
keeps the sixth moment finite and continuous as the third cumulant approaches
zero.  Four-point Gauss--Hermite
quadrature per active component integrates all input moments needed by the
cubic map.  In the symmetric benchmark only the streamwise marginal is
bimodal, so the tensor rule has ``8 x 4 x 4 = 128`` nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb, factorial
from typing import Sequence

import numpy as np

from .collision import coefficients_from_weighted_nodes
from .moments import HYQMOM_35_INDICES, central_moment, macroscopic_state


POSITION = {index: position for position, index in enumerate(HYQMOM_35_INDICES)}


@dataclass(frozen=True)
class MarginalMixture:
    """One-dimensional Gaussian location--scale mixture."""

    weights: np.ndarray
    means: np.ndarray
    variance: float
    component_variances: np.ndarray
    between_variance: float
    reconstruction_error: float
    branch: str


@dataclass(frozen=True)
class MixtureQuadrature:
    """Tensor Gauss--Hermite representation of the reconstructed VDF."""

    weights: np.ndarray
    nodes: np.ndarray
    reconstructed_moments: np.ndarray
    eigenvectors: np.ndarray
    marginals: tuple[MarginalMixture, MarginalMixture, MarginalMixture]
    relative_moment_residual: float


@dataclass(frozen=True)
class FiniteMixtureStepDiagnostics:
    """Diagnostics for one deterministic finite collision interval."""

    alpha: float
    alpha_squared: float
    affine_scale: float
    maximum_c2_over_theta: float
    quadrature_nodes: int
    reconstruction_relative_residual: float
    increment_norm: float
    realizability_margin: float


def _multinomial(alpha: tuple[int, int, int]) -> int:
    order = sum(alpha)
    return factorial(order) // np.prod([factorial(value) for value in alpha])


def _contracted_central_moment(
    moments: Sequence[float], direction: np.ndarray, order: int
) -> float:
    """Return E[(direction dot (v-u))**order] from retained moments."""

    rho = float(np.asarray(moments)[POSITION[(0, 0, 0)]])
    value = 0.0
    for ax in range(order + 1):
        for ay in range(order - ax + 1):
            az = order - ax - ay
            alpha = (ax, ay, az)
            value += (
                _multinomial(alpha)
                * direction[0] ** ax
                * direction[1] ** ay
                * direction[2] ** az
                * central_moment(moments, alpha)
            )
    return float(value / rho)


def fit_equal_variance_marginal(
    second: float,
    third: float,
    fourth: float,
    *,
    gaussian_tolerance: float = 5.0e-13,
) -> MarginalMixture:
    """Fit a one- or two-component equal-variance Gaussian marginal.

    The returned component means are centered.  For states lying on the
    equal-variance Pearson manifold, moments two through four are reproduced
    to roundoff.  A nearly Gaussian marginal collapses to one component.
    """

    if not np.all(np.isfinite([second, third, fourth])) or second <= 0.0:
        raise ValueError("marginal moments must be finite with positive variance")

    kappa3 = float(third)
    kappa4 = float(fourth - 3.0 * second**2)
    scale = max(second, 1.0)
    if abs(kappa3) <= gaussian_tolerance * scale**1.5 and abs(kappa4) <= (
        gaussian_tolerance * scale**2
    ):
        return MarginalMixture(
            weights=np.asarray([1.0]),
            means=np.asarray([0.0]),
            variance=float(second),
            component_variances=np.asarray([float(second)]),
            between_variance=0.0,
            reconstruction_error=0.0,
            branch="gaussian",
        )

    roots = np.roots([2.0, 0.0, kappa4, -(kappa3**2)])
    admissible = sorted(
        float(root.real)
        for root in roots
        if abs(root.imag) <= 5.0e-10 * max(1.0, abs(root.real))
        and root.real >= -5.0e-12 * scale
        and root.real <= second + 5.0e-10 * scale
    )
    if not admissible:
        raise ValueError(
            "marginal is outside the admissible equal-variance two-Gaussian family"
        )
    between = min(max(admissible[-1], 0.0), second)
    if between <= gaussian_tolerance * scale:
        # A symmetric leptokurtic marginal has no nonzero equal-variance
        # two-Gaussian root.  Treat it as Gaussian and expose the mismatch in
        # reconstruction_error instead of inventing a nonphysical component.
        error = abs(fourth - 3.0 * second**2) / max(abs(fourth), second**2)
        return MarginalMixture(
            weights=np.asarray([1.0]),
            means=np.asarray([0.0]),
            variance=float(second),
            component_variances=np.asarray([float(second)]),
            between_variance=0.0,
            reconstruction_error=float(error),
            branch="gaussian-fallback",
        )

    denominator = kappa3**2 + 4.0 * between**3
    pq = between**3 / denominator
    pq = float(np.clip(pq, 1.0e-15, 0.25))
    difference = kappa3 * np.sqrt(pq) / between**1.5
    difference = float(np.clip(difference, -1.0, 1.0))
    p = 0.5 * (1.0 - difference)
    q = 1.0 - p
    if min(p, q) <= 1.0e-12:
        raise ValueError("degenerate Gaussian-mixture weight")
    separation = np.sqrt(between / (p * q))
    means = np.asarray([q * separation, -p * separation])
    variance = max(float(second - between), 0.0)

    reconstructed_second = variance + p * means[0] ** 2 + q * means[1] ** 2
    reconstructed_third = p * means[0] ** 3 + q * means[1] ** 3
    reconstructed_fourth = (
        p * (means[0] ** 4 + 6.0 * means[0] ** 2 * variance + 3.0 * variance**2)
        + q
        * (means[1] ** 4 + 6.0 * means[1] ** 2 * variance + 3.0 * variance**2)
    )
    error = np.linalg.norm(
        [
            reconstructed_second - second,
            reconstructed_third - third,
            reconstructed_fourth - fourth,
        ]
    ) / max(np.linalg.norm([second, third, fourth]), 1.0e-15)
    return MarginalMixture(
        weights=np.asarray([p, q]),
        means=means,
        variance=variance,
        component_variances=np.asarray([variance, variance]),
        between_variance=between,
        reconstruction_error=float(error),
        branch="equal-variance-eqmom",
    )


def fit_location_scale_marginal(
    second: float,
    third: float,
    fourth: float,
    *,
    gaussian_tolerance: float = 5.0e-13,
) -> MarginalMixture:
    """Fit a finite unequal-variance mixture on the leptokurtic branch.

    For positive fourth cumulant and moderate skewness, equal weights are
    sufficient.  Writing the component means as ``+a`` and ``-a`` and their
    variances as ``S+h`` and ``S-h`` gives

    ``2 (a^2)^3 + kappa4 a^2 - kappa3^2/3 = 0``.

    The formula has the bounded symmetric limit
    ``a -> 0`` and ``h -> sign(kappa3)*sqrt(kappa4/3)``.  At exactly zero
    skewness the two components share the mean but have different widths.
    Very large symmetric kurtosis is covered with unequal weights and zero
    mean separation, which can represent any finite positive fourth
    cumulant without a distant vanishing-weight component.
    """

    if not np.all(np.isfinite([second, third, fourth])) or second <= 0.0:
        raise ValueError("marginal moments must be finite with positive variance")
    kappa3 = float(third)
    kappa4 = float(fourth - 3.0 * second**2)
    scale = max(second, 1.0)
    if kappa4 <= gaussian_tolerance * scale**2:
        return fit_equal_variance_marginal(
            second, third, fourth, gaussian_tolerance=gaussian_tolerance
        )

    # The equal-weight location--scale branch is analytic and continuous at
    # kappa3=0 whenever both component variances remain nonnegative.
    if abs(kappa3) <= gaussian_tolerance * scale**1.5:
        half_difference = np.sqrt(kappa4 / 3.0)
        if half_difference <= second * (1.0 + 5.0e-12):
            component_variances = np.maximum(
                np.asarray([second + half_difference, second - half_difference]),
                0.0,
            )
            return MarginalMixture(
                weights=np.asarray([0.5, 0.5]),
                means=np.asarray([0.0, 0.0]),
                variance=float(second),
                component_variances=component_variances,
                between_variance=0.0,
                reconstruction_error=0.0,
                branch="symmetric-location-scale",
            )

        # For kurtosis above six, choose a lighter broad component.  With
        # zero component means the fourth cumulant is 3*p*q*(s1-s2)^2.
        p_limit = 3.0 * second**2 / fourth
        p = 2.0 * p_limit**2
        p = float(np.clip(p, 1.0e-12, 0.5))
        q = 1.0 - p
        difference = np.sqrt(kappa4 / (3.0 * p * q))
        variances = np.asarray(
            [second + q * difference, second - p * difference]
        )
        if np.min(variances) < -5.0e-11 * scale:
            raise ValueError("no positive symmetric location-scale fit")
        variances = np.maximum(variances, 0.0)
        return MarginalMixture(
            weights=np.asarray([p, q]),
            means=np.asarray([0.0, 0.0]),
            variance=float(np.dot([p, q], variances)),
            component_variances=variances,
            between_variance=0.0,
            reconstruction_error=0.0,
            branch="symmetric-location-scale-high-kurtosis",
        )

    if kappa4 > 3.0 * second**2 * (1.0 - 5.0e-10):
        return _fit_unequal_weight_location_scale(second, third, fourth)

    roots = np.roots([2.0, 0.0, kappa4, -(kappa3**2) / 3.0])
    candidates = sorted(
        float(root.real)
        for root in roots
        if abs(root.imag) <= 5.0e-10 * max(1.0, abs(root.real))
        and root.real > 0.0
        and root.real < second * (1.0 + 5.0e-10)
    )
    if not candidates:
        return _fit_unequal_weight_location_scale(second, third, fourth)
    between = candidates[0]
    mean = np.sqrt(between)
    half_difference = kappa3 / (3.0 * mean)
    average_variance = second - between
    variances = np.asarray(
        [average_variance + half_difference, average_variance - half_difference]
    )
    if np.min(variances) < -5.0e-10 * scale:
        return _fit_unequal_weight_location_scale(second, third, fourth)
    variances = np.maximum(variances, 0.0)
    weights = np.asarray([0.5, 0.5])
    means = np.asarray([mean, -mean])
    reconstructed_second = float(
        np.dot(weights, variances + means**2)
    )
    reconstructed_third = float(
        np.dot(weights, means**3 + 3.0 * means * variances)
    )
    reconstructed_fourth = float(
        np.dot(weights, means**4 + 6.0 * means**2 * variances + 3.0 * variances**2)
    )
    error = np.linalg.norm(
        [
            reconstructed_second - second,
            reconstructed_third - third,
            reconstructed_fourth - fourth,
        ]
    ) / max(np.linalg.norm([second, third, fourth]), 1.0e-15)
    return MarginalMixture(
        weights=weights,
        means=means,
        variance=float(np.dot(weights, variances)),
        component_variances=variances,
        between_variance=float(between),
        reconstruction_error=float(error),
        branch="location-scale",
    )


def _fit_unequal_weight_location_scale(
    second: float,
    third: float,
    fourth: float,
) -> MarginalMixture:
    """Positive two-Gaussian fit with unequal weights, means, and variances.

    For a fixed component weight ``p`` and separation ``d``, the mean-zero,
    second-, and third-moment constraints determine both component variances.
    A bounded one-dimensional search then enforces the fourth moment.  Among
    admissible roots we maximize the smaller component variance, followed by
    the smaller weight, avoiding the distant vanishing-weight Pearson branch.
    """

    scale = max(second, 1.0)

    def candidate_for(p: float, separation: float):
        q = 1.0 - p
        pq = p * q
        if pq <= 0.0 or separation <= 0.0:
            return None
        average_variance = second - pq * separation**2
        variance_difference = (
            third / (pq * separation) - (q - p) * separation**2
        ) / 3.0
        variances = np.asarray(
            [
                average_variance + q * variance_difference,
                average_variance - p * variance_difference,
            ]
        )
        means = np.asarray([q * separation, -p * separation])
        predicted_fourth = float(
            np.dot(
                [p, q],
                means**4 + 6.0 * means**2 * variances + 3.0 * variances**2,
            )
        )
        return predicted_fourth - fourth, means, variances

    candidates: list[
        tuple[tuple[float, float, float, float], float, np.ndarray, np.ndarray]
    ] = []
    weight_grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                1.0 / (1.0 + np.exp(-np.linspace(-7.0, 7.0, 81))),
            ]
        )
    )
    for p in weight_grid:
        q = 1.0 - p
        maximum_separation = np.sqrt(second / (p * q)) * (1.0 - 1.0e-10)
        minimum_separation = maximum_separation * 1.0e-7
        separation_grid = np.geomspace(minimum_separation, maximum_separation, 180)
        previous = candidate_for(float(p), float(separation_grid[0]))
        for right in separation_grid[1:]:
            current = candidate_for(float(p), float(right))
            if previous is None or current is None:
                previous = current
                continue
            left = float(right / (separation_grid[1] / separation_grid[0]))
            left_value = previous[0]
            right_value = current[0]
            if left_value == 0.0 or left_value * right_value <= 0.0:
                for _ in range(55):
                    midpoint = 0.5 * (left + right)
                    middle = candidate_for(float(p), midpoint)
                    if middle is None:
                        break
                    if left_value * middle[0] <= 0.0:
                        right = midpoint
                        right_value = middle[0]
                    else:
                        left = midpoint
                        left_value = middle[0]
                root = candidate_for(float(p), 0.5 * (left + right))
                if root is not None:
                    _, means, variances = root
                    if np.min(variances) >= -2.0e-9 * scale:
                        variances = np.maximum(variances, 0.0)
                        score = (
                            float(
                                np.min(variances) / second * min(p, q)
                            ),
                            float(min(p, q)),
                            float(np.min(variances) / second),
                            float(-np.ptp(means) / np.sqrt(second)),
                        )
                        candidates.append((score, float(p), means, variances))
            previous = current

    if not candidates:
        raise ValueError("no positive unequal-weight location-scale fit")
    _, p, means, variances = max(candidates, key=lambda item: item[0])
    weights = np.asarray([p, 1.0 - p])
    reconstructed = np.asarray(
        [
            np.dot(weights, variances + means**2),
            np.dot(weights, means**3 + 3.0 * means * variances),
            np.dot(weights, means**4 + 6.0 * means**2 * variances + 3.0 * variances**2),
        ]
    )
    error = float(
        np.linalg.norm(reconstructed - [second, third, fourth])
        / max(np.linalg.norm([second, third, fourth]), 1.0e-15)
    )
    return MarginalMixture(
        weights=weights,
        means=means,
        variance=float(np.dot(weights, variances)),
        component_variances=variances,
        between_variance=float(np.dot(weights, means**2)),
        reconstruction_error=error,
        branch="unequal-weight-location-scale",
    )


def _moments_from_weighted_nodes(weights: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    powers = [
        np.vstack([np.ones(nodes.shape[0]), *(nodes[:, d] ** n for n in range(1, 5))])
        for d in range(3)
    ]
    return np.asarray(
        [
            np.dot(weights, powers[0][i] * powers[1][j] * powers[2][k])
            for i, j, k in HYQMOM_35_INDICES
        ],
        dtype=float,
    )


def reconstruct_gaussian_mixture_quadrature(
    moments: Sequence[float],
    quadrature_order: int = 4,
    *,
    force_single_gaussian: bool = False,
) -> MixtureQuadrature:
    """Reconstruct a principal-axis Gaussian mixture and tensor quadrature."""

    if quadrature_order < 4:
        raise ValueError("quadrature_order must be at least four")
    vector = np.asarray(moments, dtype=float)
    if vector.shape != (35,) or not np.all(np.isfinite(vector)):
        raise ValueError("expected a finite 35-moment vector")
    state = macroscopic_state(vector)
    eigenvalues, eigenvectors = np.linalg.eigh(state.covariance)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("covariance must be positive definite")

    marginals: list[MarginalMixture] = []
    for axis in range(3):
        direction = eigenvectors[:, axis]
        second = _contracted_central_moment(vector, direction, 2)
        third = _contracted_central_moment(vector, direction, 3)
        fourth = _contracted_central_moment(vector, direction, 4)
        if force_single_gaussian:
            marginal_error = np.linalg.norm(
                [third, fourth - 3.0 * second**2]
            ) / max(np.linalg.norm([second, third, fourth]), 1.0e-15)
            marginals.append(
                MarginalMixture(
                    weights=np.asarray([1.0]),
                    means=np.asarray([0.0]),
                    variance=float(second),
                    component_variances=np.asarray([float(second)]),
                    between_variance=0.0,
                    reconstruction_error=float(marginal_error),
                    branch="forced-single-gaussian",
                )
            )
        else:
            kappa4 = fourth - 3.0 * second**2
            if kappa4 > 5.0e-13 * max(second, 1.0) ** 2:
                marginals.append(fit_location_scale_marginal(second, third, fourth))
            else:
                marginals.append(fit_equal_variance_marginal(second, third, fourth))

    hermite_nodes, hermite_weights = np.polynomial.hermite.hermgauss(quadrature_order)
    axis_nodes: list[np.ndarray] = []
    axis_weights: list[np.ndarray] = []
    for marginal in marginals:
        nodes_for_axis = []
        weights_for_axis = []
        for component_weight, component_mean, component_variance in zip(
            marginal.weights, marginal.means, marginal.component_variances
        ):
            sigma_scale = np.sqrt(2.0 * component_variance)
            nodes_for_axis.extend(component_mean + sigma_scale * hermite_nodes)
            weights_for_axis.extend(
                component_weight * hermite_weights / np.sqrt(np.pi)
            )
        axis_nodes.append(np.asarray(nodes_for_axis, dtype=float))
        axis_weights.append(np.asarray(weights_for_axis, dtype=float))

    grid_indices = list(
        product(
            range(axis_nodes[0].size),
            range(axis_nodes[1].size),
            range(axis_nodes[2].size),
        )
    )
    principal_nodes = np.asarray(
        [
            [axis_nodes[direction][index[direction]] for direction in range(3)]
            for index in grid_indices
        ],
        dtype=float,
    )
    probability_weights = np.asarray(
        [
            np.prod(
                [axis_weights[direction][index[direction]] for direction in range(3)]
            )
            for index in grid_indices
        ],
        dtype=float,
    )
    nodes = state.velocity + principal_nodes @ eigenvectors.T
    weights = state.rho * probability_weights
    reconstructed = _moments_from_weighted_nodes(weights, nodes)
    residual = float(
        np.linalg.norm(reconstructed - vector) / max(np.linalg.norm(vector), 1.0e-15)
    )
    return MixtureQuadrature(
        weights=weights,
        nodes=nodes,
        reconstructed_moments=reconstructed,
        eigenvectors=eigenvectors,
        marginals=tuple(marginals),
        relative_moment_residual=residual,
    )


def _moments_of_isotropic_gaussian_mixture(
    weights: np.ndarray, means: np.ndarray, variance: float
) -> np.ndarray:
    """Accumulate raw moments through degree four for equal isotropic widths."""

    one_dimensional = []
    for direction in range(3):
        mean = means[:, direction]
        one_dimensional.append(
            np.vstack(
                [
                    np.ones_like(mean),
                    mean,
                    mean**2 + variance,
                    mean**3 + 3.0 * mean * variance,
                    mean**4 + 6.0 * mean**2 * variance + 3.0 * variance**2,
                ]
            )
        )
    return np.asarray(
        [
            np.dot(
                weights,
                one_dimensional[0][i]
                * one_dimensional[1][j]
                * one_dimensional[2][k],
            )
            for i, j, k in HYQMOM_35_INDICES
        ]
    )


def _centered_gaussian_moment(index: tuple[int, int, int], variance: float) -> float:
    """Moment of an isotropic zero-mean Gaussian with scalar variance."""

    value = 1.0
    for exponent in index:
        if exponent % 2:
            return 0.0
        for factor in range(1, exponent, 2):
            value *= factor * variance
    return float(value)


def _exact_ou_moment_map(
    moments: Sequence[float], relaxation: float, theta: float
) -> np.ndarray:
    """Apply the exact OU Gaussian-convolution map through total degree four.

    This is used for the moment residual that a principal-axis tensor-product
    mixture cannot represent.  Because the input and reconstructed states have
    the same collision invariants, the correction decays as the appropriate
    Hermite modes instead of remaining frozen by residual cancellation.
    """

    vector = np.asarray(moments, dtype=float)
    state = macroscopic_state(vector)
    noise_variance = theta * (1.0 - relaxation**2)
    central_after: dict[tuple[int, int, int], float] = {}
    for alpha in HYQMOM_35_INDICES:
        value = 0.0
        for bx in range(alpha[0] + 1):
            for by in range(alpha[1] + 1):
                for bz in range(alpha[2] + 1):
                    beta = (bx, by, bz)
                    noise_index = tuple(alpha[d] - beta[d] for d in range(3))
                    coefficient = np.prod(
                        [comb(alpha[d], beta[d]) for d in range(3)]
                    )
                    value += (
                        coefficient
                        * relaxation ** sum(beta)
                        * central_moment(vector, beta)
                        * _centered_gaussian_moment(noise_index, noise_variance)
                    )
        central_after[alpha] = float(value)

    mapped = np.zeros(35)
    for position, alpha in enumerate(HYQMOM_35_INDICES):
        value = 0.0
        for bx in range(alpha[0] + 1):
            for by in range(alpha[1] + 1):
                for bz in range(alpha[2] + 1):
                    beta = (bx, by, bz)
                    coefficient = np.prod(
                        [comb(alpha[d], beta[d]) for d in range(3)]
                    )
                    mean_factor = np.prod(
                        [
                            state.velocity[d] ** (alpha[d] - beta[d])
                            for d in range(3)
                        ]
                    )
                    value += coefficient * mean_factor * central_after[beta]
        mapped[position] = value
    return mapped


def exact_ou_moment_map(
    moments: Sequence[float], relaxation: float, theta: float
) -> np.ndarray:
    """Public exact Ornstein--Uhlenbeck map through total degree four."""

    return _exact_ou_moment_map(moments, relaxation, theta)


_MOMENT_MATRIX_BASIS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (2, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (0, 2, 0),
    (0, 1, 1),
    (0, 0, 2),
)


def realizability_margin_35(moments: Sequence[float]) -> float:
    """Return the smallest normalized eigenvalue of the degree-two moment matrix."""

    vector = np.asarray(moments, dtype=float)
    matrix = np.empty((len(_MOMENT_MATRIX_BASIS),) * 2)
    for row, left in enumerate(_MOMENT_MATRIX_BASIS):
        for column, right in enumerate(_MOMENT_MATRIX_BASIS):
            index = tuple(left[d] + right[d] for d in range(3))
            matrix[row, column] = vector[POSITION[index]]
    scale = max(float(np.trace(matrix)) / matrix.shape[0], 1.0e-15)
    return float(np.linalg.eigvalsh(matrix)[0] / scale)


def finite_gaussian_mixture_fp_step(
    moments: Sequence[float],
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
    gamma_scale: float = 0.05,
    speed_cap: float = 25.0,
    residual_cancel: bool = True,
    quadrature_order: int = 4,
    force_single_gaussian: bool = False,
) -> tuple[np.ndarray, FiniteMixtureStepDiagnostics]:
    """Advance one collision interval with the finite-width mixture map.

    By default the reconstructed mixture receives the nonlinear cubic-FP map,
    while the unresolved moment residual receives the exact linear OU map.
    This hybrid correction prevents both accumulation and freezing of rotated
    cross moments and becomes asymptotically exact near Maxwellian equilibrium.
    """

    if dt <= 0.0 or tau <= 0.0:
        raise ValueError("dt and tau must be positive")
    if not 0.0 < prandtl <= 1.0 or speed_cap <= 0.0:
        raise ValueError("invalid Prandtl number or speed cap")
    incoming = np.asarray(moments, dtype=float)
    state = macroscopic_state(incoming)
    quadrature = reconstruct_gaussian_mixture_quadrature(
        incoming,
        quadrature_order=quadrature_order,
        force_single_gaussian=force_single_gaussian,
    )
    coefficients = coefficients_from_weighted_nodes(
        quadrature.nodes,
        quadrature.weights,
        tau=tau,
        prandtl=prandtl,
    )
    probabilities = quadrature.weights / state.rho
    peculiar = quadrature.nodes - state.velocity
    c2 = np.einsum("ni,ni->n", peculiar, peculiar)
    c2_used = np.minimum(c2, speed_cap * state.theta)
    nonlinear = peculiar @ coefficients.C.T
    nonlinear += (c2_used - 3.0 * state.theta)[:, None] * coefficients.gamma
    nonlinear += coefficients.beta * (
        c2_used[:, None] * peculiar
        - 2.0 * state.heat_flux[None, :] / state.rho
    )

    mean_n2 = float(np.dot(probabilities, np.einsum("ni,ni->n", nonlinear, nonlinear)))
    mean_cn = float(np.dot(probabilities, np.einsum("ni,ni->n", peculiar, nonlinear)))
    relaxation = float(np.exp(-dt / tau))
    relaxation2 = relaxation**2
    alpha_squared = 1.0 + tau / (3.0 * state.theta) * (
        tau * (1.0 - relaxation) ** 2 * mean_n2
        + 2.0 * (relaxation - relaxation2) * mean_cn
    )
    alpha = float(np.sqrt(alpha_squared)) if alpha_squared > 1.0e-6 else 1.0
    noise_variance = state.theta * (1.0 - relaxation2) / alpha**2
    mapped_means = state.velocity + (
        relaxation * peculiar + (1.0 - relaxation) * tau * nonlinear
    ) / alpha

    mapped_velocity = np.sum(probabilities[:, None] * mapped_means, axis=0)
    centered_means = mapped_means - mapped_velocity
    mapped_theta = float(
        (np.dot(probabilities, np.einsum("ni,ni->n", centered_means, centered_means))
         + 3.0 * noise_variance)
        / 3.0
    )
    if mapped_theta <= 0.0:
        raise FloatingPointError("finite mixture map produced nonpositive temperature")
    affine_scale = float(np.sqrt(state.theta / mapped_theta))
    corrected_means = state.velocity + affine_scale * centered_means
    corrected_variance = affine_scale**2 * noise_variance
    mapped = _moments_of_isotropic_gaussian_mixture(
        quadrature.weights, corrected_means, corrected_variance
    )
    if residual_cancel:
        residual_after = _exact_ou_moment_map(
            incoming, relaxation, state.theta
        ) - _exact_ou_moment_map(
            quadrature.reconstructed_moments, relaxation, state.theta
        )
        updated = mapped + residual_after
    else:
        updated = mapped

    # Pin exact collision invariants after the affine transformation.
    for index in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        updated[POSITION[index]] = incoming[POSITION[index]]
    energy_indices = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    energy_error = sum(incoming[POSITION[index]] for index in energy_indices) - sum(
        updated[POSITION[index]] for index in energy_indices
    )
    for index in energy_indices:
        updated[POSITION[index]] += energy_error / 3.0
    if not np.all(np.isfinite(updated)):
        raise FloatingPointError("finite mixture map produced NaN or infinity")
    margin = realizability_margin_35(updated)
    diagnostics = FiniteMixtureStepDiagnostics(
        alpha=alpha,
        alpha_squared=float(alpha_squared),
        affine_scale=affine_scale,
        maximum_c2_over_theta=float(np.max(c2) / state.theta),
        quadrature_nodes=int(quadrature.weights.size),
        reconstruction_relative_residual=quadrature.relative_moment_residual,
        increment_norm=float(np.linalg.norm(updated - incoming)),
        realizability_margin=margin,
    )
    return updated, diagnostics

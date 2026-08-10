"""Positive low-discrepancy kinetic reference for homogeneous cubic FP.

The reference is a weighted positive discrete measure.  Gaussian-mixture
initial data and every Brownian increment are generated with scrambled Sobol
points.  Component weights are retained exactly, which is important for rare
hot and rare beam populations.  Affine moment matching removes the finite
cubature error in the mean and covariance of each Gaussian rule; convergence
of higher moments must still be demonstrated by increasing the number of
points and reducing the time step.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import norm, qmc

from .collision import coefficients_from_weighted_nodes
from .particle_reference import (
    moments_35_from_particles,
    particle_macroscopic_state,
)


@dataclass(frozen=True)
class QMCStepDiagnostics:
    """Finite-step diagnostics for the weighted low-discrepancy ensemble."""

    alpha: float
    momentum_drift: float
    energy_drift: float


def _require_power_of_two(value: int) -> int:
    if value < 8 or value & (value - 1):
        raise ValueError("Sobol point count must be a power of two and at least 8")
    return int(round(log2(value)))


def _standard_normal_sobol(points: int, seed: int) -> np.ndarray:
    exponent = _require_power_of_two(points)
    uniform = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(exponent)
    epsilon = np.finfo(float).eps
    return norm.ppf(np.clip(uniform, epsilon, 1.0 - epsilon))


def _affine_standardize(
    values: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> np.ndarray:
    """Set the weighted sample mean to zero and covariance to identity."""

    if probabilities is None:
        probabilities = np.full(values.shape[0], 1.0 / values.shape[0])
    else:
        probabilities = np.asarray(probabilities, dtype=float)
        probabilities = probabilities / np.sum(probabilities)
    centered = values - np.sum(probabilities[:, None] * values, axis=0)
    covariance = np.einsum(
        "n,ni,nj->ij", probabilities, centered, centered, optimize=True
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) <= 1.0e-14:
        raise FloatingPointError("low-discrepancy normal rule is singular")
    inverse_root = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    return centered @ inverse_root


def sample_gaussian_mixture_qmc(
    components: Iterable[
        tuple[float, Sequence[float], Sequence[Sequence[float]]]
    ],
    *,
    points_per_component: int,
    seed: int = 20_260_810,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a positive weighted Sobol cubature of a Gaussian mixture."""

    exponent = _require_power_of_two(points_per_component)
    del exponent
    component_list = list(components)
    if not component_list:
        raise ValueError("at least one Gaussian component is required")
    mixture_weights = np.asarray([item[0] for item in component_list], dtype=float)
    if np.any(mixture_weights < 0.0) or np.sum(mixture_weights) <= 0.0:
        raise ValueError("mixture weights must be nonnegative with positive sum")
    mixture_weights /= np.sum(mixture_weights)
    nodes = []
    weights = []
    for component, (probability, (_, mean, covariance)) in enumerate(
        zip(mixture_weights, component_list)
    ):
        standard = _standard_normal_sobol(
            points_per_component, seed + 104_729 * component
        )
        standard = _affine_standardize(standard)
        covariance_array = np.asarray(covariance, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_array)
        if np.min(eigenvalues) <= 0.0:
            raise ValueError("component covariance must be positive definite")
        root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        nodes.append(np.asarray(mean, dtype=float) + standard @ root)
        weights.append(
            np.full(points_per_component, probability / points_per_component)
        )
    return np.vstack(nodes), np.concatenate(weights)


def qmc_cubic_fp_step(
    velocities: Sequence[Sequence[float]],
    weights: Sequence[float],
    *,
    dt: float,
    tau: float,
    seed: int,
    prandtl: float = 2.0 / 3.0,
    enforce_discrete_invariants: bool = True,
) -> tuple[np.ndarray, QMCStepDiagnostics]:
    """Advance one positive weighted ensemble step with Sobol OU noise."""

    if dt <= 0.0 or tau <= 0.0:
        raise ValueError("dt and tau must be positive")
    nodes = np.asarray(velocities, dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("velocities must have shape (n, 3)")
    if probabilities.shape != (nodes.shape[0],) or np.any(probabilities < 0.0):
        raise ValueError("weights must be nonnegative with one entry per node")
    probabilities = probabilities / np.sum(probabilities)
    before = particle_macroscopic_state(nodes, probabilities)
    coefficients = coefficients_from_weighted_nodes(
        nodes, probabilities, tau=tau, prandtl=prandtl
    )
    peculiar = nodes - before.velocity
    c2 = np.einsum("ni,ni->n", peculiar, peculiar)
    nonlinear = peculiar @ coefficients.C.T
    nonlinear += (c2 - 3.0 * before.theta)[:, None] * coefficients.gamma
    nonlinear += coefficients.beta * (
        c2[:, None] * peculiar
        - 2.0 * before.heat_flux[None, :] / before.rho
    )

    decay = float(np.exp(-dt / tau))
    decay2 = decay * decay
    mean_n2 = float(
        np.dot(probabilities, np.einsum("ni,ni->n", nonlinear, nonlinear))
    )
    mean_cn = float(
        np.dot(probabilities, np.einsum("ni,ni->n", peculiar, nonlinear))
    )
    alpha_squared = 1.0 + tau / (3.0 * before.theta) * (
        tau * (1.0 - decay) ** 2 * mean_n2
        + 2.0 * (decay - decay2) * mean_cn
    )
    alpha = float(np.sqrt(alpha_squared)) if alpha_squared > 1.0e-6 else 1.0

    noise = _standard_normal_sobol(nodes.shape[0], seed)
    # A Sobol rule is an unordered cubature set.  Assigning its native row
    # order repeatedly to a persistent velocity array can create an artificial
    # velocity--noise correlation.  A deterministic seeded permutation keeps
    # the same low-discrepancy set while making that assignment exchangeable.
    permutation = np.random.default_rng(seed + 32_452_843).permutation(
        nodes.shape[0]
    )
    noise = noise[permutation]
    # The two component blocks carry unequal node weights.  Standardizing with
    # those weights keeps the finite cubature noise centered and isotropic.
    noise = _affine_standardize(noise, probabilities)
    sigma = float(np.sqrt(before.theta * (1.0 - decay2)))
    peculiar_new = (
        decay * peculiar + (1.0 - decay) * tau * nonlinear + sigma * noise
    ) / alpha

    if enforce_discrete_invariants:
        peculiar_new -= np.sum(
            probabilities[:, None] * peculiar_new, axis=0
        )
        old_energy = float(np.dot(probabilities, c2))
        new_energy = float(
            np.dot(
                probabilities,
                np.einsum("ni,ni->n", peculiar_new, peculiar_new),
            )
        )
        if new_energy <= 0.0:
            raise FloatingPointError("QMC ensemble collapsed its kinetic energy")
        peculiar_new *= np.sqrt(old_energy / new_energy)

    updated = before.velocity + peculiar_new
    after = particle_macroscopic_state(updated, probabilities)
    momentum_drift = float(
        np.linalg.norm(after.rho * after.velocity - before.rho * before.velocity)
    )
    energy_before = before.rho * (
        np.dot(before.velocity, before.velocity) + 3.0 * before.theta
    )
    energy_after = after.rho * (
        np.dot(after.velocity, after.velocity) + 3.0 * after.theta
    )
    return updated, QMCStepDiagnostics(
        alpha=alpha,
        momentum_drift=momentum_drift,
        energy_drift=float(energy_after - energy_before),
    )


def moments_35_from_qmc(
    velocities: Sequence[Sequence[float]], weights: Sequence[float]
) -> np.ndarray:
    """Convenience wrapper documenting that the discrete measure is positive."""

    return moments_35_from_particles(velocities, weights)

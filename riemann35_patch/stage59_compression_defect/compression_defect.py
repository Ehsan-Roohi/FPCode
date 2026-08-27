"""Measure third-order information discarded by Stage-57 Gaussian recompression."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from hyqmom_fp import macroscopic_state
from hyqmom_fp.collision import coefficients_from_weighted_nodes
from hyqmom_fp.two_population import _gauss_hermite_mixture_nodes
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import (
    PersistentGaussianMixtureState,
    persistent_gaussian_mixture_moments,
)


@dataclass(frozen=True)
class CompressionDefect:
    total: float
    trace_free: float
    heat_flux: float
    max_population: float
    affine_scale: float


def _trace_carrying_tensor(heat_flux: np.ndarray) -> np.ndarray:
    trace = 2.0 * np.asarray(heat_flux, dtype=float)
    eye = np.eye(3)
    return (
        np.einsum("ij,k->ijk", eye, trace)
        + np.einsum("ik,j->ijk", eye, trace)
        + np.einsum("jk,i->ijk", eye, trace)
    ) / 5.0


def compression_defect(
    populations: PersistentGaussianMixtureState,
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
    quadrature_nodes: int = 5,
) -> CompressionDefect:
    """Return the normalized local-third tensor erased by one Gaussian compression.

    Stage 57 maps a positive Gauss-Hermite cloud for each labelled population and
    then replaces that cloud by one Gaussian with identical mean/covariance.
    The replacement therefore erases every within-population third central moment.
    This routine evaluates that erased tensor immediately before compression.
    Additive isotropic Gaussian noise has zero third central moment, so it does not
    alter the quantity below.  The subsequent global affine conservation correction
    scales third moments by ``affine_scale**3`` and is included explicitly.
    """
    if dt <= 0.0 or tau <= 0.0:
        raise ValueError("dt and tau must be positive")
    if quadrature_nodes < 4:
        raise ValueError("quadrature_nodes must be at least four")

    incoming = persistent_gaussian_mixture_moments(populations)
    macro = macroscopic_state(incoming)
    weights, nodes = _gauss_hermite_mixture_nodes(
        populations.probabilities,
        populations.means,
        populations.covariances,
        populations.rho,
        quadrature_nodes,
    )
    coefficients = coefficients_from_weighted_nodes(
        nodes, weights, tau=tau, prandtl=prandtl
    )
    probabilities = weights / populations.rho
    peculiar = nodes - macro.velocity
    c2 = np.einsum("ni,ni->n", peculiar, peculiar)
    nonlinear = peculiar @ coefficients.C.T
    nonlinear += (c2 - 3.0 * macro.theta)[:, None] * coefficients.gamma
    nonlinear += coefficients.beta * (
        c2[:, None] * peculiar
        - 2.0 * macro.heat_flux[None, :] / populations.rho
    )
    mean_n2 = float(np.dot(probabilities, np.einsum("ni,ni->n", nonlinear, nonlinear)))
    mean_cn = float(np.dot(probabilities, np.einsum("ni,ni->n", peculiar, nonlinear)))
    relaxation = float(np.exp(-dt / tau))
    relaxation_squared = relaxation**2
    alpha_squared = 1.0 + tau / (3.0 * macro.theta) * (
        tau * (1.0 - relaxation) ** 2 * mean_n2
        + 2.0 * (relaxation - relaxation_squared) * mean_cn
    )
    if alpha_squared <= 1.0e-12:
        raise FloatingPointError("nonpositive alpha squared in compression audit")
    alpha = float(np.sqrt(alpha_squared))
    noise_variance = macro.theta * (1.0 - relaxation_squared) / alpha**2
    mapped_nodes = macro.velocity + (
        relaxation * peculiar + (1.0 - relaxation) * tau * nonlinear
    ) / alpha

    nodes_per_population = quadrature_nodes**3
    mapped_means = []
    mapped_covariances = []
    local_thirds = []
    for component, component_probability in enumerate(populations.probabilities):
        block = slice(component * nodes_per_population, (component + 1) * nodes_per_population)
        local_prob = probabilities[block] / component_probability
        local_nodes = mapped_nodes[block]
        local_mean = np.sum(local_prob[:, None] * local_nodes, axis=0)
        centered = local_nodes - local_mean
        covariance = np.einsum("n,ni,nj->ij", local_prob, centered, centered)
        covariance += noise_variance * np.eye(3)
        local_third = np.einsum("n,ni,nj,nk->ijk", local_prob, centered, centered, centered)
        mapped_means.append(local_mean)
        mapped_covariances.append(covariance)
        local_thirds.append(local_third)

    mapped_means = np.asarray(mapped_means)
    mapped_covariances = np.asarray(mapped_covariances)
    local_thirds = np.asarray(local_thirds)
    mixture_mean = np.sum(populations.probabilities[:, None] * mapped_means, axis=0)
    mixture_covariance = np.zeros((3, 3))
    for probability, mean, covariance in zip(populations.probabilities, mapped_means, mapped_covariances):
        offset = mean - mixture_mean
        mixture_covariance += probability * (covariance + np.outer(offset, offset))
    mapped_theta = float(np.trace(mixture_covariance) / 3.0)
    if mapped_theta <= 0.0:
        raise FloatingPointError("nonpositive mapped temperature in compression audit")
    affine_scale = float(np.sqrt(macro.theta / mapped_theta))

    scaled_local = affine_scale**3 * local_thirds
    discarded = populations.rho * np.einsum("p,pijk->ijk", populations.probabilities, scaled_local)
    discarded_heat_flux = 0.5 * np.einsum("ijj->i", discarded)
    discarded_trace_free = discarded - _trace_carrying_tensor(discarded_heat_flux)
    natural_third_scale = max(populations.rho * macro.theta**1.5, 1.0e-14)
    population_norms = np.linalg.norm(scaled_local.reshape(scaled_local.shape[0], -1), axis=1)
    return CompressionDefect(
        total=float(np.linalg.norm(discarded) / natural_third_scale),
        trace_free=float(np.linalg.norm(discarded_trace_free) / natural_third_scale),
        heat_flux=float(np.linalg.norm(discarded_heat_flux) / natural_third_scale),
        max_population=float(np.max(population_norms) / max(macro.theta**1.5, 1.0e-14)),
        affine_scale=affine_scale,
    )

"""Homogeneous particle reference for the cubic Fokker--Planck collision model.

This module is intentionally CPU/NumPy-only.  It mirrors the physics path in
``FP_PINN/legacy_source/147CylFP.py`` without the spatial mesh or CUDA runtime,
so the projected 35-moment source can be checked against the actual stochastic
particle update from the same initial velocity ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .collision import CubicFPCoefficients, coefficients_from_weighted_nodes
from .moments import HYQMOM_35_INDICES


@dataclass(frozen=True)
class ParticleMacroscopicState:
    """Macroscopic quantities evaluated directly from weighted particles."""

    rho: float
    velocity: np.ndarray
    theta: float
    covariance: np.ndarray
    pressure: float
    stress: np.ndarray
    heat_flux: np.ndarray


@dataclass(frozen=True)
class ParticleStepDiagnostics:
    """Finite-step quantities used by FPCode's stochastic update."""

    alpha: float
    mass_drift: float
    momentum_drift: float
    energy_drift: float


def _normalized_weights(
    velocities: np.ndarray,
    weights: Sequence[float] | None,
    rho: float,
) -> np.ndarray:
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if weights is None:
        return np.full(velocities.shape[0], rho / velocities.shape[0])
    result = np.asarray(weights, dtype=float)
    if result.shape != (velocities.shape[0],):
        raise ValueError("weights must have one entry per particle")
    if np.any(result < 0.0) or not np.all(np.isfinite(result)):
        raise ValueError("particle weights must be finite and non-negative")
    total = float(np.sum(result))
    if total <= 0.0:
        raise ValueError("particle weights must have positive sum")
    return result * (rho / total)


def moments_35_from_particles(
    velocities: Sequence[Sequence[float]],
    weights: Sequence[float] | None = None,
    rho: float = 1.0,
) -> np.ndarray:
    """Return weighted raw moments in the canonical Riemann35 ordering."""

    nodes = np.asarray(velocities, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] == 0:
        raise ValueError("velocities must have shape (n_particles, 3)")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("particle velocities must be finite")
    particle_weights = _normalized_weights(nodes, weights, rho)

    powers = [
        np.vstack([np.ones(nodes.shape[0]), *(nodes[:, d] ** n for n in range(1, 5))])
        for d in range(3)
    ]
    return np.asarray(
        [
            np.dot(
                particle_weights,
                powers[0][i] * powers[1][j] * powers[2][k],
            )
            for i, j, k in HYQMOM_35_INDICES
        ],
        dtype=float,
    )


def particle_macroscopic_state(
    velocities: Sequence[Sequence[float]],
    weights: Sequence[float] | None = None,
    rho: float = 1.0,
) -> ParticleMacroscopicState:
    """Compute FPCode density, stress, and heat-flux conventions directly."""

    nodes = np.asarray(velocities, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] == 0:
        raise ValueError("velocities must have shape (n_particles, 3)")
    particle_weights = _normalized_weights(nodes, weights, rho)
    probabilities = particle_weights / rho
    velocity = np.sum(probabilities[:, None] * nodes, axis=0)
    peculiar = nodes - velocity
    covariance = np.einsum("n,ni,nj->ij", probabilities, peculiar, peculiar)
    theta = float(np.trace(covariance) / 3.0)
    if theta <= 0.0:
        raise ValueError("particle temperature-like variance must be positive")
    pressure = rho * theta
    stress = -(rho * covariance - pressure * np.eye(3))
    c2 = np.einsum("ni,ni->n", peculiar, peculiar)
    heat_flux = 0.5 * rho * np.sum(
        probabilities[:, None] * c2[:, None] * peculiar, axis=0
    )
    return ParticleMacroscopicState(
        rho=rho,
        velocity=velocity,
        theta=theta,
        covariance=covariance,
        pressure=pressure,
        stress=stress,
        heat_flux=heat_flux,
    )


def coefficients_from_particles(
    velocities: Sequence[Sequence[float]],
    tau: float,
    prandtl: float = 2.0 / 3.0,
    gamma_scale: float = 0.05,
    weights: Sequence[float] | None = None,
    rho: float = 1.0,
) -> CubicFPCoefficients:
    """Solve FPCode's physical 9-by-9 coefficient system from particles."""

    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if not 0.0 < prandtl <= 1.0:
        raise ValueError("prandtl must lie in (0, 1]")
    del gamma_scale
    nodes = np.asarray(velocities, dtype=float)
    particle_weights = _normalized_weights(nodes, weights, rho)
    return coefficients_from_weighted_nodes(
        nodes,
        particle_weights,
        tau=tau,
        prandtl=prandtl,
    )


def sample_gaussian_mixture(
    components: Iterable[
        tuple[float, Sequence[float], Sequence[Sequence[float]]]
    ],
    particles: int,
    seed: int = 42,
) -> np.ndarray:
    """Draw a reproducible velocity ensemble from a Gaussian mixture."""

    if particles < 2:
        raise ValueError("particles must be at least two")
    component_list = list(components)
    if not component_list:
        raise ValueError("at least one component is required")
    fractions = np.asarray([item[0] for item in component_list], dtype=float)
    if np.any(fractions < 0.0) or float(np.sum(fractions)) <= 0.0:
        raise ValueError("component weights must be non-negative with positive sum")
    fractions /= np.sum(fractions)

    exact_counts = fractions * particles
    counts = np.floor(exact_counts).astype(int)
    for index in np.argsort(exact_counts - counts)[::-1][: particles - np.sum(counts)]:
        counts[index] += 1

    rng = np.random.default_rng(seed)
    samples = []
    for count, (_, mean, covariance) in zip(counts, component_list):
        if count:
            samples.append(rng.multivariate_normal(mean, covariance, size=count))
    result = np.vstack(samples)
    rng.shuffle(result, axis=0)
    return result


def _antithetic_normal(rng: np.random.Generator, count: int) -> np.ndarray:
    half = (count + 1) // 2
    base = rng.standard_normal((half, 3))
    return np.vstack([base, -base])[:count]


def particle_cubic_fp_step(
    velocities: Sequence[Sequence[float]],
    dt: float,
    tau: float,
    rng: np.random.Generator,
    prandtl: float = 2.0 / 3.0,
    gamma_scale: float = 0.05,
    rho: float = 1.0,
    limit_peculiar_speed: bool = True,
    enforce_sample_invariants: bool = True,
) -> tuple[np.ndarray, ParticleStepDiagnostics]:
    """Advance one homogeneous particle collision step.

    The update mirrors FPCode's exact OU factor, nonlinear drift, ``m2_lim``,
    and finite-step ``alpha`` correction.  Antithetic Gaussian increments reduce
    sampling noise.  The optional final recenter/rescale is a homogeneous
    validation aid: it enforces the collision invariants on the finite ensemble
    exactly, rather than changing the continuum model.
    """

    if dt <= 0.0 or tau <= 0.0:
        raise ValueError("dt and tau must be positive")
    nodes = np.asarray(velocities, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 2:
        raise ValueError("velocities must have shape (n_particles, 3)")
    before = particle_macroscopic_state(nodes, rho=rho)
    coefficients = coefficients_from_particles(
        nodes,
        tau=tau,
        prandtl=prandtl,
        gamma_scale=gamma_scale,
        rho=rho,
    )

    peculiar = nodes - before.velocity
    m2 = np.einsum("ni,ni->n", peculiar, peculiar)
    m2_used = np.minimum(m2, 25.0 * before.theta) if limit_peculiar_speed else m2
    nonlinear = peculiar @ coefficients.C.T
    nonlinear += (m2_used - 3.0 * before.theta)[:, None] * coefficients.gamma
    nonlinear += coefficients.beta * (
        m2_used[:, None] * peculiar
        - 2.0 * before.heat_flux[None, :] / before.rho
    )

    E = float(np.exp(-dt / tau))
    E2 = E * E
    mean_n2 = float(np.mean(np.einsum("ni,ni->n", nonlinear, nonlinear)))
    mean_cn = float(np.mean(np.einsum("ni,ni->n", peculiar, nonlinear)))
    alpha_squared = 1.0 + tau / (3.0 * before.theta) * (
        tau * (1.0 - E) ** 2 * mean_n2
        + 2.0 * (E - E2) * mean_cn
    )
    alpha = float(np.sqrt(alpha_squared)) if alpha_squared > 1.0e-6 else 1.0

    noise = _antithetic_normal(rng, nodes.shape[0])
    sigma = float(np.sqrt(before.theta * (1.0 - E2)))
    peculiar_new = (
        E * peculiar + (1.0 - E) * tau * nonlinear + sigma * noise
    ) / alpha

    if enforce_sample_invariants:
        peculiar_new -= np.mean(peculiar_new, axis=0)
        old_energy = float(np.mean(np.einsum("ni,ni->n", peculiar, peculiar)))
        new_energy = float(
            np.mean(np.einsum("ni,ni->n", peculiar_new, peculiar_new))
        )
        if new_energy <= 0.0:
            raise FloatingPointError("particle update collapsed the thermal energy")
        peculiar_new *= np.sqrt(old_energy / new_energy)

    updated = before.velocity + peculiar_new
    after = particle_macroscopic_state(updated, rho=rho)
    mass_drift = after.rho - before.rho
    momentum_drift = float(
        np.linalg.norm(after.rho * after.velocity - before.rho * before.velocity)
    )
    energy_before = before.rho * (
        np.dot(before.velocity, before.velocity) + 3.0 * before.theta
    )
    energy_after = after.rho * (
        np.dot(after.velocity, after.velocity) + 3.0 * after.theta
    )
    return updated, ParticleStepDiagnostics(
        alpha=alpha,
        mass_drift=mass_drift,
        momentum_drift=momentum_drift,
        energy_drift=float(energy_after - energy_before),
    )

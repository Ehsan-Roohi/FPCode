#!/usr/bin/env python3
"""Fair homogeneous two-peak audit of the 35-moment cubic-FP closure.

This runner is intentionally self-contained.  It uses only stable public
``hyqmom_fp`` APIs and adds no model parameter.  Its three independent paths
are

* the Stage-9 finite-mixture and 35-moment Grad--HyQMOM collision maps,
* a positive scrambled-Sobol QMC kinetic reference, and
* positive random-particle references.

The QMC and particle paths are references for the *same implemented cubic-FP
operator*, not independent molecular physics.  Moments through total degree
eight diagnose information that is predicted, but not transported, by the
35-moment closure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from math import comb
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage34-two-peak")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    HYQMOM_35_INDICES,
    finite_gaussian_mixture_fp_step,
    grad_hyqmom_fp_step,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
    particle_cubic_fp_step,
    qmc_cubic_fp_step,
    realizability_margin_35,
    reconstruct_grad_hyqmom_quadrature,
    reconstruct_gaussian_mixture_quadrature,
    sample_gaussian_mixture,
    sample_gaussian_mixture_qmc,
)
from hyqmom_fp.moments import multivariate_gaussian_raw_moment  # noqa: E402
from riemann35_patch.stage10.run_general_realizability_audit import (  # noqa: E402
    deterministic_states,
)


TWO_COMPONENT_UNIMODAL_CONTROL = (
    (0.5, np.asarray([+0.55, 0.0, 0.0]), 0.45 * np.eye(3)),
    (0.5, np.asarray([-0.55, 0.0, 0.0]), 0.45 * np.eye(3)),
)
_ARCHIVED_STATES = {state.name: state for state in deterministic_states()}
QUALIFICATION_CASE = "counterstream_ma20"
COMPONENTS = tuple(_ARCHIVED_STATES[QUALIFICATION_CASE].components)
MAXIMUM_DEGREE = 8
EXTENDED_BASIS_DEGREE = 4
FAILURE_TOLERANCE = -1.0e-10


def all_indices_through(maximum_degree: int) -> tuple[tuple[int, int, int], ...]:
    """Return 3-D multi-indices ordered by total degree, then lexicographically."""

    return tuple(
        (i, j, total - i - j)
        for total in range(maximum_degree + 1)
        for i in range(total + 1)
        for j in range(total - i + 1)
    )


INDICES = all_indices_through(MAXIMUM_DEGREE)
INDEX_POSITION = {index: position for position, index in enumerate(INDICES)}
RETAINED_POSITION = {
    index: position for position, index in enumerate(HYQMOM_35_INDICES)
}
BASIS = all_indices_through(EXTENDED_BASIS_DEGREE)


def analytic_mixture_moments(
    components: Iterable[
        tuple[float, Sequence[float], Sequence[Sequence[float]]]
    ],
    indices: Sequence[tuple[int, int, int]] = INDICES,
) -> np.ndarray:
    """Evaluate exact Gaussian-mixture raw moments for arbitrary indices."""

    component_list = list(components)
    return np.asarray(
        [
            sum(
                weight
                * multivariate_gaussian_raw_moment(index, mean, covariance)
                for weight, mean, covariance in component_list
            )
            for index in indices
        ],
        dtype=float,
    )


def moments_from_nodes(
    nodes: Sequence[Sequence[float]],
    weights: Sequence[float] | None,
    indices: Sequence[tuple[int, int, int]] = INDICES,
) -> np.ndarray:
    """Compute weighted raw moments without assuming positivity of weights."""

    velocities = np.asarray(nodes, dtype=float)
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError("nodes must have shape (n, 3)")
    if weights is None:
        probabilities = np.full(velocities.shape[0], 1.0 / velocities.shape[0])
    else:
        probabilities = np.asarray(weights, dtype=float)
        if probabilities.shape != (velocities.shape[0],):
            raise ValueError("weights must have one value per node")
    maximum = max(max(index) for index in indices)
    powers = tuple(
        np.vstack(
            [
                np.ones(velocities.shape[0]),
                *(velocities[:, direction] ** order for order in range(1, maximum + 1)),
            ]
        )
        for direction in range(3)
    )
    return np.asarray(
        [
            np.dot(
                probabilities,
                powers[0][i] * powers[1][j] * powers[2][k],
            )
            for i, j, k in indices
        ],
        dtype=float,
    )


def _raw_to_standardized(
    raw: Sequence[float],
    indices: Sequence[tuple[int, int, int]] = INDICES,
) -> np.ndarray:
    """Translate and scale a complete degree-eight raw sequence."""

    vector = np.asarray(raw, dtype=float)
    position = {index: offset for offset, index in enumerate(indices)}
    rho = vector[position[(0, 0, 0)]]
    if rho <= 0.0:
        raise ValueError("zeroth moment must be positive")
    mean = np.asarray(
        [
            vector[position[(1, 0, 0)]],
            vector[position[(0, 1, 0)]],
            vector[position[(0, 0, 1)]],
        ]
    ) / rho
    variances = np.asarray(
        [
            vector[position[(2, 0, 0)]] / rho - mean[0] ** 2,
            vector[position[(0, 2, 0)]] / rho - mean[1] ** 2,
            vector[position[(0, 0, 2)]] / rho - mean[2] ** 2,
        ]
    )
    if np.min(variances) <= 0.0:
        raise ValueError("coordinate variance must be positive")
    scales = np.sqrt(variances)
    standardized = np.empty_like(vector)
    for destination, alpha in enumerate(indices):
        value = 0.0
        for px, py, pz in product(
            range(alpha[0] + 1),
            range(alpha[1] + 1),
            range(alpha[2] + 1),
        ):
            beta = (px, py, pz)
            coefficient = 1.0
            for direction in range(3):
                coefficient *= (
                    comb(alpha[direction], beta[direction])
                    * (-mean[direction]) ** (alpha[direction] - beta[direction])
                    / scales[direction] ** alpha[direction]
                )
            value += coefficient * vector[position[beta]] / rho
        standardized[destination] = value
    return standardized


def degree_eight_margin(
    raw: Sequence[float],
    indices: Sequence[tuple[int, int, int]] = INDICES,
) -> tuple[float, float]:
    """Smallest normalized eigenvalue of the degree-four polynomial matrix.

    The 35-by-35 matrix uses products of every monomial through degree four,
    hence all raw moments through degree eight.  Positive measures make this
    matrix positive semidefinite.  Standardizing each velocity coordinate
    prevents a dimensional condition number from being mistaken for loss of
    realizability.
    """

    standardized = _raw_to_standardized(raw, indices)
    lookup = {index: standardized[position] for position, index in enumerate(indices)}
    matrix = np.asarray(
        [
            [
                lookup[tuple(left[d] + right[d] for d in range(3))]
                for right in BASIS
            ]
            for left in BASIS
        ]
    )
    matrix = 0.5 * (matrix + matrix.T)
    raw_eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
    diagonal = np.diag(matrix)
    diagonal_scale = max(float(np.mean(np.abs(diagonal))), 1.0e-15)
    if np.min(diagonal) <= 0.0:
        # A nonpositive square moment is itself a realizability failure and
        # prevents construction of D^{-1/2} H D^{-1/2}.
        return min(raw_eigenvalue, float(np.min(diagonal))) / diagonal_scale, raw_eigenvalue
    inverse_root = diagonal ** -0.5
    equilibrated = inverse_root[:, None] * matrix * inverse_root[None, :]
    normalized_eigenvalue = float(np.linalg.eigvalsh(equilibrated)[0])
    return normalized_eigenvalue, raw_eigenvalue


def _sample_steps(steps: int, stride: int) -> tuple[int, ...]:
    values = [0, *range(stride, steps + 1, stride)]
    if values[-1] != steps:
        values.append(steps)
    return tuple(values)


def _extended_record(raw_moments: np.ndarray) -> tuple[float, float]:
    return degree_eight_margin(raw_moments, INDICES)


def symmetry_project(history: np.ndarray) -> np.ndarray:
    """Average a history over the case's three coordinate reflections.

    The exact two-peak initial state and its homogeneous collision dynamics are
    invariant under independent sign changes in x, y, and z.  Setting moments
    with any odd exponent to zero is therefore not smoothing: it is the exact
    eight-reflection group average of a positive empirical measure.  Raw,
    unprojected histories remain the source of the reference PSD diagnostics.
    """

    projected = np.asarray(history, dtype=float).copy()
    odd = [
        position
        for position, index in enumerate(INDICES)
        if any(exponent % 2 for exponent in index)
    ]
    projected[..., odd] = 0.0
    return projected


def run_stage9_mixture(
    *,
    dt: float,
    final_time: float,
    sample_interval: float,
    tau: float,
    prandtl: float,
) -> dict[str, object]:
    """Advance the positive finite-Gaussian-mixture Stage-9 map.

    The Stage-9 time step itself retains its historical four-point quadrature.
    At output times only, a five-point reconstruction exposes implied moments
    through degree eight.  Retained M0--M4 entries are replaced by the actual
    evolved 35-vector before the degree-eight moment-matrix audit.
    """

    steps = int(round(final_time / dt))
    stride = max(1, int(round(sample_interval / dt)))
    requested = set(_sample_steps(steps, stride))
    moments = mixture_of_gaussians_moments_35(COMPONENTS)
    retained_history = []
    extended_history = []
    margin35_history = []
    margin8_history = []
    reconstruction_history = []
    times = []
    status = "REACHED_FINAL_TIME"
    message = ""

    def record(step: int) -> None:
        quadrature = reconstruct_gaussian_mixture_quadrature(
            moments, quadrature_order=5
        )
        extended = moments_from_nodes(quadrature.nodes, quadrature.weights)
        for index, retained_position in RETAINED_POSITION.items():
            extended[INDEX_POSITION[index]] = moments[retained_position]
        margin8, _ = _extended_record(extended)
        times.append(step * dt)
        retained_history.append(moments.copy())
        extended_history.append(extended)
        margin35_history.append(realizability_margin_35(moments))
        margin8_history.append(margin8)
        reconstruction_history.append(quadrature.relative_moment_residual)

    start = time.perf_counter()
    record(0)
    completed = 0
    for step in range(1, steps + 1):
        try:
            moments, _ = finite_gaussian_mixture_fp_step(
                moments,
                dt,
                tau,
                prandtl=prandtl,
                speed_cap=np.inf,
                quadrature_order=4,
            )
            completed = step
            if step in requested:
                record(step)
        except Exception as error:
            status = "FAILED"
            message = f"{type(error).__name__}: {error}"
            break
    return {
        "method": "stage9_finite_gaussian_mixture",
        "status": status,
        "message": message,
        "dt": dt,
        "steps": steps,
        "completed_steps": completed,
        "times": np.asarray(times),
        "retained_history": np.asarray(retained_history),
        "raw_history": np.asarray(extended_history),
        "aligned_history": np.asarray(extended_history),
        "margin35_history": np.asarray(margin35_history),
        "margin8_history": np.asarray(margin8_history),
        "reconstruction_history": np.asarray(reconstruction_history),
        "elapsed_seconds": time.perf_counter() - start,
    }


def _gaussian_moments_all(
    rho: float, mean: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            rho * multivariate_gaussian_raw_moment(index, mean, covariance)
            for index in INDICES
        ]
    )


def analytic_bgk_esbgk_histories(
    times: Sequence[float], *, tau: float, prandtl: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return homogeneous BGK and ES-BGK comparison histories.

    The collision clocks are matched through the stress-relaxation time, not
    by reusing the same symbol in dissimilar operators.  For cubic ``tau_FP``
    we use ``tau_sigma=tau_FP/2``.  BGK has ``tau_BGK=tau_sigma``.  ES-BGK is
    ``df/dt=(Pr/tau_sigma)(G_ES-f)`` with
    ``T_ES=(1-nu)*theta*I+nu*Theta`` and ``nu=1-1/Pr``. Thus both
    baselines have the same viscous/stress clock.  A 64-point Gauss--Legendre
    rule evaluates the semi-analytic homogeneous Duhamel formula. These alternative
    collision models are never scored as cubic-FP accuracy references.
    """

    if tau <= 0.0 or not 2.0 / 3.0 <= prandtl <= 1.0:
        raise ValueError("ES-BGK positivity requires Prandtl in [2/3, 1]")
    initial = analytic_mixture_moments(COMPONENTS)
    initial35 = mixture_of_gaussians_moments_35(COMPONENTS)
    state = macroscopic_state(initial35)
    equilibrium = _gaussian_moments_all(
        state.rho, state.velocity, state.theta * np.eye(3)
    )
    time_values = np.asarray(times, dtype=float)
    tau_sigma = 0.5 * tau
    tau_bgk = tau_sigma
    bgk = np.asarray(
        [
            np.exp(-value / tau_bgk) * initial
            + (1.0 - np.exp(-value / tau_bgk)) * equilibrium
            for value in time_values
        ]
    )

    nu = 1.0 - 1.0 / prandtl
    es_rate = prandtl / tau_sigma
    abscissae, weights = np.polynomial.legendre.leggauss(64)
    es_history = []
    for value in time_values:
        if value == 0.0:
            es_history.append(initial.copy())
            continue
        sample_times = 0.5 * value * (abscissae + 1.0)
        integral = np.zeros(len(INDICES))
        for sample_time, weight in zip(sample_times, weights):
            covariance = (
                state.theta * np.eye(3)
                + np.exp(-sample_time / tau_sigma)
                * (state.covariance - state.theta * np.eye(3))
            )
            target_covariance = (
                (1.0 - nu) * state.theta * np.eye(3) + nu * covariance
            )
            target = _gaussian_moments_all(
                state.rho, state.velocity, target_covariance
            )
            integral += (
                weight
                * np.exp(-es_rate * (value - sample_time))
                * es_rate
                * target
            )
        integral *= 0.5 * value
        es_history.append(np.exp(-es_rate * value) * initial + integral)
    return bgk, np.asarray(es_history), {
        "prandtl": prandtl,
        "nu": nu,
        "tau_FP": tau,
        "tau_sigma": tau_sigma,
        "tau_BGK": tau_bgk,
        "ES_collision_rate": es_rate,
        "clock_matching": "tau_sigma=tau_FP/2; BGK and ES-BGK share this stress-relaxation time",
        "convention": "df/dt=(Pr/tau_sigma)(G_ES-f); T_ES=(1-nu)theta I+nu Theta",
        "evaluation": "semi-analytic homogeneous Duhamel integral",
        "duhamel_gauss_legendre_order": 64,
    }


def legacy_ma20_initial_audit() -> dict[str, object]:
    """Audit official archived states and the demoted unimodal control at t=0."""

    archived = {state.name: state for state in deterministic_states()}
    cases = {
        name: archived[name].components
        for name in ("counterstream_ma20", "crossing_ma20")
    }
    cases["two_component_unimodal_control"] = TWO_COMPONENT_UNIMODAL_CONTROL
    records: dict[str, object] = {}
    for name, components in cases.items():
        exact = analytic_mixture_moments(components)
        moments35 = mixture_of_gaussians_moments_35(components)
        theta = macroscopic_state(moments35).theta
        exact_margin, _ = degree_eight_margin(exact)
        methods = {}
        for method in ("stage9_finite_mixture", "grad_hyqmom_35"):
            try:
                if method == "stage9_finite_mixture":
                    quadrature = reconstruct_gaussian_mixture_quadrature(
                        moments35, quadrature_order=5
                    )
                    negative_mass = 0.0
                    residual = quadrature.relative_moment_residual
                else:
                    quadrature = reconstruct_grad_hyqmom_quadrature(
                        moments35, quadrature_nodes=7
                    )
                    negative_mass = quadrature.negative_mass_fraction
                    residual = quadrature.relative_moment_residual
                implied = moments_from_nodes(quadrature.nodes, quadrature.weights)
                for index, retained_position in RETAINED_POSITION.items():
                    implied[INDEX_POSITION[index]] = moments35[retained_position]
                margin, _ = degree_eight_margin(implied)
                methods[method] = {
                    "status": "PASS",
                    "retained_reconstruction_residual": float(residual),
                    "negative_mass_fraction": float(negative_mass),
                    "necessary_H4_PSD_margin": margin,
                    "H4_scope": "necessary condition; positive is not proof, negative is decisive",
                    "degreewise_dimensionless_t0_rmse_vs_exact": _degree_rmse(
                        implied[None, :], exact[None, :], theta
                    ),
                }
            except Exception as error:
                methods[method] = {
                    "status": "FAILED",
                    "message": f"{type(error).__name__}: {error}",
                }
        records[name] = {
            "components": [
                {
                    "weight": float(weight),
                    "mean": np.asarray(mean).tolist(),
                    "covariance": np.asarray(component_covariance).tolist(),
                }
                for weight, mean, component_covariance in components
            ],
            "exact_positive_measure_necessary_H4_PSD_margin": exact_margin,
            "methods": methods,
        }
    return records


def run_grad(
    *,
    dt: float,
    final_time: float,
    sample_interval: float,
    tau: float,
    prandtl: float,
) -> dict[str, object]:
    """Advance the existing 35-moment map and expose its implied M5--M8."""

    steps = int(round(final_time / dt))
    stride = max(1, int(round(sample_interval / dt)))
    requested = set(_sample_steps(steps, stride))
    moments = mixture_of_gaussians_moments_35(COMPONENTS)
    retained_history = []
    extended_history = []
    margin35_history = []
    margin8_history = []
    negative_mass_history = []
    limiter_history = []
    minimum_hankel_history = []
    times = []
    status = "REACHED_FINAL_TIME"
    message = ""

    def record(step: int, limiter: float = 1.0) -> None:
        quadrature = reconstruct_grad_hyqmom_quadrature(
            moments, quadrature_nodes=7
        )
        extended = moments_from_nodes(quadrature.nodes, quadrature.weights)
        margin8, _ = _extended_record(extended)
        times.append(step * dt)
        retained_history.append(moments.copy())
        extended_history.append(extended)
        margin35_history.append(realizability_margin_35(moments))
        margin8_history.append(margin8)
        negative_mass_history.append(quadrature.negative_mass_fraction)
        limiter_history.append(limiter)
        minimum_hankel_history.append(quadrature.minimum_hankel_margin)

    start = time.perf_counter()
    record(0)
    completed = 0
    for step in range(1, steps + 1):
        try:
            moments, diagnostics = grad_hyqmom_fp_step(
                moments,
                dt,
                tau,
                prandtl=prandtl,
                quadrature_nodes=7,
            )
            completed = step
            if step in requested:
                record(step, diagnostics.limiter_fraction)
        except Exception as error:  # audit must preserve failures verbatim
            status = "FAILED"
            message = f"{type(error).__name__}: {error}"
            break
    elapsed = time.perf_counter() - start
    return {
        "method": "grad_hyqmom_35",
        "status": status,
        "message": message,
        "dt": dt,
        "steps": steps,
        "completed_steps": completed,
        "times": np.asarray(times),
        "retained_history": np.asarray(retained_history),
        "raw_history": np.asarray(extended_history),
        "aligned_history": np.asarray(extended_history),
        "margin35_history": np.asarray(margin35_history),
        "margin8_history": np.asarray(margin8_history),
        "negative_mass_history": np.asarray(negative_mass_history),
        "limiter_history": np.asarray(limiter_history),
        "minimum_hankel_history": np.asarray(minimum_hankel_history),
        "elapsed_seconds": elapsed,
    }


def run_qmc(task: tuple) -> dict[str, object]:
    label, points_per_component, dt, final_time, sample_interval, tau, prandtl, seed = task
    steps = int(round(final_time / dt))
    stride = max(1, int(round(sample_interval / dt)))
    requested = set(_sample_steps(steps, stride))
    nodes, weights = sample_gaussian_mixture_qmc(
        COMPONENTS, points_per_component=points_per_component, seed=seed
    )
    raw_history = []
    margin8_history = []
    margin35_history = []
    times = []
    minimum_alpha = np.inf
    maximum_energy_drift = 0.0
    maximum_momentum_drift = 0.0

    def record(step: int) -> None:
        raw = moments_from_nodes(nodes, weights)
        retained = np.asarray([raw[INDEX_POSITION[index]] for index in HYQMOM_35_INDICES])
        margin8, _ = _extended_record(raw)
        raw_history.append(raw)
        margin8_history.append(margin8)
        margin35_history.append(realizability_margin_35(retained))
        times.append(step * dt)

    start = time.perf_counter()
    record(0)
    for step in range(1, steps + 1):
        nodes, diagnostics = qmc_cubic_fp_step(
            nodes,
            weights,
            dt=dt,
            tau=tau,
            seed=seed + 1_000_003 + 104_729 * step,
            prandtl=prandtl,
        )
        minimum_alpha = min(minimum_alpha, diagnostics.alpha)
        maximum_energy_drift = max(maximum_energy_drift, abs(diagnostics.energy_drift))
        maximum_momentum_drift = max(
            maximum_momentum_drift, abs(diagnostics.momentum_drift)
        )
        if step in requested:
            record(step)
    raw = np.asarray(raw_history)
    exact_initial = analytic_mixture_moments(COMPONENTS)
    return {
        "method": label,
        "seed": seed,
        "points_per_component": points_per_component,
        "positive_nodes": int(weights.size),
        "minimum_weight": float(np.min(weights)),
        "dt": dt,
        "steps": steps,
        "times": np.asarray(times),
        "raw_history": raw,
        "aligned_history": raw - raw[:1] + exact_initial[None, :],
        "margin35_history": np.asarray(margin35_history),
        "margin8_history": np.asarray(margin8_history),
        "minimum_alpha": float(minimum_alpha),
        "maximum_energy_drift": maximum_energy_drift,
        "maximum_momentum_drift": maximum_momentum_drift,
        "elapsed_seconds": time.perf_counter() - start,
    }


def run_particle(task: tuple) -> dict[str, object]:
    particles, dt, final_time, sample_interval, tau, prandtl, seed = task
    steps = int(round(final_time / dt))
    stride = max(1, int(round(sample_interval / dt)))
    requested = set(_sample_steps(steps, stride))
    nodes = sample_gaussian_mixture(COMPONENTS, particles=particles, seed=seed)
    rng = np.random.default_rng(seed + 1_000_003)
    raw_history = []
    margin8_history = []
    margin35_history = []
    times = []
    minimum_alpha = np.inf
    maximum_energy_drift = 0.0
    maximum_momentum_drift = 0.0

    def record(step: int) -> None:
        raw = moments_from_nodes(nodes, None)
        retained = np.asarray([raw[INDEX_POSITION[index]] for index in HYQMOM_35_INDICES])
        margin8, _ = _extended_record(raw)
        raw_history.append(raw)
        margin8_history.append(margin8)
        margin35_history.append(realizability_margin_35(retained))
        times.append(step * dt)

    start = time.perf_counter()
    record(0)
    for step in range(1, steps + 1):
        nodes, diagnostics = particle_cubic_fp_step(
            nodes,
            dt=dt,
            tau=tau,
            rng=rng,
            prandtl=prandtl,
            limit_peculiar_speed=False,
            enforce_sample_invariants=True,
        )
        minimum_alpha = min(minimum_alpha, diagnostics.alpha)
        maximum_energy_drift = max(maximum_energy_drift, abs(diagnostics.energy_drift))
        maximum_momentum_drift = max(
            maximum_momentum_drift, abs(diagnostics.momentum_drift)
        )
        if step in requested:
            record(step)
    raw = np.asarray(raw_history)
    exact_initial = analytic_mixture_moments(COMPONENTS)
    return {
        "method": "particle",
        "seed": seed,
        "particles": particles,
        "dt": dt,
        "steps": steps,
        "times": np.asarray(times),
        "raw_history": raw,
        "aligned_history": raw - raw[:1] + exact_initial[None, :],
        "margin35_history": np.asarray(margin35_history),
        "margin8_history": np.asarray(margin8_history),
        "minimum_alpha": float(minimum_alpha),
        "maximum_energy_drift": maximum_energy_drift,
        "maximum_momentum_drift": maximum_momentum_drift,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _first_failure(times: np.ndarray, values: np.ndarray) -> dict[str, float] | None:
    failing = np.flatnonzero(values < FAILURE_TOLERANCE)
    if failing.size == 0:
        return None
    offset = int(failing[0])
    return {"snapshot": offset, "time_over_tau": float(times[offset]), "margin": float(values[offset])}


def _mean_sem(histories: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.asarray(histories)
    mean = np.mean(stacked, axis=0)
    if stacked.shape[0] > 1:
        sem = np.std(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0])
    else:
        sem = np.full_like(mean, np.nan)
    return mean, sem


def _degree_rmse(model: np.ndarray, reference: np.ndarray, theta0: float) -> dict[str, float]:
    difference = model - reference
    result = {}
    for degree in range(MAXIMUM_DEGREE + 1):
        positions = [position for position, index in enumerate(INDICES) if sum(index) == degree]
        scale = theta0 ** (degree / 2.0)
        result[str(degree)] = float(
            np.sqrt(np.mean((difference[:, positions] / scale) ** 2))
        )
    return result


def degreewise_history_relative_l2(
    model: np.ndarray, reference: np.ndarray
) -> dict[str, float | None]:
    """Relative L2 over all snapshots/components in each total-degree block."""

    difference = np.asarray(model) - np.asarray(reference)
    result: dict[str, float | None] = {}
    for degree in range(MAXIMUM_DEGREE + 1):
        positions = [
            position
            for position, index in enumerate(INDICES)
            if sum(index) == degree
        ]
        numerator = float(np.linalg.norm(difference[:, positions]))
        denominator = float(np.linalg.norm(np.asarray(reference)[:, positions]))
        if denominator <= 1.0e-14:
            result[str(degree)] = 0.0 if numerator <= 1.0e-14 else None
        else:
            result[str(degree)] = numerator / denominator
    return result


def retained_accuracy_gate(
    model: np.ndarray,
    reference: np.ndarray,
    theta0: float,
    *,
    threshold: float = 0.03,
) -> dict[str, object]:
    """Predeclared 3% gates by degree block and active retained component."""

    positions = [INDEX_POSITION[index] for index in HYQMOM_35_INDICES]
    scales = np.asarray(
        [theta0 ** (sum(index) / 2.0) for index in HYQMOM_35_INDICES]
    )
    model_scaled = model[:, positions] / scales[None, :]
    reference_scaled = reference[:, positions] / scales[None, :]
    difference = model_scaled - reference_scaled
    block_relative_l2 = float(
        np.linalg.norm(difference) / max(np.linalg.norm(reference_scaled), 1.0e-15)
    )
    degree_blocks: dict[str, float] = {}
    for degree in range(5):
        columns = [
            column
            for column, index in enumerate(HYQMOM_35_INDICES)
            if sum(index) == degree
        ]
        numerator = float(np.linalg.norm(difference[:, columns]))
        denominator = float(np.linalg.norm(reference_scaled[:, columns]))
        degree_blocks[str(degree)] = (
            numerator / denominator
            if denominator > 1.0e-14
            else (0.0 if numerator <= 1.0e-14 else float("inf"))
        )
    component_records = []
    inactive_components = []
    for column, index in enumerate(HYQMOM_35_INDICES):
        denominator = float(np.linalg.norm(reference_scaled[:, column]))
        name = f"M{index[0]}{index[1]}{index[2]}"
        if denominator <= 1.0e-10:
            inactive_components.append(name)
            continue
        component_records.append(
            {
                "moment": name,
                "degree": sum(index),
                "history_relative_l2": float(
                    np.linalg.norm(difference[:, column]) / denominator
                ),
            }
        )
    worst = max(component_records, key=lambda item: item["history_relative_l2"])
    degree_pass = all(value <= threshold for value in degree_blocks.values())
    component_pass = all(
        item["history_relative_l2"] <= threshold for item in component_records
    )
    return {
        "threshold": threshold,
        "metric": "dimensionless history relative L2 for every retained degree block and every active/nonzero retained component",
        "block_relative_l2": block_relative_l2,
        "degree_block_relative_l2": degree_blocks,
        "degree_blocks_pass": degree_pass,
        "active_components_pass": component_pass,
        "active_component_count": len(component_records),
        "inactive_symmetry_zero_components": inactive_components,
        "pass": bool(degree_pass and component_pass),
        "worst_component": worst,
    }


def invariant_drift(history: np.ndarray) -> dict[str, float]:
    """Maximum mass, momentum, and total second-moment drift."""

    values = np.asarray(history)
    mass = values[:, INDEX_POSITION[(0, 0, 0)]]
    momentum = np.column_stack(
        [
            values[:, INDEX_POSITION[(1, 0, 0)]],
            values[:, INDEX_POSITION[(0, 1, 0)]],
            values[:, INDEX_POSITION[(0, 0, 1)]],
        ]
    )
    energy = sum(
        values[:, INDEX_POSITION[index]]
        for index in ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    )
    return {
        "maximum_mass_drift": float(np.max(np.abs(mass - mass[0]))),
        "maximum_momentum_drift": float(
            np.max(np.linalg.norm(momentum - momentum[0], axis=1))
        ),
        "maximum_energy_trace_drift": float(np.max(np.abs(energy - energy[0]))),
    }


def history_realizability(history: np.ndarray) -> dict[str, object]:
    """H2 and necessary-H4-PSD margins plus invariant drift."""

    h2 = []
    h4 = []
    for row in np.asarray(history):
        retained = np.asarray(
            [row[INDEX_POSITION[index]] for index in HYQMOM_35_INDICES]
        )
        h2.append(realizability_margin_35(retained))
        h4.append(degree_eight_margin(row)[0])
    return {
        "minimum_H2_margin": float(np.min(h2)),
        "minimum_necessary_H4_PSD_margin": float(np.min(h4)),
        "H4_scope": "necessary PSD condition only; a positive margin is not proof of full realizability, while a negative margin is decisive failure",
        "invariant_drift": invariant_drift(history),
    }


def _final_moment_rows(
    stage9: np.ndarray,
    grad: np.ndarray,
    qmc: np.ndarray,
    qmc_sem: np.ndarray,
    particle: np.ndarray,
    particle_sem: np.ndarray,
    theta0: float,
) -> list[dict[str, object]]:
    rows = []
    for position, index in enumerate(INDICES):
        scale = theta0 ** (sum(index) / 2.0)
        rows.append(
            {
                "moment": f"M{index[0]}{index[1]}{index[2]}",
                "total_degree": sum(index),
                "stage9_finite_mixture": float(stage9[-1, position]),
                "grad_hyqmom": float(grad[-1, position]),
                "fine_qmc_mean": float(qmc[-1, position]),
                "fine_qmc_sem": float(qmc_sem[-1, position]),
                "particle_mean": float(particle[-1, position]),
                "particle_sem": float(particle_sem[-1, position]),
                "stage9_minus_qmc_scaled": float(
                    (stage9[-1, position] - qmc[-1, position]) / scale
                ),
                "grad_minus_qmc_scaled": float(
                    (grad[-1, position] - qmc[-1, position]) / scale
                ),
                "particle_minus_qmc_scaled": float(
                    (particle[-1, position] - qmc[-1, position]) / scale
                ),
            }
        )
    return rows


def _plot(
    path: Path,
    times: np.ndarray,
    stage9: np.ndarray,
    grad: np.ndarray,
    qmc: np.ndarray,
    qmc_sem: np.ndarray,
    particle: np.ndarray,
    particle_sem: np.ndarray,
    bgk: np.ndarray,
    esbgk: np.ndarray,
    stage9_margin35: np.ndarray,
    stage9_margin8: np.ndarray,
    grad_margin35: np.ndarray,
    grad_margin8: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.7, 6.0))
    selected = ((2, 0, 0), (4, 0, 0), (6, 0, 0), (8, 0, 0))
    for axis, index in zip(axes.flat, selected):
        position = INDEX_POSITION[index]
        axis.fill_between(
            times,
            qmc[:, position] - 2.0 * qmc_sem[:, position],
            qmc[:, position] + 2.0 * qmc_sem[:, position],
            color="0.82",
            label="fine QMC ±2 SEM",
        )
        axis.plot(times, qmc[:, position], "-k", lw=1.3, label="fine positive QMC")
        axis.plot(times, particle[:, position], "--", color="#0072B2", lw=1.1, label="particles")
        axis.plot(times, stage9[:, position], "-.", color="#D55E00", lw=1.35, label="Stage-9 finite mixture")
        axis.plot(times, grad[:, position], ":", color="#009E73", lw=1.35, label="Grad–HyQMOM-35")
        axis.plot(times, bgk[:, position], color="0.5", lw=0.8, alpha=0.8, label="BGK model")
        axis.plot(times, esbgk[:, position], color="#CC79A7", lw=0.8, alpha=0.8, label="ES-BGK model")
        axis.set_xlabel(r"Time, $t/\tau$")
        axis.set_ylabel(rf"$M_{{{index[0]}{index[1]}{index[2]}}}$")
        axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.01))
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.0, 3.4))
    axis.plot(times, stage9_margin35, "-", color="#D55E00", label="Stage-9 retained H2")
    axis.plot(times, stage9_margin8, "--", color="#D55E00", label="Stage-9 necessary H4 PSD")
    axis.plot(times, grad_margin35, "-", color="#009E73", label="Grad retained H2")
    axis.plot(times, grad_margin8, "--", color="#009E73", label="Grad necessary H4 PSD")
    axis.axhline(0.0, color="black", lw=0.8)
    axis.set_xlabel(r"Time, $t/\tau$")
    axis.set_ylabel("normalized minimum eigenvalue")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path.with_name("stage34_two_peak_realizability.png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=5.0e-3, help="coarse QMC dt/tau")
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=float, default=5.0e-2)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--qmc-base-points", "--qmc-coarse-points", dest="qmc_base_points", type=int, default=4096)
    parser.add_argument("--qmc-refined-points", "--qmc-fine-points", dest="qmc_refined_points", type=int, default=16384)
    parser.add_argument("--qmc-scrambles", type=int, default=4)
    parser.add_argument("--particles", type=int, default=32_768)
    parser.add_argument("--particle-seeds", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=20_260_816)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.smoke:
        args.dt = 2.0e-2
        args.final_time = min(args.final_time, 0.1)
        args.sample_interval = 2.0e-2
        args.qmc_base_points = 128
        args.qmc_refined_points = 512
        args.qmc_scrambles = 2
        args.particles = 2048
        args.particle_seeds = 2
    fine_dt = 0.5 * args.dt
    if min(args.dt, fine_dt, args.final_time, args.sample_interval, args.tau) <= 0.0:
        raise SystemExit("all time controls and tau must be positive")
    if args.qmc_scrambles < 2 or args.particle_seeds < 2:
        raise SystemExit("at least two QMC scrambles and particle seeds are required")
    args.output.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed_base + 104_729 * offset for offset in range(args.qmc_scrambles)]
    qmc_tasks = []
    for seed in seeds:
        qmc_tasks.append(("qmc_base", args.qmc_base_points, args.dt, args.final_time, args.sample_interval, args.tau, args.prandtl, seed))
        qmc_tasks.append(("qmc_node_refined", args.qmc_refined_points, args.dt, args.final_time, args.sample_interval, args.tau, args.prandtl, seed))
        qmc_tasks.append(("qmc_time_refined", args.qmc_refined_points, fine_dt, args.final_time, args.sample_interval, args.tau, args.prandtl, seed))
    particle_seeds = [args.seed_base + 32_452_843 + 104_729 * offset for offset in range(args.particle_seeds)]
    particle_tasks = [
        (args.particles, fine_dt, args.final_time, args.sample_interval, args.tau, args.prandtl, seed)
        for seed in particle_seeds
    ]

    wall_start = time.perf_counter()
    stage9 = run_stage9_mixture(
        dt=fine_dt,
        final_time=args.final_time,
        sample_interval=args.sample_interval,
        tau=args.tau,
        prandtl=args.prandtl,
    )
    grad = run_grad(
        dt=fine_dt,
        final_time=args.final_time,
        sample_interval=args.sample_interval,
        tau=args.tau,
        prandtl=args.prandtl,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        qmc_results = list(executor.map(run_qmc, qmc_tasks))
        particle_results = list(executor.map(run_particle, particle_tasks))
    wall_seconds = time.perf_counter() - wall_start

    fine_results = [item for item in qmc_results if item["method"] == "qmc_time_refined"]
    node_results = [item for item in qmc_results if item["method"] == "qmc_node_refined"]
    coarse_results = [item for item in qmc_results if item["method"] == "qmc_base"]
    fine_mean, fine_sem = _mean_sem(
        [symmetry_project(item["aligned_history"]) for item in fine_results]
    )
    coarse_mean, coarse_sem = _mean_sem(
        [symmetry_project(item["aligned_history"]) for item in coarse_results]
    )
    node_mean, node_sem = _mean_sem(
        [symmetry_project(item["aligned_history"]) for item in node_results]
    )
    particle_mean, particle_sem = _mean_sem(
        [symmetry_project(item["aligned_history"]) for item in particle_results]
    )
    stage9_history = symmetry_project(np.asarray(stage9["aligned_history"]))
    grad_history = symmetry_project(np.asarray(grad["aligned_history"]))
    times = np.asarray(grad["times"])
    if not np.array_equal(times, stage9["times"]):
        raise RuntimeError("Stage-9 and Grad sample times do not match")
    if not all(np.array_equal(times, item["times"]) for item in qmc_results + particle_results):
        raise RuntimeError("reference and closure sample times do not match")

    initial35 = mixture_of_gaussians_moments_35(COMPONENTS)
    state0 = macroscopic_state(initial35)
    bgk_history, esbgk_history, esbgk_convention = analytic_bgk_esbgk_histories(
        times, tau=args.tau, prandtl=args.prandtl
    )
    degree_rmse_stage9 = _degree_rmse(stage9_history, fine_mean, state0.theta)
    degree_rmse_grad = _degree_rmse(grad_history, fine_mean, state0.theta)
    degree_rmse_particle = _degree_rmse(particle_mean, fine_mean, state0.theta)
    qmc_refinement = _degree_rmse(coarse_mean, fine_mean, state0.theta)
    qmc_node_refinement = _degree_rmse(coarse_mean, node_mean, state0.theta)
    qmc_time_refinement = _degree_rmse(node_mean, fine_mean, state0.theta)
    degree_relative_stage9 = degreewise_history_relative_l2(
        stage9_history, fine_mean
    )
    degree_relative_grad = degreewise_history_relative_l2(
        grad_history, fine_mean
    )
    degree_relative_particle = degreewise_history_relative_l2(
        particle_mean, fine_mean
    )
    degree_relative_qmc_refinement = degreewise_history_relative_l2(
        coarse_mean, fine_mean
    )
    degree_relative_qmc_node = degreewise_history_relative_l2(
        coarse_mean, node_mean
    )
    degree_relative_qmc_time = degreewise_history_relative_l2(
        node_mean, fine_mean
    )
    qmc_node_gate = retained_accuracy_gate(
        coarse_mean, node_mean, state0.theta
    )
    qmc_time_gate = retained_accuracy_gate(
        node_mean, fine_mean, state0.theta
    )
    retained_reference_convergence = max(
        max(qmc_node_gate["degree_block_relative_l2"].values()),
        max(qmc_time_gate["degree_block_relative_l2"].values()),
    )
    reference_convergence_gate = {
        "threshold": 0.03,
        "metric": "separate 3% node- and time-refinement gates for every retained degree block and active/nonzero component",
        "maximum_relative_l2": retained_reference_convergence,
        "node_refinement_base_to_4N_same_dt": qmc_node_gate,
        "time_refinement_4N_dt_to_4N_half_dt": qmc_time_gate,
        "pass": bool(qmc_node_gate["pass"] and qmc_time_gate["pass"]),
        "higher_degrees": "M5-M8 are diagnostic unless separately converged",
    }
    stage9_accuracy_gate = retained_accuracy_gate(
        stage9_history, fine_mean, state0.theta
    )
    grad_accuracy_gate = retained_accuracy_gate(
        grad_history, fine_mean, state0.theta
    )
    final_rows = _final_moment_rows(
        stage9_history,
        grad_history,
        fine_mean,
        fine_sem,
        particle_mean,
        particle_sem,
        state0.theta,
    )
    worst_predictive = max(
        (row for row in final_rows if row["total_degree"] >= 5),
        key=lambda row: abs(float(row["grad_minus_qmc_scaled"])),
    )
    worst_predictive_stage9 = max(
        (row for row in final_rows if row["total_degree"] >= 5),
        key=lambda row: abs(float(row["stage9_minus_qmc_scaled"])),
    )

    stage9_margin35 = np.asarray(stage9["margin35_history"])
    stage9_margin8 = np.asarray(stage9["margin8_history"])
    grad_margin35 = np.asarray(grad["margin35_history"])
    grad_margin8 = np.asarray(grad["margin8_history"])
    legacy_controls = legacy_ma20_initial_audit()
    source_path = Path(__file__).resolve()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        "schema": "riemann35-stage34-rodney-two-peak-v1",
        "case": {
            "name": QUALIFICATION_CASE,
            "source": "official deterministic_states() definition from Stage 10",
            "weights": [float(component[0]) for component in COMPONENTS],
            "means": [np.asarray(component[1]).tolist() for component in COMPONENTS],
            "covariances": [np.asarray(component[2]).tolist() for component in COMPONENTS],
            "peak_separation_ratio_a_over_sigma": float(
                abs(np.asarray(COMPONENTS[0][1])[0])
                / np.sqrt(np.asarray(COMPONENTS[0][2])[0, 0])
            ),
            "homogeneous": True,
            "transport": False,
            "symmetric_zero_heat_flux": True,
            "scope_caveat": "does not test heat-flux relaxation or identify Prandtl response",
        },
        "controls": {
            "tau": args.tau,
            "prandtl": args.prandtl,
            "final_time_over_tau": args.final_time / args.tau,
            "stage9_grad_particle_and_time_refined_qmc_dt_over_tau": fine_dt / args.tau,
            "base_and_node_refined_qmc_dt_over_tau": args.dt / args.tau,
            "sample_interval_over_tau": args.sample_interval / args.tau,
            "qmc_base_points_per_component": args.qmc_base_points,
            "qmc_node_and_time_refined_points_per_component": args.qmc_refined_points,
            "qmc_scrambles": args.qmc_scrambles,
            "qmc_seed_base": args.seed_base,
            "qmc_seed_values": seeds,
            "particle_count_per_seed": args.particles,
            "particle_seed_count": args.particle_seeds,
            "particle_seed_values": particle_seeds,
            "maximum_total_moment_degree": MAXIMUM_DEGREE,
            "number_of_exported_moments": len(INDICES),
            "failure_tolerance": FAILURE_TOLERANCE,
            "reference_accuracy_symmetry_projection": (
                "exact eight-reflection positive-measure group average; "
                "raw histories retained for PSD diagnostics"
            ),
        },
        "provenance": {
            "git_head": git_head,
            "source_relative_path": str(source_path.relative_to(REPOSITORY_ROOT)),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "stage9_finite_gaussian_mixture": {
            "status": stage9["status"],
            "message": stage9["message"],
            "completed_steps": stage9["completed_steps"],
            "minimum_retained_margin": float(np.min(stage9_margin35)),
            "minimum_necessary_H4_PSD_margin": float(np.min(stage9_margin8)),
            "first_retained_realizability_failure": _first_failure(times, stage9_margin35),
            "first_necessary_H4_PSD_failure": _first_failure(times, stage9_margin8),
            "H4_scope": "necessary PSD condition only; positive is not proof of full realizability, negative is decisive failure",
            "maximum_reconstruction_relative_residual": float(np.max(stage9["reconstruction_history"])),
            "degreewise_dimensionless_history_rmse_vs_fine_qmc": degree_rmse_stage9,
            "degreewise_history_relative_l2_vs_fine_qmc": degree_relative_stage9,
            "retained_accuracy_gate": stage9_accuracy_gate,
            "worst_final_predictive_moment": worst_predictive_stage9,
            "invariant_drift": invariant_drift(stage9_history),
            "elapsed_seconds": stage9["elapsed_seconds"],
        },
        "grad_hyqmom_35": {
            "status": grad["status"],
            "message": grad["message"],
            "completed_steps": grad["completed_steps"],
            "minimum_retained_margin": float(np.min(grad_margin35)),
            "minimum_necessary_H4_PSD_margin": float(np.min(grad_margin8)),
            "first_retained_realizability_failure": _first_failure(times, grad_margin35),
            "first_necessary_H4_PSD_failure": _first_failure(times, grad_margin8),
            "H4_scope": "necessary PSD condition only; positive is not proof of full realizability, negative is decisive failure",
            "maximum_negative_quadrature_mass_fraction": float(np.max(grad["negative_mass_history"])),
            "minimum_limiter_fraction_at_samples": float(np.min(grad["limiter_history"])),
            "minimum_univariate_hankel_margin": float(np.min(grad["minimum_hankel_history"])),
            "degreewise_dimensionless_history_rmse_vs_fine_qmc": degree_rmse_grad,
            "degreewise_history_relative_l2_vs_fine_qmc": degree_relative_grad,
            "retained_accuracy_gate": grad_accuracy_gate,
            "worst_final_predictive_moment": worst_predictive,
            "invariant_drift": invariant_drift(grad_history),
            "elapsed_seconds": grad["elapsed_seconds"],
        },
        "reference_agreement": {
            "particle_degreewise_dimensionless_history_rmse_vs_fine_qmc": degree_rmse_particle,
            "coarse_to_fine_qmc_degreewise_dimensionless_history_rmse": qmc_refinement,
            "base_to_node_refined_qmc_degreewise_dimensionless_history_rmse": qmc_node_refinement,
            "node_to_time_refined_qmc_degreewise_dimensionless_history_rmse": qmc_time_refinement,
            "particle_degreewise_history_relative_l2_vs_fine_qmc": degree_relative_particle,
            "coarse_to_fine_qmc_degreewise_history_relative_l2": degree_relative_qmc_refinement,
            "base_to_node_refined_qmc_degreewise_history_relative_l2": degree_relative_qmc_node,
            "node_to_time_refined_qmc_degreewise_history_relative_l2": degree_relative_qmc_time,
            "retained_reference_convergence_gate": reference_convergence_gate,
            "minimum_raw_qmc_retained_margin": float(min(np.min(item["margin35_history"]) for item in qmc_results)),
            "minimum_raw_qmc_necessary_H4_PSD_margin": float(min(np.min(item["margin8_history"]) for item in qmc_results)),
            "minimum_raw_particle_retained_margin": float(min(np.min(item["margin35_history"]) for item in particle_results)),
            "minimum_raw_particle_necessary_H4_PSD_margin": float(min(np.min(item["margin8_history"]) for item in particle_results)),
            "fine_qmc_mean_audit": history_realizability(fine_mean),
            "particle_mean_audit": history_realizability(particle_mean),
            "maximum_fine_qmc_step_energy_drift": float(
                max(item["maximum_energy_drift"] for item in fine_results)
            ),
            "maximum_fine_qmc_step_momentum_drift": float(
                max(item["maximum_momentum_drift"] for item in fine_results)
            ),
            "maximum_particle_step_energy_drift": float(
                max(item["maximum_energy_drift"] for item in particle_results)
            ),
            "maximum_particle_step_momentum_drift": float(
                max(item["maximum_momentum_drift"] for item in particle_results)
            ),
        },
        "alternative_collision_models_not_accuracy_references": {
            "BGK": {
                "convention": "df/dt=(Maxwellian-f)/tau_BGK",
                "tau_BGK": 0.5 * args.tau,
                **history_realizability(bgk_history),
            },
            "ES_BGK": {
                **esbgk_convention,
                **history_realizability(esbgk_history),
            },
        },
        "legacy_ma20_initial_closure_controls": legacy_controls,
        "runtime": {
            "wall_seconds": wall_seconds,
            "stage9_seconds": stage9["elapsed_seconds"],
            "grad_seconds": grad["elapsed_seconds"],
            "qmc_cpu_seconds_sum": float(sum(item["elapsed_seconds"] for item in qmc_results)),
            "particle_cpu_seconds_sum": float(sum(item["elapsed_seconds"] for item in particle_results)),
            "workers": args.workers,
        },
    }

    outcome = (
        "PASS"
        if stage9["status"] == "REACHED_FINAL_TIME"
        and summary["stage9_finite_gaussian_mixture"]["first_retained_realizability_failure"] is None
        and summary["stage9_finite_gaussian_mixture"]["first_necessary_H4_PSD_failure"] is None
        and stage9_accuracy_gate["pass"]
        and reference_convergence_gate["pass"]
        else "HOLD"
    )
    summary["qualification"] = {
        "outcome": outcome,
        "requirements": {
            "stage9_completed": stage9["status"] == "REACHED_FINAL_TIME",
            "retained_H2_condition": summary["stage9_finite_gaussian_mixture"]["first_retained_realizability_failure"] is None,
            "necessary_H4_PSD_condition": summary["stage9_finite_gaussian_mixture"]["first_necessary_H4_PSD_failure"] is None,
            "retained_accuracy_each_degree_and_active_component": stage9_accuracy_gate["pass"],
            "reference_node_and_time_convergence": reference_convergence_gate["pass"],
        },
        "H4_scope": "positive is necessary but not proof of full realizability; negative is decisive failure",
    }

    with (args.output / "stage34_two_peak_moments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    np.savez_compressed(
        args.output / "stage34_two_peak_histories.npz",
        times=times,
        indices=np.asarray(INDICES, dtype=int),
        analytic_initial=analytic_mixture_moments(COMPONENTS),
        stage9_finite_mixture=stage9_history,
        grad_hyqmom=grad_history,
        qmc_time_refined_mean=fine_mean,
        qmc_time_refined_sem=fine_sem,
        qmc_node_refined_mean=node_mean,
        qmc_node_refined_sem=node_sem,
        qmc_base_mean=coarse_mean,
        qmc_base_sem=coarse_sem,
        qmc_seed_values=np.asarray(seeds, dtype=np.int64),
        qmc_base_raw=np.asarray([item["raw_history"] for item in coarse_results]),
        qmc_base_aligned=np.asarray([item["aligned_history"] for item in coarse_results]),
        qmc_node_refined_raw=np.asarray([item["raw_history"] for item in node_results]),
        qmc_node_refined_aligned=np.asarray([item["aligned_history"] for item in node_results]),
        qmc_time_refined_raw=np.asarray([item["raw_history"] for item in fine_results]),
        qmc_time_refined_aligned=np.asarray([item["aligned_history"] for item in fine_results]),
        particle_mean=particle_mean,
        particle_sem=particle_sem,
        particle_seed_values=np.asarray(particle_seeds, dtype=np.int64),
        particle_raw=np.asarray([item["raw_history"] for item in particle_results]),
        particle_aligned=np.asarray([item["aligned_history"] for item in particle_results]),
        git_head=np.asarray(git_head),
        source_sha256=np.asarray(summary["provenance"]["source_sha256"]),
        bgk=bgk_history,
        esbgk=esbgk_history,
        stage9_margin35=stage9_margin35,
        stage9_margin8=stage9_margin8,
        grad_margin35=grad_margin35,
        grad_margin8=grad_margin8,
    )
    (args.output / "stage34_two_peak_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot(
        args.output / "stage34_two_peak_M2_M4_M6_M8.png",
        times,
        stage9_history,
        grad_history,
        fine_mean,
        fine_sem,
        particle_mean,
        particle_sem,
        bgk_history,
        esbgk_history,
        stage9_margin35,
        stage9_margin8,
        grad_margin35,
        grad_margin8,
    )

    focus = {row["moment"]: row for row in final_rows if row["moment"] in {"M200", "M400", "M600", "M800"}}
    lines = [
        "# Stage 34: Rodney two-peak homogeneous audit",
        "",
        f"**Stage-9 qualification outcome: `{outcome}`.** Stage-9 completion, retained H2 and necessary H4 PSD conditions, the declared 3% gate for every retained degree block and active component, and separate 3% QMC node/time convergence gates determine this label. A positive H4 margin is necessary but is not proof of full realizability; a negative margin is decisive. Grad–HyQMOM is a disclosed comparator; predictive M5–M8 remain diagnostics unless separately converged.",
        "",
        "The qualification state is the official Stage-10 `counterstream_ma20` equal-weight, genuinely bimodal mixture. The former `±0.55 e_x`, `0.45 I` state is only a two-component unimodal control and is not qualification evidence. There is no transport, projection, activation, or kinetic memory. The Stage-9 finite Gaussian-mixture map and Grad–HyQMOM independently advance the retained 35 moments; QMC and particles advance positive velocity measures under the same implemented cubic-FP operator.",
        "",
        "| Diagnostic | Value |",
        "|---|---:|",
        f"| Stage-9 minimum retained H2 margin | {summary['stage9_finite_gaussian_mixture']['minimum_retained_margin']:.6e} |",
        f"| Stage-9 minimum necessary H4 PSD margin | {summary['stage9_finite_gaussian_mixture']['minimum_necessary_H4_PSD_margin']:.6e} |",
        f"| Grad minimum retained H2 margin | {summary['grad_hyqmom_35']['minimum_retained_margin']:.6e} |",
        f"| Grad minimum necessary H4 PSD margin | {summary['grad_hyqmom_35']['minimum_necessary_H4_PSD_margin']:.6e} |",
        f"| maximum signed negative mass fraction | {summary['grad_hyqmom_35']['maximum_negative_quadrature_mass_fraction']:.6e} |",
        f"| minimum limiter fraction at samples | {summary['grad_hyqmom_35']['minimum_limiter_fraction_at_samples']:.6e} |",
        f"| Stage-9 aggregate retained error (diagnostic) | {stage9_accuracy_gate['block_relative_l2']:.3%} |",
        f"| Stage-9 worst active retained component | {stage9_accuracy_gate['worst_component']['moment']}: {stage9_accuracy_gate['worst_component']['history_relative_l2']:.3%} |",
        f"| Stage-9 per-degree/per-active-component gate | {stage9_accuracy_gate['pass']} |",
        f"| QMC node-refinement retained gate | {qmc_node_gate['pass']} |",
        f"| QMC time-refinement retained gate | {qmc_time_gate['pass']} |",
        f"| wall runtime | {wall_seconds:.2f} s |",
        "",
        "| Degree | Stage-9 relative L2 | Grad relative L2 | Particle relative L2 | QMC node refinement | QMC time refinement |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for degree in range(1, MAXIMUM_DEGREE + 1):
        lines.append(
            f"| {degree} | {degree_relative_stage9[str(degree)]:.3%} | {degree_relative_grad[str(degree)]:.3%} | {degree_relative_particle[str(degree)]:.3%} | {degree_relative_qmc_node[str(degree)]:.3%} | {degree_relative_qmc_time[str(degree)]:.3%} |"
        )
    lines.extend(
        [
            "",
            "| Moment | Stage-9 final | Grad final | fine QMC final | particle final | Stage-9 minus QMC | Grad minus QMC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("M200", "M400", "M600", "M800"):
        row = focus[name]
        lines.append(
            f"| {name} | {row['stage9_finite_mixture']:.8g} | {row['grad_hyqmom']:.8g} | {row['fine_qmc_mean']:.8g} | {row['particle_mean']:.8g} | {row['stage9_minus_qmc_scaled']:.6e} | {row['grad_minus_qmc_scaled']:.6e} |"
        )
    lines.extend(
        [
            "",
            f"The largest final Stage-9 scaled discrepancy among predictive M5–M8 is `{worst_predictive_stage9['moment']}`: {worst_predictive_stage9['stage9_minus_qmc_scaled']:.6e}. The corresponding Grad worst moment is `{worst_predictive['moment']}`: {worst_predictive['grad_minus_qmc_scaled']:.6e}.",
            "",
            "BGK and semi-analytic ES-BGK are exported only as alternative collision-model trajectories. The fair clock match is tau_sigma=tau_FP/2: tau_BGK=tau_sigma, while ES-BGK uses df/dt=(Pr/tau_sigma)(G_ES-f), nu=1-1/Pr, and requires Pr>=2/3. Disagreement with cubic FP is not labeled numerical error. This symmetric zero-heat-flux state does not test heat-flux relaxation or identify Prandtl response.",
            "",
            "The QMC/particle references validate only the numerical closure for the implemented cubic FP collision operator. They are not hard-sphere, DSMC, or molecular-dynamics validation.",
            "The archived crossing-Ma20 control is a caveat for the Grad tail: its necessary H4 PSD margin is negative at t=0. This does not change the Stage-9 outcome; the positive Stage-9 reconstruction satisfies that necessary condition in the control, without claiming that positivity alone proves the full multidimensional moment problem.",
        ]
    )
    (args.output / "STAGE34_TWO_PEAK_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

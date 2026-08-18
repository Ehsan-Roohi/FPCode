#!/usr/bin/env python3
"""Stage-10 general-state audit for cubic-FP closures in Riemann35.

The audit compares four M5/M6 closures against exact moments of known
multivariate Gaussian mixtures:

* exact mixture moments (reference),
* Appendix-C Grad--HyQMOM with Gaussian--GQMOM marginals,
* the Stage-9 principal-axis tensor Gaussian-mixture reconstruction, and
* a single multivariate Gaussian tail.

It deliberately includes random realizable mixtures, rare hot populations,
and near-delta high-Mach two-stream/crossing-jet states.  The output separates
closure accuracy from time-integration robustness and records where the
Grad approximation becomes signed or approaches the univariate Hankel
boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    GaussianTailClosure,
    WeightedNodeTailClosure,
    coefficients_from_moments,
    finite_gaussian_mixture_fp_step,
    fit_equal_variance_marginal,
    fit_location_scale_marginal,
    grad_hyqmom_fp_step,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
    projected_fp_collision_source,
    realizability_margin_35,
    reconstruct_gaussian_mixture_quadrature,
    reconstruct_grad_hyqmom_quadrature,
)
from hyqmom_fp.moments import (  # noqa: E402
    HYQMOM_35_INDICES,
    central_moment,
    moment_value,
    multivariate_gaussian_raw_moment,
)
from riemann35_patch.stage7.run_stage7 import case_components  # noqa: E402


TAIL_INDICES = tuple(
    (i, j, order - i - j)
    for order in (5, 6)
    for i in range(order + 1)
    for j in range(order - i + 1)
)


@dataclass(frozen=True)
class AuditState:
    name: str
    family: str
    components: tuple[tuple[float, np.ndarray, np.ndarray], ...]


class ExactMixtureTailClosure:
    def __init__(
        self, components: Sequence[tuple[float, np.ndarray, np.ndarray]]
    ) -> None:
        self.components = tuple(components)

    def __call__(self, index, moments, state=None) -> float:
        del state
        if sum(index) <= 4:
            return moment_value(moments, index)
        return float(
            sum(
                probability
                * multivariate_gaussian_raw_moment(index, mean, covariance)
                for probability, mean, covariance in self.components
            )
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-states", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--long-time", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _normalize_components(
    components: Sequence[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    weights = np.asarray([item[0] for item in components], dtype=float)
    weights /= np.sum(weights)
    means = np.asarray([item[1] for item in components], dtype=float)
    covariances = np.asarray([item[2] for item in components], dtype=float)
    mean = np.sum(weights[:, None] * means, axis=0)
    means = means - mean
    theta = float(
        sum(
            weight * (np.trace(covariance) + np.dot(local_mean, local_mean))
            for weight, local_mean, covariance in zip(weights, means, covariances)
        )
        / 3.0
    )
    if theta <= 0.0:
        raise ValueError("mixture temperature must be positive")
    scale = np.sqrt(theta)
    means /= scale
    covariances /= theta
    return tuple(
        (float(weight), local_mean, covariance)
        for weight, local_mean, covariance in zip(weights, means, covariances)
    )


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))[None, :]
    return q


def random_state(index: int, seed: int) -> AuditState:
    rng = np.random.default_rng(seed + 104729 * index)
    count = int(rng.integers(2, 5))
    logits = rng.uniform(-4.5, 0.0, size=count)
    weights = np.exp(logits - np.max(logits))
    weights /= np.sum(weights)
    mean_scale = 10.0 ** rng.uniform(-0.4, 1.1)
    means = rng.normal(size=(count, 3)) * mean_scale
    components = []
    for component in range(count):
        rotation = _random_rotation(rng)
        eigenvalues = 10.0 ** rng.uniform(-4.0, 0.6, size=3)
        covariance = rotation @ np.diag(eigenvalues) @ rotation.T
        components.append((weights[component], means[component], covariance))
    return AuditState(
        name=f"random_{index:04d}",
        family="random_multivariate_mixture",
        components=_normalize_components(components),
    )


def deterministic_states() -> list[AuditState]:
    states = [
        AuditState(
            name=f"stage9_{name}",
            family="stage9_regression",
            components=tuple(case_components(name)),
        )
        for name in ("symmetric", "asymmetric", "correlated", "leptokurtic", "rare_hot")
    ]

    for weight in (0.5, 0.1, 0.02, 0.005):
        for ratio in (4.0, 25.0, 100.0, 1000.0):
            cold = np.eye(3)
            hot = np.diag([ratio, 0.55 * ratio, 1.45 * ratio])
            states.append(
                AuditState(
                    name=f"rare_hot_anisotropic_w{weight:g}_r{ratio:g}",
                    family="rare_hot_anisotropic",
                    components=_normalize_components(
                        [(1.0 - weight, np.zeros(3), cold), (weight, np.zeros(3), hot)]
                    ),
                )
            )

    for mach in (2.0, 4.0, 8.0, 20.0, 100.0):
        variance = 1.0 / mach**2
        covariance = variance * np.eye(3)
        states.append(
            AuditState(
                name=f"counterstream_ma{mach:g}",
                family="near_delta_counterstream",
                components=_normalize_components(
                    [
                        (0.5, np.asarray([1.0, 0.0, 0.0]), covariance),
                        (0.5, np.asarray([-1.0, 0.0, 0.0]), covariance),
                    ]
                ),
            )
        )
        states.append(
            AuditState(
                name=f"crossing_ma{mach:g}",
                family="near_delta_crossing",
                components=_normalize_components(
                    [
                        (0.5, np.asarray([1.0, 0.0, 0.0]), covariance),
                        (0.5, np.asarray([0.0, 1.0, 0.0]), covariance),
                    ]
                ),
            )
        )
        first_weight = 0.08
        first_mean = np.asarray([1.0, 0.35, -0.15])
        second_mean = -(first_weight / (1.0 - first_weight)) * first_mean
        states.append(
            AuditState(
                name=f"rare_beam_ma{mach:g}",
                family="near_delta_rare_beam",
                components=_normalize_components(
                    [
                        (first_weight, first_mean, covariance),
                        (1.0 - first_weight, second_mean, covariance),
                    ]
                ),
            )
        )
    return states


def tail_vector(closure, moments: np.ndarray) -> np.ndarray:
    state = macroscopic_state(moments)
    return np.asarray([closure(index, moments, state) for index in TAIL_INDICES])


def relative_error(approximation: np.ndarray, exact: np.ndarray) -> float:
    return float(
        np.linalg.norm(approximation - exact) / max(np.linalg.norm(exact), 1.0e-14)
    )


def source_from_closure(moments: np.ndarray, closure) -> tuple[np.ndarray, object]:
    coefficients = coefficients_from_moments(
        moments, tau=1.0, prandtl=2.0 / 3.0, closure=closure
    )
    source = projected_fp_collision_source(moments, coefficients, closure=closure)
    return source, coefficients


def one_state_audit(state: AuditState, dt: float) -> dict[str, object]:
    moments = mixture_of_gaussians_moments_35(state.components)
    exact_closure = ExactMixtureTailClosure(state.components)
    exact_tail = tail_vector(exact_closure, moments)
    exact_source, exact_coefficients = source_from_closure(moments, exact_closure)
    source_norm = float(np.linalg.norm(exact_source))
    exact_euler_candidate = moments + dt * exact_source
    record: dict[str, object] = {
        "name": state.name,
        "family": state.family,
        "components": len(state.components),
        "initial_realizability_margin": realizability_margin_35(moments),
        "exact_source_norm": source_norm,
        # This control separates time-discretization failures from closure
        # failures.  The M5/M6 tail is exact for the generating Gaussian
        # mixture, but a forward-Euler step can still leave the moment cone.
        "exact_euler_margin": realizability_margin_35(exact_euler_candidate),
        "exact_euler_finite": bool(np.all(np.isfinite(exact_euler_candidate))),
        "cubic_beta": float(exact_coefficients.beta),
        "gamma_norm": float(np.linalg.norm(exact_coefficients.gamma)),
        "cubic_active": bool(
            abs(exact_coefficients.beta) + np.linalg.norm(exact_coefficients.gamma)
            > 1.0e-10
        ),
        "methods": {},
    }

    method_builders: dict[str, Callable[[], tuple[object, dict[str, float]]]] = {
        "single_gaussian": lambda: (GaussianTailClosure(), {}),
    }

    def mixture_builder():
        quadrature = reconstruct_gaussian_mixture_quadrature(moments)
        return WeightedNodeTailClosure(quadrature.nodes, quadrature.weights), {
            "known_moment_residual": quadrature.relative_moment_residual,
            "quadrature_nodes": int(quadrature.weights.size),
        }

    def grad_builder():
        quadrature = reconstruct_grad_hyqmom_quadrature(moments)
        return WeightedNodeTailClosure(quadrature.nodes, quadrature.weights), {
            "known_moment_residual": quadrature.relative_moment_residual,
            "quadrature_nodes": int(quadrature.weights.size),
            "negative_mass_fraction": quadrature.negative_mass_fraction,
            "minimum_hankel_margin": quadrature.minimum_hankel_margin,
        }

    method_builders["stage9_tensor_mixture_tail"] = mixture_builder
    method_builders["grad_hyqmom_gqmom"] = grad_builder

    for method, builder in method_builders.items():
        try:
            start = time.perf_counter()
            closure, diagnostics = builder()
            approximate_tail = tail_vector(closure, moments)
            source, _ = source_from_closure(moments, closure)
            candidate = moments + dt * source
            source_error = relative_error(source, exact_source)
            cosine = float(
                np.dot(source, exact_source)
                / max(np.linalg.norm(source) * np.linalg.norm(exact_source), 1.0e-30)
            )
            method_record = {
                "status": "PASS",
                "tail_relative_error": relative_error(approximate_tail, exact_tail),
                "source_relative_error": source_error,
                "source_direction_cosine": cosine,
                "euler_step_margin": realizability_margin_35(candidate),
                "elapsed_seconds": time.perf_counter() - start,
            }
            method_record.update(diagnostics)
        except Exception as error:
            method_record = {
                "status": "FAILED",
                "message": f"{type(error).__name__}: {error}",
            }
        record["methods"][method] = method_record

    try:
        updated, diagnostics = finite_gaussian_mixture_fp_step(moments, dt, 1.0)
        record["stage9_finite_map"] = {
            "status": "PASS",
            "margin": diagnostics.realizability_margin,
            "known_moment_residual": diagnostics.reconstruction_relative_residual,
            "increment_norm": diagnostics.increment_norm,
            "finite": bool(np.all(np.isfinite(updated))),
        }
    except Exception as error:
        record["stage9_finite_map"] = {
            "status": "FAILED",
            "message": f"{type(error).__name__}: {error}",
        }
    try:
        updated, diagnostics = grad_hyqmom_fp_step(moments, dt, 1.0)
        record["guarded_grad_map"] = {
            "status": "PASS",
            "margin": diagnostics.realizability_margin,
            "limiter_fraction": diagnostics.limiter_fraction,
            "negative_mass_fraction": diagnostics.negative_mass_fraction,
            "minimum_hankel_margin": diagnostics.minimum_hankel_margin,
            "finite": bool(np.all(np.isfinite(updated))),
        }
    except Exception as error:
        record["guarded_grad_map"] = {
            "status": "FAILED",
            "message": f"{type(error).__name__}: {error}",
        }
    return record


def marginal_moment(fit, order: int) -> float:
    nodes, weights = np.polynomial.hermite.hermgauss(max(6, (order + 2) // 2))
    weights /= np.sqrt(np.pi)
    return float(
        sum(
            probability
            * np.dot(
                weights,
                (mean + np.sqrt(2.0 * variance) * nodes) ** order,
            )
            for probability, mean, variance in zip(
                fit.weights, fit.means, fit.component_variances
            )
        )
    )


def continuity_audit() -> list[dict[str, object]]:
    rows = []
    epsilon = 1.0e-8
    for third in (0.0, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0):
        row: dict[str, object] = {"s3": third, "kappa4_epsilon": epsilon}
        try:
            left = fit_equal_variance_marginal(1.0, third, 3.0 - epsilon)
            right = fit_location_scale_marginal(1.0, third, 3.0 + epsilon)
            for order in (5, 6):
                left_value = marginal_moment(left, order)
                right_value = marginal_moment(right, order)
                row[f"stage9_M{order}_left"] = left_value
                row[f"stage9_M{order}_right"] = right_value
                row[f"stage9_M{order}_seam_jump"] = abs(right_value - left_value)
            maximum_jump = max(
                float(row["stage9_M5_seam_jump"]),
                float(row["stage9_M6_seam_jump"]),
            )
            row["stage9_status"] = (
                "CONTINUOUS_WITHIN_TOLERANCE"
                if maximum_jump <= 1.0e-6
                else "FINITE_BUT_DISCONTINUOUS"
            )
        except Exception as error:
            row["stage9_status"] = "FAILED"
            row["stage9_message"] = f"{type(error).__name__}: {error}"
        try:
            from hyqmom_fp import gaussian_gqmom_marginal

            left = gaussian_gqmom_marginal(third, 3.0 - epsilon)
            right = gaussian_gqmom_marginal(third, 3.0 + epsilon)
            for order in (5, 6):
                left_value = float(np.dot(left.weights, left.nodes**order))
                right_value = float(np.dot(right.weights, right.nodes**order))
                row[f"grad_M{order}_seam_jump"] = abs(right_value - left_value)
            row["grad_status"] = "PASS"
        except Exception as error:
            row["grad_status"] = "FAILED"
            row["grad_message"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows


def summarize_method(records: list[dict[str, object]], method: str) -> dict[str, object]:
    method_rows = [record["methods"][method] for record in records]
    passed = [row for row in method_rows if row["status"] == "PASS"]
    active_passed = [
        record["methods"][method]
        for record in records
        if record["cubic_active"] and record["methods"][method]["status"] == "PASS"
    ]

    def distribution(rows, key):
        values = np.asarray([row[key] for row in rows if key in row], dtype=float)
        if values.size == 0:
            return None
        return {
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "p99": float(np.quantile(values, 0.99)),
            "maximum": float(np.max(values)),
        }

    result = {
        "attempted": len(method_rows),
        "passed": len(passed),
        "success_fraction": len(passed) / max(len(method_rows), 1),
        "tail_relative_error": distribution(passed, "tail_relative_error"),
        "active_cubic_source_relative_error": distribution(
            active_passed, "source_relative_error"
        ),
        "active_cubic_source_direction_cosine": distribution(
            active_passed, "source_direction_cosine"
        ),
        "euler_realizability_failure_fraction": (
            sum(row.get("euler_step_margin", -np.inf) < 0.0 for row in passed)
            / max(len(passed), 1)
        ),
        "elapsed_seconds": distribution(passed, "elapsed_seconds"),
    }
    if method == "grad_hyqmom_gqmom":
        result["negative_mass_fraction"] = distribution(
            passed, "negative_mass_fraction"
        )
        result["signed_quadrature_fraction"] = (
            sum(row.get("negative_mass_fraction", 0.0) > 1.0e-12 for row in passed)
            / max(len(passed), 1)
        )
        result["minimum_hankel_margin"] = distribution(
            passed, "minimum_hankel_margin"
        )
    return result


def long_time_case(
    state: AuditState, method: str, dt: float, final_time: float
) -> dict[str, object]:
    moments = mixture_of_gaussians_moments_35(state.components)
    initial = moments.copy()
    steps = int(round(final_time / dt))
    minimum_margin = realizability_margin_35(moments)
    rejected = 0
    minimum_h = dt
    maximum_negative_mass = 0.0
    limiter_history: list[dict[str, float | int]] = []

    def grad_source(vector: np.ndarray):
        quadrature = reconstruct_grad_hyqmom_quadrature(vector)
        closure = WeightedNodeTailClosure(quadrature.nodes, quadrature.weights)
        source, _ = source_from_closure(vector, closure)
        return source, quadrature.negative_mass_fraction

    for step in range(1, steps + 1):
        try:
            if method == "stage9_finite_map":
                moments, diagnostics = finite_gaussian_mixture_fp_step(
                    moments, dt, 1.0
                )
                minimum_margin = min(minimum_margin, diagnostics.realizability_margin)
                limiter_history.append(
                    {
                        "step": step,
                        "time": step * dt,
                        "lambda": 1.0,
                        "realizability_margin": float(
                            diagnostics.realizability_margin
                        ),
                    }
                )
                continue

            if method == "guarded_grad_map":
                moments, diagnostics = grad_hyqmom_fp_step(moments, dt, 1.0)
                minimum_margin = min(minimum_margin, diagnostics.realizability_margin)
                minimum_h = min(minimum_h, diagnostics.limiter_fraction * dt)
                maximum_negative_mass = max(
                    maximum_negative_mass, diagnostics.negative_mass_fraction
                )
                if diagnostics.limiter_fraction < 1.0 - 1.0e-12:
                    rejected += 1
                limiter_history.append(
                    {
                        "step": step,
                        "time": step * dt,
                        "lambda": float(diagnostics.limiter_fraction),
                        "realizability_margin": float(
                            diagnostics.realizability_margin
                        ),
                        "negative_mass_fraction": float(
                            diagnostics.negative_mass_fraction
                        ),
                    }
                )
                continue

            remaining = dt
            h = dt
            while remaining > 1.0e-15 * dt:
                h = min(h, remaining)
                source1, negative1 = grad_source(moments)
                stage1 = moments + h * source1
                if realizability_margin_35(stage1) <= 1.0e-12:
                    h *= 0.5
                    rejected += 1
                    if h < dt / 2.0**24:
                        raise FloatingPointError("Grad-HyQMOM SSPRK2 exhausted subcycling")
                    continue
                source2, negative2 = grad_source(stage1)
                candidate = 0.5 * moments + 0.5 * (stage1 + h * source2)
                margin = realizability_margin_35(candidate)
                if margin <= 1.0e-12 or not np.all(np.isfinite(candidate)):
                    h *= 0.5
                    rejected += 1
                    if h < dt / 2.0**24:
                        raise FloatingPointError("Grad-HyQMOM SSPRK2 exhausted subcycling")
                    continue
                moments = candidate
                remaining -= h
                minimum_h = min(minimum_h, h)
                minimum_margin = min(minimum_margin, margin)
                maximum_negative_mass = max(
                    maximum_negative_mass, negative1, negative2
                )
                h = min(2.0 * h, remaining) if remaining > 0.0 else h
        except Exception as error:
            return {
                "status": "FAILED",
                "failure_step": step,
                "failure_time": (step - 1) * dt,
                "message": f"{type(error).__name__}: {error}",
                "minimum_realizability_margin": minimum_margin,
                "rejected_substeps": rejected,
                "minimum_h_over_dt": minimum_h / dt,
                "per_step_history": limiter_history,
            }

    initial_state = macroscopic_state(initial)
    final_state = macroscopic_state(moments)
    return {
        "status": "REACHED_FINAL_TIME",
        "final_time": final_time,
        "minimum_realizability_margin": minimum_margin,
        "rejected_substeps": rejected,
        "minimum_h_over_dt": minimum_h / dt,
        "maximum_negative_mass_fraction": maximum_negative_mass,
        "mass_drift": abs(final_state.rho - initial_state.rho),
        "momentum_drift": float(
            np.linalg.norm(final_state.velocity - initial_state.velocity)
        ),
        "temperature_drift": abs(final_state.theta - initial_state.theta),
        "final_heat_flux_norm": float(
            np.linalg.norm(final_state.heat_flux)
            / (final_state.rho * final_state.theta**1.5)
        ),
        "final_excess_kurtosis_axes": [
            float(
                central_moment(moments, tuple(4 if axis == direction else 0 for direction in range(3)))
                / final_state.rho
                / final_state.covariance[axis, axis] ** 2
                - 3.0
            )
            for axis in range(3)
        ],
        "per_step_history": limiter_history,
    }


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "name",
        "family",
        "components",
        "initial_realizability_margin",
        "cubic_beta",
        "gamma_norm",
        "cubic_active",
        "exact_euler_margin",
        "exact_euler_finite",
    ]
    for method in (
        "single_gaussian",
        "stage9_tensor_mixture_tail",
        "grad_hyqmom_gqmom",
    ):
        fields.extend(
            [
                f"{method}_status",
                f"{method}_tail_relative_error",
                f"{method}_source_relative_error",
                f"{method}_source_direction_cosine",
                f"{method}_euler_step_margin",
            ]
        )
    fields.extend(
        [
            "stage9_finite_map_status",
            "stage9_finite_map_margin",
            "guarded_grad_map_status",
            "guarded_grad_map_margin",
            "guarded_grad_map_limiter_fraction",
        ]
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fields}
            for method in (
                "single_gaussian",
                "stage9_tensor_mixture_tail",
                "grad_hyqmom_gqmom",
            ):
                for key, value in record["methods"][method].items():
                    column = f"{method}_{key}"
                    if column in fields:
                        row[column] = value
            for key, value in record["stage9_finite_map"].items():
                column = f"stage9_finite_map_{key}"
                if column in fields:
                    row[column] = value
            for key, value in record["guarded_grad_map"].items():
                column = f"guarded_grad_map_{key}"
                if column in fields:
                    row[column] = value
            writer.writerow(row)


def make_plot(path: Path, records: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
        }
    )
    methods = (
        ("single_gaussian", "Single Gaussian", "#777777"),
        ("stage9_tensor_mixture_tail", "Stage-9 tensor mixture", "#cc3311"),
        ("grad_hyqmom_gqmom", "Grad-HyQMOM / GQMOM", "#0077bb"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for method, label, color in methods:
        active = [
            record["methods"][method]
            for record in records
            if record["cubic_active"]
            and record["methods"][method]["status"] == "PASS"
        ]
        tail = [row["tail_relative_error"] for row in active]
        source = [row["source_relative_error"] for row in active]
        axes[0].ecdf(tail, label=label, color=color)
        axes[1].ecdf(source, label=label, color=color)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel(r"relative $M_5$--$M_6$ error")
    axes[1].set_xlabel("relative cubic-FP source error")
    axes[0].set_ylabel("empirical CDF")
    axes[1].set_ylabel("empirical CDF")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)

    grad_rows = [
        record["methods"]["grad_hyqmom_gqmom"]
        for record in records
        if record["methods"]["grad_hyqmom_gqmom"]["status"] == "PASS"
    ]
    axes[2].scatter(
        [row["minimum_hankel_margin"] for row in grad_rows],
        [max(row["negative_mass_fraction"], 1.0e-16) for row in grad_rows],
        s=8,
        alpha=0.55,
        color="#0077bb",
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("minimum univariate Hankel margin")
    axes[2].set_ylabel("Grad negative-mass fraction")
    axes[2].grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.80, bottom=0.19, left=0.07, right=0.99, wspace=0.34)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_arguments()
    if args.random_states < 0 or args.workers <= 0 or args.dt <= 0.0:
        raise SystemExit("random-states must be nonnegative; workers and dt positive")
    states = deterministic_states() + [
        random_state(index, args.seed) for index in range(args.random_states)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(one_state_audit, states, [args.dt] * len(states)))

    methods = (
        "single_gaussian",
        "stage9_tensor_mixture_tail",
        "grad_hyqmom_gqmom",
    )
    summary = {
        "schema": "riemann35-cubic-fp-stage10-general-audit-v1",
        "state_count": len(records),
        "random_state_count": args.random_states,
        "source_time_step": args.dt,
        "methods": {method: summarize_method(records, method) for method in methods},
        "exact_euler_control": {
            "attempted": len(records),
            "realizability_failures": sum(
                (not record["exact_euler_finite"])
                or record["exact_euler_margin"] < 0.0
                for record in records
            ),
            "failure_fraction": sum(
                (not record["exact_euler_finite"])
                or record["exact_euler_margin"] < 0.0
                for record in records
            )
            / max(len(records), 1),
            "boundary_attempted": sum(
                record["initial_realizability_margin"] < 1.0e-4
                for record in records
            ),
            "boundary_failures": sum(
                record["initial_realizability_margin"] < 1.0e-4
                and (
                    (not record["exact_euler_finite"])
                    or record["exact_euler_margin"] < 0.0
                )
                for record in records
            ),
        },
        "stage9_finite_map": {
            "passed": sum(
                record["stage9_finite_map"]["status"] == "PASS"
                for record in records
            ),
            "attempted": len(records),
            "realizability_failures": sum(
                record["stage9_finite_map"].get("margin", -np.inf) < 0.0
                for record in records
                if record["stage9_finite_map"]["status"] == "PASS"
            ),
        },
        "guarded_grad_map": {
            "passed": sum(
                record["guarded_grad_map"]["status"] == "PASS"
                for record in records
            ),
            "attempted": len(records),
            "realizability_failures": sum(
                record["guarded_grad_map"].get("margin", -np.inf) < 0.0
                for record in records
                if record["guarded_grad_map"]["status"] == "PASS"
            ),
            "limited_steps": sum(
                record["guarded_grad_map"].get("limiter_fraction", 0.0)
                < 1.0 - 1.0e-12
                for record in records
                if record["guarded_grad_map"]["status"] == "PASS"
            ),
            "minimum_limiter_fraction": min(
                (
                    record["guarded_grad_map"].get("limiter_fraction", np.inf)
                    for record in records
                    if record["guarded_grad_map"]["status"] == "PASS"
                ),
                default=None,
            ),
        },
        "branch_continuity": continuity_audit(),
    }

    selected_names = {
        "stage9_correlated",
        "rare_hot_anisotropic_w0.02_r25",
        "counterstream_ma20",
        "crossing_ma20",
        "rare_beam_ma20",
        "counterstream_ma100",
    }
    selected = [state for state in states if state.name in selected_names]
    summary["selected_long_time"] = {
        state.name: {
            method: long_time_case(state, method, args.dt, args.long_time)
            for method in ("stage9_finite_map", "guarded_grad_map")
        }
        for state in selected
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stage10_general_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "stage10_state_records.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    write_records(args.output / "stage10_state_records.csv", records)
    make_plot(args.output / "stage10_general_behavior.png", records)

    print(f"states: {len(records)}")
    for method in methods:
        report = summary["methods"][method]
        active = report["active_cubic_source_relative_error"]
        print(
            f"{method:>28}: pass={report['passed']}/{report['attempted']} "
            f"source median/p90={active['median']:.3e}/{active['p90']:.3e}"
        )
    finite = summary["stage9_finite_map"]
    print(
        "stage9 finite-map pass/realizability failures: "
        f"{finite['passed']}/{finite['attempted']} / {finite['realizability_failures']}"
    )
    guarded = summary["guarded_grad_map"]
    print(
        "guarded Grad pass/realizability failures/limited: "
        f"{guarded['passed']}/{guarded['attempted']} / "
        f"{guarded['realizability_failures']} / {guarded['limited_steps']}"
    )
    for name, methods_result in summary["selected_long_time"].items():
        print(
            f"long-time {name}: "
            + ", ".join(
                f"{method}={result['status']}"
                for method, result in methods_result.items()
            )
        )
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()

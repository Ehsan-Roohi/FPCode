#!/usr/bin/env python3
"""Stage-8 adversarial audit of the Gaussian-EQMOM-type FP source closure."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    GaussianTailClosure,
    bgk_collision_source,
    coefficients_from_moments,
    finite_gaussian_mixture_fp_step,
    fit_equal_variance_marginal,
    fit_location_scale_marginal,
    mixture_of_gaussians_moments_35,
    projected_fp_collision_source,
    realizability_margin_35,
    reconstruct_gaussian_mixture_quadrature,
)
from riemann35_patch.stage7.run_stage7 import case_components  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--t-final", type=float, default=3.0)
    parser.add_argument("--timing-repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mixture_moment(fit, order: int) -> float:
    nodes, weights = np.polynomial.hermite.hermgauss(max(4, (order + 2) // 2))
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
    rows: list[dict[str, object]] = []
    for skewness in (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-6, 0.0):
        row: dict[str, object] = {"kappa3": skewness}
        for name, fitter in (
            ("equal_variance", fit_equal_variance_marginal),
            ("location_scale", fit_location_scale_marginal),
        ):
            try:
                fit = fitter(1.0, skewness, 4.5)
                row[f"{name}_branch"] = fit.branch
                row[f"{name}_M6"] = mixture_moment(fit, 6)
                row[f"{name}_M4_residual"] = fit.reconstruction_error
                row[f"{name}_status"] = "FINITE"
            except Exception as error:  # diagnostic script: preserve exact failure
                row[f"{name}_status"] = "FAILED"
                row[f"{name}_message"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows


def long_time_audit(dt: float, final_time: float) -> dict[str, dict[str, object]]:
    maximum_steps = int(round(final_time / dt))
    if not np.isclose(maximum_steps * dt, final_time):
        raise ValueError("t-final must be an integer multiple of dt")
    results: dict[str, dict[str, object]] = {}
    for case in ("symmetric", "asymmetric", "correlated", "leptokurtic", "rare_hot"):
        moments = mixture_of_gaussians_moments_35(case_components(case))
        record: dict[str, object] = {
            "initial_margin": realizability_margin_35(moments),
            "requested_final_time": final_time,
            "dt": dt,
            "status": "REACHED_FINAL_TIME",
        }
        maximum_residual = 0.0
        for step in range(1, maximum_steps + 1):
            try:
                moments, diagnostics = finite_gaussian_mixture_fp_step(
                    moments, dt, 1.0
                )
            except Exception as error:  # diagnostic script: preserve exact failure
                record.update(
                    {
                        "status": "RECONSTRUCTION_FAILURE",
                        "failure_step": step,
                        "failure_time": step * dt,
                        "failure_message": f"{type(error).__name__}: {error}",
                    }
                )
                break
            maximum_residual = max(
                maximum_residual, diagnostics.reconstruction_relative_residual
            )
            if diagnostics.realizability_margin < 0.0:
                record.update(
                    {
                        "status": "LEFT_REALIZABILITY_CONE",
                        "failure_step": step,
                        "failure_time": step * dt,
                        "failure_margin": diagnostics.realizability_margin,
                    }
                )
                break
        record["maximum_reconstruction_relative_residual"] = maximum_residual
        results[case] = record
    return results


def nonseparable_components() -> list[tuple[float, np.ndarray, np.ndarray]]:
    p = 0.45
    q = 1.0 - p
    first_mean = np.asarray([-0.70, 0.40, 0.10])
    second_mean = -(p / q) * first_mean
    first_covariance = np.asarray(
        [[0.50, 0.15, 0.05], [0.15, 0.30, 0.08], [0.05, 0.08, 0.25]]
    )
    second_covariance = np.asarray(
        [[0.25, -0.08, 0.04], [-0.08, 0.55, -0.12], [0.04, -0.12, 0.35]]
    )
    return [
        (p, first_mean, first_covariance),
        (q, second_mean, second_covariance),
    ]


def full_gaussian_mixture_quadrature(
    components: list[tuple[float, np.ndarray, np.ndarray]], order: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    hermite_nodes, hermite_weights = np.polynomial.hermite.hermgauss(order)
    standard_nodes = np.asarray(list(product(hermite_nodes, repeat=3))) * np.sqrt(2.0)
    standard_weights = np.asarray(
        [np.prod(item) for item in product(hermite_weights, repeat=3)]
    ) / np.pi**1.5
    all_weights = []
    all_nodes = []
    for probability, mean, covariance in components:
        cholesky = np.linalg.cholesky(covariance)
        all_nodes.append(mean + standard_nodes @ cholesky.T)
        all_weights.append(probability * standard_weights)
    return np.concatenate(all_weights), np.vstack(all_nodes)


def indexed_moments(
    weights: np.ndarray,
    nodes: np.ndarray,
    indices: list[tuple[int, int, int]],
) -> np.ndarray:
    return np.asarray(
        [
            np.dot(
                weights,
                nodes[:, 0] ** i * nodes[:, 1] ** j * nodes[:, 2] ** k,
            )
            for i, j, k in indices
        ]
    )


def nonseparable_audit() -> dict[str, object]:
    components = nonseparable_components()
    moments = mixture_of_gaussians_moments_35(components)
    reconstruction = reconstruct_gaussian_mixture_quadrature(moments)
    exact_weights, exact_nodes = full_gaussian_mixture_quadrature(components)
    tail_indices = [
        (i, j, order - i - j)
        for order in (5, 6)
        for i in range(order + 1)
        for j in range(order - i + 1)
    ]
    exact_tail = indexed_moments(exact_weights, exact_nodes, tail_indices)
    reconstructed_tail = indexed_moments(
        reconstruction.weights, reconstruction.nodes, tail_indices
    )
    tail_residual = float(
        np.linalg.norm(reconstructed_tail - exact_tail)
        / max(np.linalg.norm(exact_tail), 1.0e-15)
    )
    return {
        "known_M0_to_M4_relative_residual": reconstruction.relative_moment_residual,
        "M5_M6_relative_residual": tail_residual,
        "quadrature_nodes": int(reconstruction.weights.size),
        "marginal_branches": [item.branch for item in reconstruction.marginals],
        "initial_realizability_margin": realizability_margin_35(moments),
        "status": "FAILS_TENSOR_PRODUCT_GATE"
        if reconstruction.relative_moment_residual > 1.0e-8
        else "PASS",
    }


def cost_audit(repeats: int) -> dict[str, float]:
    moments = mixture_of_gaussians_moments_35(case_components("symmetric"))
    tail = GaussianTailClosure()
    coefficients = coefficients_from_moments(moments, tau=1.0)
    finite_gaussian_mixture_fp_step(moments, 2.5e-4, 1.0)
    projected_fp_collision_source(moments, coefficients, closure=tail)
    bgk_collision_source(moments, tau=1.0)

    start = time.perf_counter()
    for _ in range(repeats):
        finite_gaussian_mixture_fp_step(moments, 2.5e-4, 1.0)
    mixture_seconds = (time.perf_counter() - start) / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        projected_fp_collision_source(moments, coefficients, closure=tail)
    gaussian_seconds = (time.perf_counter() - start) / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        bgk_collision_source(moments, tau=1.0)
    bgk_seconds = (time.perf_counter() - start) / repeats
    return {
        "mixture_seconds_per_cell_source_step": mixture_seconds,
        "single_gaussian_seconds_per_cell_source_step": gaussian_seconds,
        "bgk_seconds_per_cell_source_step": bgk_seconds,
        "mixture_to_single_gaussian_ratio": mixture_seconds / gaussian_seconds,
        "mixture_to_bgk_ratio": mixture_seconds / bgk_seconds,
        "timing_repeats": repeats,
        "note": "Python prototype timing; not a Julia production benchmark",
    }


def main() -> None:
    args = parse_arguments()
    if args.dt <= 0.0 or args.t_final <= 0.0 or args.timing_repeats <= 0:
        raise SystemExit("dt, t-final, and timing-repeats must be positive")
    summary = {
        "schema": "riemann35-cubic-fp-stage8-structural-audit-v1",
        "continuity_audit": continuity_audit(),
        "long_time_audit": long_time_audit(args.dt, args.t_final),
        "nonseparable_multivariate_audit": nonseparable_audit(),
        "cost_audit": cost_audit(args.timing_repeats),
    }
    failed_cases = [
        case
        for case, result in summary["long_time_audit"].items()
        if result["status"] != "REACHED_FINAL_TIME"
    ]
    summary["deployment_status"] = (
        "NOT_READY_FOR_SPATIAL"
        if failed_cases
        or summary["nonseparable_multivariate_audit"]["status"] != "PASS"
        else "STRUCTURAL_GATES_PASSED"
    )
    summary["failed_long_time_cases"] = failed_cases
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"deployment status: {summary['deployment_status']}")
    for row in summary["continuity_audit"]:
        print(
            "continuity: "
            f"kappa3={row['kappa3']:.1e} "
            f"equal={row['equal_variance_status']} "
            f"location-scale M6={row.get('location_scale_M6', np.nan):.8g}"
        )
    for case, result in summary["long_time_audit"].items():
        print(
            f"long-time {case:>12}: {result['status']} "
            f"at t={result.get('failure_time', args.t_final):.6g}"
        )
    multivariate = summary["nonseparable_multivariate_audit"]
    print(
        "nonseparable residuals M0-M4/M5-M6: "
        f"{multivariate['known_M0_to_M4_relative_residual']:.3%} / "
        f"{multivariate['M5_M6_relative_residual']:.3%}"
    )
    print(
        "prototype mixture/single-Gaussian/BGK cost ratios: "
        f"{summary['cost_audit']['mixture_to_single_gaussian_ratio']:.3f} / "
        f"{summary['cost_audit']['mixture_to_bgk_ratio']:.3f}"
    )
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()

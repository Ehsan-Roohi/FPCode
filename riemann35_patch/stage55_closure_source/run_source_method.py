#!/usr/bin/env python3
"""Run one branch of the Stage-55 instantaneous source audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyqmom_fp import HYQMOM_35_INDICES, moments_35_from_qmc, qmc_cubic_fp_step
from hyqmom_fp.collision import (
    coefficients_from_moments,
    coefficients_from_weighted_nodes,
    projected_fp_collision_source,
)
from hyqmom_fp.mixture_closure import reconstruct_gaussian_mixture_quadrature
from riemann35_patch.stage54_heat_flux_third_order.run_heat_flux_method import (
    THIRD_INDICES,
    oblique_heat_flux_state,
)

METHODS = (
    "qmc_base",
    "qmc_refined",
    "exact_coeff_gaussian_projection",
    "gaussian_coeff_exact_projection",
    "gaussian_both",
    "compact_positive",
)
POSITION = {index: position for position, index in enumerate(HYQMOM_35_INDICES)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-component", type=int, default=32768)
    parser.add_argument("--base-points-per-component", type=int, default=8192)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--audit-times", default="0,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_827)
    return parser.parse_args()


def _raw_closure(nodes: np.ndarray, weights: np.ndarray):
    probabilities = np.asarray(weights, dtype=float) / np.sum(weights)

    def closure(index, moments, state):
        del moments, state
        powers = np.asarray(index, dtype=int)
        return float(np.dot(probabilities, np.prod(nodes ** powers[None, :], axis=1)))

    return closure


def _coefficient_vector(coefficients) -> np.ndarray:
    return np.concatenate((coefficients.C[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]], coefficients.gamma, [coefficients.beta]))


def _m5_contraction(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(weights, dtype=float) / np.sum(weights)
    mean = np.sum(probabilities[:, None] * nodes, axis=0)
    c = nodes - mean
    c2 = np.einsum("ni,ni->n", c, c)
    return np.einsum("n,ni,n->i", probabilities, c, c2 * c2)


def _source(method: str, nodes: np.ndarray, weights: np.ndarray, moments: np.ndarray, tau: float, prandtl: float):
    exact_coefficients = coefficients_from_weighted_nodes(nodes, weights, tau=tau, prandtl=prandtl)
    gaussian_coefficients = coefficients_from_moments(moments, tau=tau, prandtl=prandtl)
    exact_closure = _raw_closure(nodes, weights)
    support = len(weights)
    residual = 0.0
    minimum_weight = float(np.min(weights))
    if method in ("qmc_base", "qmc_refined"):
        coefficients, closure = exact_coefficients, exact_closure
    elif method == "exact_coeff_gaussian_projection":
        coefficients, closure = exact_coefficients, None
    elif method == "gaussian_coeff_exact_projection":
        coefficients, closure = gaussian_coefficients, exact_closure
    elif method == "gaussian_both":
        coefficients, closure = gaussian_coefficients, None
    else:
        quadrature = reconstruct_gaussian_mixture_quadrature(moments)
        compact_nodes = np.asarray(quadrature.nodes)
        compact_weights = np.asarray(quadrature.weights)
        coefficients = coefficients_from_weighted_nodes(compact_nodes, compact_weights, tau=tau, prandtl=prandtl)
        closure = _raw_closure(compact_nodes, compact_weights)
        support = len(compact_weights)
        residual = float(quadrature.relative_moment_residual)
        minimum_weight = float(np.min(compact_weights))
    source = projected_fp_collision_source(moments, coefficients, closure=closure)
    return source, _coefficient_vector(coefficients), support, residual, minimum_weight


def _steps_for_times(times: np.ndarray, dt: float, tau: float) -> tuple[int, ...]:
    raw = times * tau / dt
    rounded = np.rint(raw).astype(int)
    if np.max(np.abs(raw - rounded)) > 1.0e-10:
        raise ValueError("every audit time must be an integer multiple of dt/tau")
    return tuple(int(value) for value in rounded)


def main() -> None:
    args = arguments()
    times = np.asarray([float(value) for value in args.audit_times.split(",")])
    if times[0] != 0.0 or np.any(np.diff(times) <= 0.0) or times[-1] > args.final_time + 1e-14:
        raise ValueError("audit times must increase from zero and remain within final time")
    audit_steps = _steps_for_times(times, args.dt, args.tau)
    points = args.base_points_per_component if args.method == "qmc_base" else args.points_per_component
    state = oblique_heat_flux_state()
    components = state["components"]
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        source_replicates, coefficient_replicates, m5_replicates = [], [], []
        support_counts, residuals, minimum_weights = [], [], []
        from hyqmom_fp.qmc_reference import sample_gaussian_mixture_qmc
        for replicate in range(args.replicates):
            seed = args.seed + replicate * 15_485_863
            nodes, weights = sample_gaussian_mixture_qmc(components, points_per_component=points, seed=seed)
            sources, coefficients, contractions = [], [], []
            local_support, local_residual, local_minimum = [], [], []
            audit_set = set(audit_steps)
            maximum_step = audit_steps[-1]
            for step in range(maximum_step + 1):
                if step in audit_set:
                    moments = moments_35_from_qmc(nodes, weights)
                    source, coefficient, support, residual, minimum = _source(
                        args.method, nodes, weights, moments, args.tau, args.prandtl
                    )
                    sources.append(source)
                    coefficients.append(coefficient)
                    contractions.append(_m5_contraction(nodes, weights))
                    local_support.append(support)
                    local_residual.append(residual)
                    local_minimum.append(minimum)
                if step < maximum_step:
                    nodes, _ = qmc_cubic_fp_step(
                        nodes, weights, dt=args.dt, tau=args.tau,
                        seed=seed + 1_000_003 + 104_729 * (step + 1), prandtl=args.prandtl,
                    )
            source_replicates.append(sources)
            coefficient_replicates.append(coefficients)
            m5_replicates.append(contractions)
            support_counts.append(local_support)
            residuals.append(local_residual)
            minimum_weights.append(local_minimum)
        archive = args.output / f"stage55_{args.method}.npz"
        np.savez_compressed(
            archive, times=times, sources=np.asarray(source_replicates),
            coefficients=np.asarray(coefficient_replicates), m5=np.asarray(m5_replicates),
            support=np.asarray(support_counts), residual=np.asarray(residuals),
            minimum_weight=np.asarray(minimum_weights), third_positions=np.asarray([POSITION[index] for index in THIRD_INDICES]),
        )
        summary = {
            "schema": "riemann35-stage55-source-method-v1", "method": args.method,
            "replicates": args.replicates, "points_per_component": points,
            "dt_over_tau": args.dt / args.tau, "audit_times": times.tolist(),
            "minimum_weight": float(np.min(minimum_weights)),
            "maximum_retained_moment_residual": float(np.max(residuals)),
            "maximum_support": int(np.max(support_counts)),
        }
        (args.output / f"stage55_{args.method}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        failure = {"method": args.method, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        (args.output / f"stage55_{args.method}_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()

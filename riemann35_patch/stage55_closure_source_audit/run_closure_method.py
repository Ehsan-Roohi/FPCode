#!/usr/bin/env python3
"""Run one method in the Stage-55 closure-source audit.

Stage 54 showed that the positive 35-moment finite-mixture map is robust but
does not reproduce the third-order history accurately.  This runner separates
the two possible causes: the instantaneous M5/M6 source closure and the
finite-time map.  The new candidate carries only the 35 retained moments plus
49 raw M5/M6 values.  It does not retain a velocity microstate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from math import comb
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    DynamicHighOrderState,
    HYQMOM_35_INDICES,
    coefficients_from_weighted_nodes,
    dynamic_high_order_fp_step,
    finite_gaussian_mixture_fp_step,
    macroscopic_state,
    moments_35_from_qmc,
    positive_microstate_from_components,
    qmc_cubic_fp_step,
    realizability_margin_35,
    reconstruct_two_population_quadrature,
)
from hyqmom_fp.grad_hyqmom import WeightedNodeTailClosure  # noqa: E402
from hyqmom_fp.moments import multivariate_gaussian_raw_moment  # noqa: E402
from riemann35_patch.stage54_heat_flux_third_order.run_heat_flux_method import (  # noqa: E402
    oblique_heat_flux_state,
)


METHODS = (
    "qmc_reference",
    "gaussian_hyqmom35",
    "dynamic_unprojected",
    "projected_tail_base",
    "projected_tail_node_refined",
    "projected_tail_time_refined",
)
POSITION = {index: position for position, index in enumerate(HYQMOM_35_INDICES)}
TAIL_INDICES = tuple(
    index
    for order in (5, 6)
    for index in product(range(order + 1), repeat=3)
    if sum(index) == order
)
INVARIANT_INDICES = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=float, default=2.5e-2)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--points-per-component", type=int, default=32768)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20_260_822)
    parser.add_argument("--tail-relaxation-time", type=float, default=1.0e-2)
    parser.add_argument("--high-order-quadrature-nodes", type=int, default=4)
    parser.add_argument("--regularization-fraction", type=float, default=0.03)
    parser.add_argument("--planar-rotation-degrees", type=float, default=17.0)
    parser.add_argument("--tilt-y-degrees", type=float, default=29.0)
    parser.add_argument("--tilt-x-degrees", type=float, default=41.0)
    return parser.parse_args()


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    value = int(round(numerator / denominator))
    if value < 1 or not np.isclose(value * denominator, numerator, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be a positive integer multiple of dt")
    return value


def _raw_tail_from_nodes(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    powers = [
        np.vstack([np.ones(nodes.shape[0]), *(nodes[:, axis] ** order for order in range(1, 7))])
        for axis in range(3)
    ]
    return np.asarray(
        [
            np.dot(
                weights,
                powers[0][index[0]] * powers[1][index[1]] * powers[2][index[2]],
            )
            for index in TAIL_INDICES
        ],
        dtype=float,
    )


def _direct_node_source(
    moments: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
    *,
    tau: float,
    prandtl: float,
) -> np.ndarray:
    """Evaluate the continuous cubic-FP generator directly on positive nodes."""

    probabilities = np.asarray(weights, dtype=float) / np.sum(weights)
    state = macroscopic_state(moments)
    coefficients = coefficients_from_weighted_nodes(
        nodes, probabilities, tau=tau, prandtl=prandtl
    )
    peculiar = nodes - state.velocity
    c2 = np.einsum("ni,ni->n", peculiar, peculiar)
    drift = -peculiar / tau + peculiar @ coefficients.C.T
    drift += (c2 - 3.0 * state.theta)[:, None] * coefficients.gamma
    drift += coefficients.beta * (
        c2[:, None] * peculiar - 2.0 * state.heat_flux[None, :] / state.rho
    )
    powers = [
        np.vstack([np.ones(nodes.shape[0]), *(nodes[:, axis] ** order for order in range(1, 5))])
        for axis in range(3)
    ]
    diffusion = state.theta / tau
    source = np.zeros(len(HYQMOM_35_INDICES))
    for position, alpha in enumerate(HYQMOM_35_INDICES):
        value = 0.0
        for direction in range(3):
            if alpha[direction] > 0:
                reduced = list(alpha)
                reduced[direction] -= 1
                feature = (
                    powers[0][reduced[0]]
                    * powers[1][reduced[1]]
                    * powers[2][reduced[2]]
                )
                value += alpha[direction] * np.dot(
                    probabilities, feature * drift[:, direction]
                )
            if alpha[direction] > 1:
                reduced = list(alpha)
                reduced[direction] -= 2
                raw_position = POSITION[tuple(reduced)]
                value += (
                    alpha[direction]
                    * (alpha[direction] - 1)
                    * diffusion
                    * moments[raw_position]
                    / state.rho
                )
        source[position] = state.rho * value
    # Enforce the identical round-off projection used by the moment source.
    for index in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        source[POSITION[index]] = 0.0
    energy_positions = [
        POSITION[(2, 0, 0)], POSITION[(0, 2, 0)], POSITION[(0, 0, 2)]
    ]
    energy_leak = float(sum(source[position] for position in energy_positions))
    for position in energy_positions:
        source[position] -= energy_leak / 3.0
    return source


def exact_initial_tail(components: tuple) -> np.ndarray:
    """Return analytic raw M5/M6 values of the known Gaussian mixture."""

    return np.asarray(
        [
            sum(
                float(weight)
                * multivariate_gaussian_raw_moment(index, mean, covariance)
                for weight, mean, covariance in components
            )
            for index in TAIL_INDICES
        ],
        dtype=float,
    )


def central_source_components(moments: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Convert raw collision sources to the ten central third-order sources.

    Collision conserves mass and momentum, so the mean velocity is constant
    and only the raw-to-central binomial transform is required.
    """

    velocity = macroscopic_state(moments).velocity
    values = []
    for alpha in (index for index in HYQMOM_35_INDICES if sum(index) == 3):
        value = 0.0
        for bx in range(alpha[0] + 1):
            for by in range(alpha[1] + 1):
                for bz in range(alpha[2] + 1):
                    beta = (bx, by, bz)
                    coefficient = np.prod(
                        [comb(alpha[direction], beta[direction]) for direction in range(3)]
                    )
                    mean_factor = np.prod(
                        [
                            (-velocity[direction]) ** (alpha[direction] - beta[direction])
                            for direction in range(3)
                        ]
                    )
                    value += coefficient * mean_factor * source[POSITION[beta]]
        values.append(value)
    return np.asarray(values, dtype=float)


def _qmc_sample(
    moments: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
    tau: float,
    prandtl: float,
) -> tuple[np.ndarray, np.ndarray]:
    source = _direct_node_source(
        moments,
        nodes,
        weights,
        tau=tau,
        prandtl=prandtl,
    )
    return source, _raw_tail_from_nodes(nodes, weights)


def _run_qmc_replicate(task: tuple) -> dict[str, object]:
    (
        components,
        replicate,
        points_per_component,
        dt,
        steps,
        sample_steps,
        tau,
        prandtl,
        seed,
    ) = task
    micro, target, projection = positive_microstate_from_components(
        components,
        points_per_component=points_per_component,
        seed=seed,
        provenance=f"stage55-positive-full-fp-reference-{replicate}",
    )
    nodes = micro.velocities.copy()
    weights = micro.weights.copy()
    source, tail = _qmc_sample(target, nodes, weights, tau, prandtl)
    histories = [target.copy()]
    sources = [source]
    tails = [tail]
    maximum_momentum_drift = 0.0
    maximum_energy_drift = 0.0
    sample_set = set(sample_steps[1:])
    start = time.perf_counter()
    for step in range(1, steps + 1):
        nodes, diagnostics = qmc_cubic_fp_step(
            nodes,
            weights,
            dt=dt,
            tau=tau,
            seed=seed + 1_000_003 + 104_729 * step,
            prandtl=prandtl,
        )
        maximum_momentum_drift = max(maximum_momentum_drift, diagnostics.momentum_drift)
        maximum_energy_drift = max(maximum_energy_drift, abs(diagnostics.energy_drift))
        if step in sample_set:
            moments = moments_35_from_qmc(nodes, weights)
            source, tail = _qmc_sample(moments, nodes, weights, tau, prandtl)
            histories.append(moments)
            sources.append(source)
            tails.append(tail)
        if step == steps or step % max(steps // 8, 1) == 0:
            print(
                f"[stage55] qmc replicate={replicate} step={step}/{steps} time={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                flush=True,
            )
    return {
        "histories": np.asarray(histories),
        "sources": np.asarray(sources),
        "tails": np.asarray(tails),
        "minimum_weight": float(np.min(weights)),
        "projection_relative_residual": projection.relative_moment_residual,
        "minimum_probability": projection.minimum_probability,
        "maximum_momentum_drift": maximum_momentum_drift,
        "maximum_energy_drift": maximum_energy_drift,
        "minimum_H2_margin": float(
            min(realizability_margin_35(item) for item in histories)
        ),
        "elapsed_seconds": time.perf_counter() - start,
    }


def _invariants(histories: np.ndarray) -> dict[str, float]:
    mass = histories[..., POSITION[(0, 0, 0)]]
    momentum = histories[..., [POSITION[(1, 0, 0)], POSITION[(0, 1, 0)], POSITION[(0, 0, 1)]]]
    energy = sum(
        histories[..., POSITION[index]]
        for index in ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    )
    return {
        "maximum_mass_drift": float(np.max(np.abs(mass - mass[..., :1]))),
        "maximum_momentum_drift": float(
            np.max(np.linalg.norm(momentum - momentum[..., :1, :], axis=-1))
        ),
        "maximum_energy_trace_drift": float(np.max(np.abs(energy - energy[..., :1]))),
        "minimum_H2_margin": float(
            min(realizability_margin_35(history) for replicate in histories for history in replicate)
        ),
    }


def _run_gaussian(
    target: np.ndarray,
    *,
    dt: float,
    steps: int,
    sample_steps: tuple[int, ...],
    tau: float,
    prandtl: float,
) -> dict[str, object]:
    moments = target.copy()
    histories = [moments.copy()]
    minimum_margin = realizability_margin_35(moments)
    sample_set = set(sample_steps[1:])
    start = time.perf_counter()
    for step in range(1, steps + 1):
        moments, _ = finite_gaussian_mixture_fp_step(
            moments, dt, tau, prandtl=prandtl, speed_cap=np.inf
        )
        minimum_margin = min(minimum_margin, realizability_margin_35(moments))
        if step in sample_set:
            histories.append(moments.copy())
        if step == steps or step % max(steps // 8, 1) == 0:
            print(f"[stage55] gaussian step={step}/{steps}", flush=True)
    return {
        "histories": np.asarray(histories),
        "sources": np.empty((0, 35)),
        "tails": np.empty((0, len(TAIL_INDICES))),
        "minimum_weight": 0.0,
        "minimum_H2_margin": float(minimum_margin),
        "minimum_limiter": 1.0,
        "maximum_tail_projection_distance": 0.0,
        "persistent_state_scalars": 35,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _run_dynamic(
    target: np.ndarray,
    components: tuple,
    *,
    dt: float,
    steps: int,
    sample_steps: tuple[int, ...],
    tau: float,
    prandtl: float,
    projected: bool,
    tail_relaxation_time: float,
    high_order_quadrature_nodes: int,
) -> dict[str, object]:
    state = DynamicHighOrderState(
        moments=target.copy(),
        tail_moments=exact_initial_tail(components),
        maximum_order=6,
    )
    histories = [state.moments.copy()]
    tails = [state.tail_moments.copy()]
    minimum_margin = realizability_margin_35(state.moments)
    minimum_limiter = 1.0
    maximum_projection_distance = 0.0
    minimum_projection_weight = float("inf")
    sample_set = set(sample_steps[1:])
    retention = np.exp(-dt / tail_relaxation_time) if projected else 1.0
    start = time.perf_counter()
    for step in range(1, steps + 1):
        state, diagnostics = dynamic_high_order_fp_step(
            state,
            dt,
            tau,
            prandtl=prandtl,
            minimum_skewness_norm=0.05,
            high_order_quadrature_nodes=high_order_quadrature_nodes,
        )
        minimum_limiter = min(minimum_limiter, diagnostics.limiter_fraction)
        minimum_margin = min(minimum_margin, diagnostics.realizability_margin)
        if projected:
            projection_quadrature = reconstruct_two_population_quadrature(
                state.moments,
                quadrature_nodes=4,
                minimum_skewness_norm=0.05,
                residual_correction=False,
            )
            projection_closure = WeightedNodeTailClosure(
                projection_quadrature.nodes,
                projection_quadrature.weights,
                maximum_order=6,
            )
            algebraic_tail = np.asarray(
                [projection_closure(index, state.moments) for index in TAIL_INDICES]
            )
            minimum_projection_weight = min(
                minimum_projection_weight,
                float(np.min(projection_quadrature.weights)),
            )
            denominator = max(float(np.linalg.norm(algebraic_tail)), 1.0e-14)
            maximum_projection_distance = max(
                maximum_projection_distance,
                float(np.linalg.norm(state.tail_moments - algebraic_tail) / denominator),
            )
            projected_tail = algebraic_tail + retention * (
                state.tail_moments - algebraic_tail
            )
            state = DynamicHighOrderState(
                moments=state.moments,
                tail_moments=projected_tail,
                maximum_order=6,
            )
        if step in sample_set:
            histories.append(state.moments.copy())
            tails.append(state.tail_moments.copy())
        if step == steps or step % max(steps // 8, 1) == 0:
            print(
                f"[stage55] dynamic projected={projected} step={step}/{steps} limiter={diagnostics.limiter_fraction:.3e}",
                flush=True,
            )
    return {
        "histories": np.asarray(histories),
        "sources": np.empty((0, 35)),
        "tails": np.asarray(tails),
        "minimum_weight": float(minimum_projection_weight) if projected else 0.0,
        "minimum_H2_margin": float(minimum_margin),
        "minimum_limiter": float(minimum_limiter),
        "maximum_tail_projection_distance": float(maximum_projection_distance),
        "persistent_state_scalars": 35 + len(TAIL_INDICES),
        "tail_relaxation_time_over_tau": tail_relaxation_time / tau if projected else None,
        "high_order_quadrature_nodes": high_order_quadrature_nodes,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    args = arguments()
    if min(args.dt, args.final_time, args.tau, args.sample_interval) <= 0.0:
        raise ValueError("all time scales must be positive")
    if args.method.startswith("projected_") and args.tail_relaxation_time <= 0.0:
        raise ValueError("tail-relaxation-time must be positive")
    steps = _integer_ratio(args.final_time, args.dt, "final-time")
    sample_every = _integer_ratio(args.sample_interval, args.dt, "sample-interval")
    sample_steps = tuple([0, *range(sample_every, steps + 1, sample_every)])
    if sample_steps[-1] != steps:
        sample_steps = (*sample_steps, steps)

    initial = oblique_heat_flux_state(
        regularization_fraction=args.regularization_fraction,
        planar_rotation_degrees=args.planar_rotation_degrees,
        tilt_y_degrees=args.tilt_y_degrees,
        tilt_x_degrees=args.tilt_x_degrees,
    )
    components = tuple(initial["components"])
    target = np.asarray(initial["moments"], dtype=float)
    args.output.mkdir(parents=True, exist_ok=True)
    failure_path = args.output / f"stage55_{args.method}_failure.json"
    print(
        f"[stage55] method={args.method} dt={args.dt} steps={steps} started={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        flush=True,
    )
    try:
        if args.method == "qmc_reference":
            tasks = [
                (
                    components,
                    replicate,
                    args.points_per_component,
                    args.dt,
                    steps,
                    sample_steps,
                    args.tau,
                    args.prandtl,
                    args.seed + 15_485_863 * replicate,
                )
                for replicate in range(args.replicates)
            ]
            with ProcessPoolExecutor(max_workers=min(args.workers, args.replicates)) as executor:
                results = list(executor.map(_run_qmc_replicate, tasks))
        elif args.method == "gaussian_hyqmom35":
            results = [
                _run_gaussian(
                    target,
                    dt=args.dt,
                    steps=steps,
                    sample_steps=sample_steps,
                    tau=args.tau,
                    prandtl=args.prandtl,
                )
            ]
        else:
            results = [
                _run_dynamic(
                    target,
                    components,
                    dt=args.dt,
                    steps=steps,
                    sample_steps=sample_steps,
                    tau=args.tau,
                    prandtl=args.prandtl,
                    projected=args.method != "dynamic_unprojected",
                    tail_relaxation_time=args.tail_relaxation_time,
                    high_order_quadrature_nodes=args.high_order_quadrature_nodes,
                )
            ]
    except Exception as error:
        failure = {
            "schema": "riemann35-stage55-method-failure-v1",
            "method": args.method,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise

    histories = np.asarray([item["histories"] for item in results])
    source_arrays = [item["sources"] for item in results]
    tail_arrays = [item["tails"] for item in results]
    sources = np.asarray(source_arrays) if source_arrays[0].size else np.empty((0, 0, 35))
    tails = np.asarray(tail_arrays) if tail_arrays[0].size else np.empty((0, 0, len(TAIL_INDICES)))
    times = np.asarray(sample_steps, dtype=float) * args.dt / args.tau
    np.savez_compressed(
        args.output / f"stage55_{args.method}.npz",
        times=times,
        histories=histories,
        sources=sources,
        tails=tails,
        tail_indices=np.asarray(TAIL_INDICES, dtype=int),
    )
    summary = {
        "schema": "riemann35-stage55-closure-method-v1",
        "method": args.method,
        "controls": {
            "dt_over_tau": args.dt / args.tau,
            "final_time_over_tau": args.final_time / args.tau,
            "sample_interval_over_tau": args.sample_interval / args.tau,
            "points_per_component": args.points_per_component if args.method == "qmc_reference" else None,
            "replicates": len(results),
            "prandtl": args.prandtl,
            "tail_relaxation_time_over_tau": (
                args.tail_relaxation_time / args.tau if args.method.startswith("projected_") else None
            ),
            "high_order_quadrature_nodes": (
                args.high_order_quadrature_nodes if "tail" in args.method or args.method == "dynamic_unprojected" else None
            ),
        },
        "initial_state": {
            "mass_error": initial["mass_error"],
            "momentum_norm": initial["momentum_norm"],
            "energy_trace_error": initial["energy_trace_error"],
            "heat_flux": initial["heat_flux"],
            "third_components": initial["third_components"],
            "analytic_tail_norm": np.linalg.norm(exact_initial_tail(components)),
        },
        "invariants": _invariants(histories),
        "minimum_weight": float(min(item.get("minimum_weight", 0.0) for item in results)),
        "minimum_limiter": float(min(item.get("minimum_limiter", 1.0) for item in results)),
        "minimum_H2_margin": float(min(item["minimum_H2_margin"] for item in results)),
        "persistent_state_scalars": int(max(item.get("persistent_state_scalars", 35) for item in results)),
        "replicate_diagnostics": [
            {key: value for key, value in item.items() if key not in ("histories", "sources", "tails")}
            for item in results
        ],
    }
    (args.output / f"stage55_{args.method}_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    if failure_path.exists():
        failure_path.unlink()
    print(json.dumps(_jsonable(summary), indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()

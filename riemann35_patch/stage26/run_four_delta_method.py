#!/usr/bin/env python3
"""Run one method in the Stage-26 regularized four-delta audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    HYQMOM_35_INDICES,
    ActivationHysteresis,
    PositiveMicrostate,
    adaptive_tail_memory_fp_step,
    finite_gaussian_mixture_fp_step,
    grad_hyqmom_fp_step,
    initialize_adaptive_tail_memory,
    kinetic_activation_sensor,
    macroscopic_state,
    moments_35_from_qmc,
    positive_microstate_from_components,
    qmc_cubic_fp_step,
    realizability_margin_35,
    reconstruct_gaussian_mixture_quadrature,
    reconstruct_grad_hyqmom_quadrature,
    regularized_four_delta_state,
)


METHODS = ("full_fp_qmc", "stage9_mixture", "grad_gqmom", "adaptive_memory")
TAIL_INDICES = ((4, 2, 0), (6, 0, 0))
METRIC_NAMES = (
    "rho",
    "momentum_norm",
    "energy_trace",
    "M300",
    "M030",
    "M210",
    "M120",
    "M400",
    "M040",
    "M220",
    "M420",
    "M600",
    "third_order_norm",
    "heat_flux_norm",
    "realizability_margin",
)
POSITION = {index: offset for offset, index in enumerate(HYQMOM_35_INDICES)}
THIRD_INDICES = tuple(index for index in HYQMOM_35_INDICES if sum(index) == 3)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--points-per-component", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20_260_814)
    parser.add_argument("--regularization-fraction", type=float, default=0.03)
    parser.add_argument("--rotation-degrees", type=float, default=17.0)
    parser.add_argument("--energy-trace", type=float, default=1.0)
    parser.add_argument("--sensor-every", type=int, default=10)
    return parser.parse_args()


def _sample_steps(steps: int, sample_every: int) -> tuple[int, ...]:
    values = [0, *range(sample_every, steps + 1, sample_every)]
    if values[-1] != steps:
        values.append(steps)
    return tuple(values)


def _raw_node_moments(
    nodes: np.ndarray, weights: np.ndarray, indices: tuple[tuple[int, int, int], ...]
) -> np.ndarray:
    return np.asarray(
        [
            np.dot(
                weights,
                nodes[:, 0] ** index[0]
                * nodes[:, 1] ** index[1]
                * nodes[:, 2] ** index[2],
            )
            for index in indices
        ],
        dtype=float,
    )


def _metrics(
    moments: np.ndarray,
    *,
    tail_nodes: np.ndarray,
    tail_weights: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(moments, dtype=float)
    macro = macroscopic_state(vector)
    energy = float(
        sum(
            vector[POSITION[index]]
            for index in ((2, 0, 0), (0, 2, 0), (0, 0, 2))
        )
    )
    retained = {
        index: float(vector[POSITION[index]])
        for index in (
            (3, 0, 0),
            (0, 3, 0),
            (2, 1, 0),
            (1, 2, 0),
            (4, 0, 0),
            (0, 4, 0),
            (2, 2, 0),
        )
    }
    tail = _raw_node_moments(tail_nodes, tail_weights, TAIL_INDICES)
    third = np.asarray([vector[POSITION[index]] for index in THIRD_INDICES])
    return np.asarray(
        [
            macro.rho,
            np.linalg.norm(macro.rho * macro.velocity),
            energy,
            retained[(3, 0, 0)],
            retained[(0, 3, 0)],
            retained[(2, 1, 0)],
            retained[(1, 2, 0)],
            retained[(4, 0, 0)],
            retained[(0, 4, 0)],
            retained[(2, 2, 0)],
            tail[0],
            tail[1],
            np.linalg.norm(third),
            np.linalg.norm(macro.heat_flux),
            realizability_margin_35(vector),
        ],
        dtype=float,
    )


def _algebraic_metrics(moments: np.ndarray, method: str) -> np.ndarray:
    if method == "stage9_mixture":
        quadrature = reconstruct_gaussian_mixture_quadrature(moments)
    elif method == "grad_gqmom":
        quadrature = reconstruct_grad_hyqmom_quadrature(moments)
    else:  # pragma: no cover - protected by callers
        raise ValueError(f"no algebraic tail builder for {method}")
    return _metrics(
        moments,
        tail_nodes=np.asarray(quadrature.nodes),
        tail_weights=np.asarray(quadrature.weights),
    )


def _progress(method: str, replicate: int, step: int, steps: int) -> None:
    print(
        f"[stage26] method={method} replicate={replicate} "
        f"step={step}/{steps} time={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        flush=True,
    )


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
        provenance=f"stage26-four-delta-full-fp-qmc-{replicate}",
    )
    velocities = micro.velocities.copy()
    weights = micro.weights.copy()
    histories = [target.copy()]
    metrics = [
        _metrics(target, tail_nodes=velocities, tail_weights=weights)
    ]
    maximum_momentum_drift = 0.0
    maximum_energy_drift = 0.0
    start = time.perf_counter()
    sample_set = set(sample_steps[1:])
    for step in range(1, steps + 1):
        velocities, diagnostics = qmc_cubic_fp_step(
            velocities,
            weights,
            dt=dt,
            tau=tau,
            seed=seed + 1_000_003 + 104_729 * step,
            prandtl=prandtl,
        )
        maximum_momentum_drift = max(
            maximum_momentum_drift, diagnostics.momentum_drift
        )
        maximum_energy_drift = max(
            maximum_energy_drift, abs(diagnostics.energy_drift)
        )
        if step in sample_set:
            moments = moments_35_from_qmc(velocities, weights)
            histories.append(moments)
            metrics.append(
                _metrics(moments, tail_nodes=velocities, tail_weights=weights)
            )
        if step == steps or step % max(steps // 8, 1) == 0:
            _progress("full_fp_qmc", replicate, step, steps)
    return {
        "replicate": replicate,
        "histories": np.asarray(histories),
        "metrics": np.asarray(metrics),
        "modes": np.ones(len(histories), dtype=int),
        "projection_relative_residual": projection.relative_moment_residual,
        "minimum_probability": projection.minimum_probability,
        "maximum_momentum_drift": maximum_momentum_drift,
        "maximum_energy_drift": maximum_energy_drift,
        "elapsed_seconds": time.perf_counter() - start,
        "blocked_activations": 0,
        "transitions": [],
    }


def _run_adaptive_replicate(task: tuple) -> dict[str, object]:
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
        sensor_every,
    ) = task
    candidate, target, projection = positive_microstate_from_components(
        components,
        points_per_component=points_per_component,
        seed=seed,
        provenance=f"stage26-known-four-delta-initial-state-{replicate}",
    )
    policy = ActivationHysteresis()
    adaptive = initialize_adaptive_tail_memory(
        target,
        candidate_microstate=candidate,
        hysteresis=policy,
        noise_seed=seed + 1_000_003,
        tau=tau,
        prandtl=prandtl,
    )

    def sampled_metrics() -> np.ndarray:
        if adaptive.microstate is not None:
            return _metrics(
                adaptive.moments,
                tail_nodes=adaptive.microstate.velocities,
                tail_weights=adaptive.microstate.weights,
            )
        return _algebraic_metrics(adaptive.moments, "stage9_mixture")

    histories = [adaptive.moments.copy()]
    metrics = [sampled_metrics()]
    modes = [int(adaptive.mode == "micro")]
    blocked = 0
    blocked_releases = 0
    transitions: list[dict[str, object]] = []
    minimum_margin = float(realizability_margin_35(adaptive.moments))
    start = time.perf_counter()
    sample_set = set(sample_steps[1:])
    for step in range(1, steps + 1):
        adaptive, diagnostics = adaptive_tail_memory_fp_step(
            adaptive,
            dt,
            tau,
            hysteresis=policy,
            prandtl=prandtl,
            sensor_interval_steps=sensor_every,
            causal_reactivation_available=False,
        )
        blocked += int(diagnostics.activation_blocked)
        blocked_releases += int(
            diagnostics.release_blocked_without_causal_donor
        )
        minimum_margin = min(minimum_margin, diagnostics.realizability_margin)
        if diagnostics.transition not in ("macro->macro", "micro->micro"):
            transitions.append(
                {
                    "step": step,
                    "time_over_tau": step * dt / tau,
                    "transition": diagnostics.transition,
                }
            )
        if step in sample_set:
            histories.append(adaptive.moments.copy())
            metrics.append(sampled_metrics())
            modes.append(int(adaptive.mode == "micro"))
        if step == steps or step % max(steps // 8, 1) == 0:
            _progress("adaptive_memory", replicate, step, steps)
    return {
        "replicate": replicate,
        "histories": np.asarray(histories),
        "metrics": np.asarray(metrics),
        "modes": np.asarray(modes, dtype=int),
        "projection_relative_residual": projection.relative_moment_residual,
        "minimum_probability": projection.minimum_probability,
        "maximum_momentum_drift": 0.0,
        "maximum_energy_drift": 0.0,
        "minimum_realizability_margin": minimum_margin,
        "elapsed_seconds": time.perf_counter() - start,
        "blocked_activations": blocked,
        "blocked_releases_without_causal_donor": blocked_releases,
        "transitions": transitions,
        "final_mode": adaptive.mode,
        "tail_ambiguous_final": adaptive.tail_ambiguous,
    }


def _run_deterministic(
    method: str,
    target: np.ndarray,
    *,
    dt: float,
    steps: int,
    sample_steps: tuple[int, ...],
    tau: float,
    prandtl: float,
) -> dict[str, object]:
    if method == "stage9_mixture":
        stepper: Callable = finite_gaussian_mixture_fp_step
    elif method == "grad_gqmom":
        stepper = grad_hyqmom_fp_step
    else:  # pragma: no cover - protected by caller
        raise ValueError(f"unsupported deterministic method {method}")
    moments = target.copy()
    histories = [moments.copy()]
    metrics = [_algebraic_metrics(moments, method)]
    minimum_margin = float(realizability_margin_35(moments))
    minimum_limiter = 1.0
    start = time.perf_counter()
    sample_set = set(sample_steps[1:])
    for step in range(1, steps + 1):
        moments, diagnostics = stepper(
            moments, dt, tau, prandtl=prandtl
        )
        minimum_margin = min(minimum_margin, realizability_margin_35(moments))
        if method == "grad_gqmom":
            minimum_limiter = min(minimum_limiter, diagnostics.limiter_fraction)
        if step in sample_set:
            histories.append(moments.copy())
            metrics.append(_algebraic_metrics(moments, method))
        if step == steps or step % max(steps // 8, 1) == 0:
            _progress(method, 0, step, steps)
    return {
        "replicate": 0,
        "histories": np.asarray(histories),
        "metrics": np.asarray(metrics),
        "modes": np.zeros(len(histories), dtype=int),
        "projection_relative_residual": 0.0,
        "minimum_probability": 0.0,
        "maximum_momentum_drift": 0.0,
        "maximum_energy_drift": 0.0,
        "minimum_realizability_margin": minimum_margin,
        "minimum_limiter_fraction": minimum_limiter,
        "elapsed_seconds": time.perf_counter() - start,
        "blocked_activations": 0,
        "transitions": [],
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
    if args.dt <= 0.0 or args.final_time <= 0.0 or args.tau <= 0.0:
        raise ValueError("dt, final_time, and tau must be positive")
    steps = int(round(args.final_time / args.dt))
    if not np.isclose(steps * args.dt, args.final_time, rtol=0.0, atol=1.0e-12):
        raise ValueError("final_time must be an integer multiple of dt")
    if args.sample_every < 1 or args.sensor_every < 1:
        raise ValueError("sampling and sensor intervals must be positive")
    if args.replicates < 1 or args.workers < 1:
        raise ValueError("replicates and workers must be positive")
    sample_steps = _sample_steps(steps, args.sample_every)
    state = regularized_four_delta_state(
        energy_trace=args.energy_trace,
        regularization_fraction=args.regularization_fraction,
        rotation_degrees=args.rotation_degrees,
    )
    reading = kinetic_activation_sensor(
        state.moments, tau=args.tau, prandtl=args.prandtl
    )
    args.output.mkdir(parents=True, exist_ok=True)
    failure_path = args.output / f"stage26_{args.method}_failure.json"
    try:
        if args.method in ("full_fp_qmc", "adaptive_memory"):
            worker = (
                _run_qmc_replicate
                if args.method == "full_fp_qmc"
                else _run_adaptive_replicate
            )
            tasks = []
            method_seed = args.seed + (
                91_427_661 if args.method == "adaptive_memory" else 0
            )
            for replicate in range(args.replicates):
                common = (
                    state.components,
                    replicate,
                    args.points_per_component,
                    args.dt,
                    steps,
                    sample_steps,
                    args.tau,
                    args.prandtl,
                    method_seed + 15_485_863 * replicate,
                )
                tasks.append(common if args.method == "full_fp_qmc" else (*common, args.sensor_every))
            with ProcessPoolExecutor(
                max_workers=min(args.workers, args.replicates)
            ) as executor:
                results = list(executor.map(worker, tasks))
        else:
            results = [
                _run_deterministic(
                    args.method,
                    state.moments,
                    dt=args.dt,
                    steps=steps,
                    sample_steps=sample_steps,
                    tau=args.tau,
                    prandtl=args.prandtl,
                )
            ]
    except Exception as error:
        failure = {
            "schema": "riemann35-stage26-method-failure-v1",
            "method": args.method,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise

    histories = np.asarray([result["histories"] for result in results])
    metric_histories = np.asarray([result["metrics"] for result in results])
    modes = np.asarray([result["modes"] for result in results])
    times = np.asarray(sample_steps, dtype=float) * args.dt / args.tau
    archive_path = args.output / f"stage26_{args.method}.npz"
    np.savez_compressed(
        archive_path,
        times=times,
        histories=histories,
        metrics=metric_histories,
        metric_names=np.asarray(METRIC_NAMES),
        modes=modes,
        initial_weights=state.weights,
        initial_centers=state.centers,
    )
    summary = {
        "schema": "riemann35-stage26-four-delta-method-v1",
        "method": args.method,
        "controls": {
            "dt_over_tau": args.dt / args.tau,
            "final_time_over_tau": args.final_time / args.tau,
            "steps": steps,
            "sample_every": args.sample_every,
            "sensor_every": args.sensor_every,
            "replicates": len(results),
            "points_per_component": (
                args.points_per_component
                if args.method in ("full_fp_qmc", "adaptive_memory")
                else None
            ),
            "regularization_fraction": args.regularization_fraction,
            "component_variance": state.component_variance,
            "rotation_degrees": args.rotation_degrees,
            "seed_base": args.seed
            + (91_427_661 if args.method == "adaptive_memory" else 0),
            "energy_trace_definition": "(M200+M020+M002)/M000",
        },
        "initial_constraints": {
            "mass_error": state.mass_error,
            "momentum_norm": state.momentum_norm,
            "energy_trace_error": state.energy_trace_error,
            "central_third_norm": state.central_third_norm,
            "weights": state.weights,
            "centers": state.centers,
        },
        "initial_sensor": {
            "source_disagreement": reading.fourth_source_disagreement,
            "tail_disagreement": reading.tail_disagreement,
            "standardized_skewness_norm": reading.standardized_skewness_norm,
            "reconstruction_failure": reading.reconstruction_failure,
            "requests_activation": ActivationHysteresis().requests_activation(reading),
            "stage9_status": reading.stage9_status,
            "grad_status": reading.grad_status,
        },
        "replicate_diagnostics": [
            {
                key: value
                for key, value in result.items()
                if key not in ("histories", "metrics", "modes")
            }
            for result in results
        ],
        "minimum_realizability_margin": float(
            np.min(metric_histories[..., METRIC_NAMES.index("realizability_margin")])
        ),
        "maximum_mass_error": float(
            np.max(np.abs(metric_histories[..., METRIC_NAMES.index("rho")] - 1.0))
        ),
        "maximum_momentum_norm": float(
            np.max(metric_histories[..., METRIC_NAMES.index("momentum_norm")])
        ),
        "maximum_energy_trace_error": float(
            np.max(
                np.abs(
                    metric_histories[..., METRIC_NAMES.index("energy_trace")]
                    - args.energy_trace
                )
            )
        ),
        "mean_micro_active_fraction": float(np.mean(modes)),
        "total_elapsed_seconds": float(
            sum(float(result["elapsed_seconds"]) for result in results)
        ),
        "archive": str(archive_path),
    }
    (args.output / f"stage26_{args.method}_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    if failure_path.exists():
        failure_path.unlink()
    print(json.dumps(_jsonable(summary), indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()

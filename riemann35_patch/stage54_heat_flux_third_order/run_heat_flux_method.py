#!/usr/bin/env python3
"""Run one branch of the Stage-54 homogeneous heat-flux audit.

The initial state is Rodney Fox's regularized four-population construction,
rotated out of the coordinate plane.  The rotation preserves every invariant
while making all ten independent components of the symmetric third-order
central-moment tensor nonzero.  This prevents a symmetry-zero component from
being mistaken for evidence of accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    HYQMOM_35_INDICES,
    finite_gaussian_mixture_fp_step,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
    realizability_margin_35,
    regularized_four_delta_state,
)
from hyqmom_fp.moments import central_moment  # noqa: E402
from riemann35_patch.stage26.run_four_delta_method import (  # noqa: E402
    _algebraic_metrics_and_diagnostics,
    _progress,
    _run_adaptive_replicate,
    _run_deterministic,
    _run_qmc_replicate,
)


METHODS = (
    "qmc_base",
    "qmc_node_refined",
    "qmc_time_refined",
    "proposed_hyqmom35",
    "grad_comparator",
    "positive_tail_memory",
)
THIRD_INDICES = tuple(index for index in HYQMOM_35_INDICES if sum(index) == 3)
POSITION = {index: offset for offset, index in enumerate(HYQMOM_35_INDICES)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=float, default=2.5e-2)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--points-per-component", type=int, default=8192)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20_260_821)
    parser.add_argument("--regularization-fraction", type=float, default=0.03)
    parser.add_argument("--planar-rotation-degrees", type=float, default=17.0)
    parser.add_argument("--tilt-y-degrees", type=float, default=29.0)
    parser.add_argument("--tilt-x-degrees", type=float, default=41.0)
    parser.add_argument("--sensor-interval", type=float, default=2.5e-2)
    return parser.parse_args()


def _rotation_x(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
    )


def _rotation_y(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.asarray(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )


def oblique_heat_flux_state(
    *,
    regularization_fraction: float = 0.03,
    planar_rotation_degrees: float = 17.0,
    tilt_y_degrees: float = 29.0,
    tilt_x_degrees: float = 41.0,
) -> dict[str, object]:
    """Return the fixed three-dimensional skew state used by Stage 54."""

    base = regularized_four_delta_state(
        energy_trace=1.0,
        regularization_fraction=regularization_fraction,
        rotation_degrees=planar_rotation_degrees,
    )
    rotation = _rotation_x(tilt_x_degrees) @ _rotation_y(tilt_y_degrees)
    components = tuple(
        (
            float(weight),
            rotation @ np.asarray(mean, dtype=float),
            rotation @ np.asarray(covariance, dtype=float) @ rotation.T,
        )
        for weight, mean, covariance in base.components
    )
    moments = mixture_of_gaussians_moments_35(components)
    macro = macroscopic_state(moments)
    third = np.asarray([central_moment(moments, index) for index in THIRD_INDICES])
    energy = float(
        sum(moments[POSITION[index]] for index in ((2, 0, 0), (0, 2, 0), (0, 0, 2)))
    )
    return {
        "components": components,
        "moments": moments,
        "rotation": rotation,
        "mass_error": abs(float(moments[POSITION[(0, 0, 0)]]) - 1.0),
        "momentum_norm": float(np.linalg.norm(macro.rho * macro.velocity)),
        "energy_trace_error": abs(energy - 1.0),
        "heat_flux": np.asarray(macro.heat_flux),
        "third_components": third,
        "third_norm": float(np.linalg.norm(third)),
    }


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    value = int(round(numerator / denominator))
    if value < 1 or not np.isclose(value * denominator, numerator, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} must be a positive integer multiple of dt")
    return value


def _invariant_diagnostics(histories: np.ndarray) -> dict[str, float]:
    mass = histories[..., POSITION[(0, 0, 0)]]
    momentum = histories[..., [POSITION[(1, 0, 0)], POSITION[(0, 1, 0)], POSITION[(0, 0, 1)]]]
    energy = sum(
        histories[..., POSITION[index]]
        for index in ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    )
    return {
        "maximum_mass_drift": float(np.max(np.abs(mass - mass[..., :1]))),
        "maximum_momentum_drift": float(np.max(np.linalg.norm(momentum - momentum[..., :1, :], axis=-1))),
        "maximum_energy_trace_drift": float(np.max(np.abs(energy - energy[..., :1]))),
        "minimum_H2_margin": float(
            min(realizability_margin_35(history) for replicate in histories for history in replicate)
        ),
    }


def _run_unclipped_proposed(
    target: np.ndarray,
    *,
    dt: float,
    steps: int,
    sample_steps: tuple[int, ...],
    tau: float,
    prandtl: float,
) -> dict[str, object]:
    """Advance the proposed map with the same unclipped operator as QMC."""

    moments = target.copy()
    histories = [moments.copy()]
    initial = _algebraic_metrics_and_diagnostics(moments, "stage9_mixture")
    metrics = [initial[0]]
    minimum_quadrature_weight = initial[1]
    maximum_negative_weight_count = initial[2]
    maximum_negative_mass_fraction = initial[3]
    minimum_even_tail_moment = initial[4]
    minimum_margin = float(realizability_margin_35(moments))
    sample_set = set(sample_steps[1:])
    start = time.perf_counter()
    for step in range(1, steps + 1):
        moments, _ = finite_gaussian_mixture_fp_step(
            moments,
            dt,
            tau,
            prandtl=prandtl,
            speed_cap=np.inf,
        )
        minimum_margin = min(minimum_margin, realizability_margin_35(moments))
        if step in sample_set:
            histories.append(moments.copy())
            sampled = _algebraic_metrics_and_diagnostics(moments, "stage9_mixture")
            metrics.append(sampled[0])
            minimum_quadrature_weight = min(minimum_quadrature_weight, sampled[1])
            maximum_negative_weight_count = max(maximum_negative_weight_count, sampled[2])
            maximum_negative_mass_fraction = max(maximum_negative_mass_fraction, sampled[3])
            minimum_even_tail_moment = min(minimum_even_tail_moment, sampled[4])
        if step == steps or step % max(steps // 8, 1) == 0:
            _progress("stage54_unclipped_proposed", 0, step, steps)
    return {
        "replicate": 0,
        "histories": np.asarray(histories),
        "metrics": np.asarray(metrics),
        "modes": np.zeros(len(histories), dtype=int),
        "projection_relative_residual": 0.0,
        "minimum_probability": 0.0,
        "minimum_quadrature_weight": minimum_quadrature_weight,
        "maximum_negative_weight_count": maximum_negative_weight_count,
        "maximum_negative_mass_fraction": maximum_negative_mass_fraction,
        "minimum_even_tail_moment": minimum_even_tail_moment,
        "maximum_momentum_drift": 0.0,
        "maximum_energy_drift": 0.0,
        "minimum_realizability_margin": minimum_margin,
        "elapsed_seconds": time.perf_counter() - start,
        "blocked_activations": 0,
        "transitions": [],
        "speed_cap": "infinity",
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
        raise ValueError("dt, final-time, and tau must be positive")
    if args.replicates < 1 or args.workers < 1:
        raise ValueError("replicates and workers must be positive")
    steps = _integer_ratio(args.final_time, args.dt, "final-time")
    sample_every = _integer_ratio(args.sample_interval, args.dt, "sample-interval")
    sensor_every = _integer_ratio(args.sensor_interval, args.dt, "sensor-interval")
    sample_steps = tuple([0, *range(sample_every, steps + 1, sample_every)])
    if sample_steps[-1] != steps:
        sample_steps = (*sample_steps, steps)

    state = oblique_heat_flux_state(
        regularization_fraction=args.regularization_fraction,
        planar_rotation_degrees=args.planar_rotation_degrees,
        tilt_y_degrees=args.tilt_y_degrees,
        tilt_x_degrees=args.tilt_x_degrees,
    )
    components = state["components"]
    target = np.asarray(state["moments"])
    args.output.mkdir(parents=True, exist_ok=True)
    failure_path = args.output / f"stage54_{args.method}_failure.json"
    print(
        f"[stage54] method={args.method} steps={steps} nodes/component={args.points_per_component} "
        f"replicates={args.replicates} started={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        flush=True,
    )

    try:
        if args.method.startswith("qmc_"):
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
        elif args.method == "positive_tail_memory":
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
                    args.seed + 91_427_661 + 15_485_863 * replicate,
                    sensor_every,
                )
                for replicate in range(args.replicates)
            ]
            with ProcessPoolExecutor(max_workers=min(args.workers, args.replicates)) as executor:
                results = list(executor.map(_run_adaptive_replicate, tasks))
        elif args.method == "proposed_hyqmom35":
            results = [
                _run_unclipped_proposed(
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
                _run_deterministic(
                    "grad_gqmom",
                    target,
                    dt=args.dt,
                    steps=steps,
                    sample_steps=sample_steps,
                    tau=args.tau,
                    prandtl=args.prandtl,
                )
            ]
    except Exception as error:
        failure = {
            "schema": "riemann35-stage54-method-failure-v1",
            "method": args.method,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise

    histories = np.asarray([result["histories"] for result in results])
    times = np.asarray(sample_steps, dtype=float) * args.dt / args.tau
    np.savez_compressed(
        args.output / f"stage54_{args.method}.npz",
        times=times,
        histories=histories,
        third_indices=np.asarray(THIRD_INDICES, dtype=int),
    )
    summary = {
        "schema": "riemann35-stage54-heat-flux-method-v1",
        "method": args.method,
        "controls": {
            "dt_over_tau": args.dt / args.tau,
            "final_time_over_tau": args.final_time / args.tau,
            "sample_interval_over_tau": args.sample_interval / args.tau,
            "points_per_component": args.points_per_component if args.method.startswith("qmc_") or args.method == "positive_tail_memory" else None,
            "replicates": len(results),
            "prandtl": args.prandtl,
            "speed_cap": "infinity",
            "regularization_fraction": args.regularization_fraction,
            "planar_rotation_degrees": args.planar_rotation_degrees,
            "tilt_y_degrees": args.tilt_y_degrees,
            "tilt_x_degrees": args.tilt_x_degrees,
            "seed_base": args.seed,
        },
        "initial_state": {
            "mass_error": state["mass_error"],
            "momentum_norm": state["momentum_norm"],
            "energy_trace_error": state["energy_trace_error"],
            "heat_flux": state["heat_flux"],
            "heat_flux_norm": np.linalg.norm(state["heat_flux"]),
            "third_indices": THIRD_INDICES,
            "third_components": state["third_components"],
            "third_norm": state["third_norm"],
            "rotation_matrix": state["rotation"],
        },
        "invariants": _invariant_diagnostics(histories),
        "minimum_quadrature_weight": float(min(result["minimum_quadrature_weight"] for result in results)),
        "maximum_negative_weight_count": int(max(result["maximum_negative_weight_count"] for result in results)),
        "maximum_negative_mass_fraction": float(max(result["maximum_negative_mass_fraction"] for result in results)),
        "minimum_even_tail_moment": float(min(result["minimum_even_tail_moment"] for result in results)),
        "replicate_diagnostics": [
            {key: value for key, value in result.items() if key not in ("histories", "metrics", "modes")}
            for result in results
        ],
    }
    (args.output / f"stage54_{args.method}_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    if failure_path.exists():
        failure_path.unlink()
    print(json.dumps(_jsonable(summary), indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()

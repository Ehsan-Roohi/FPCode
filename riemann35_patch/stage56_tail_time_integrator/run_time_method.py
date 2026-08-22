#!/usr/bin/env python3
"""Run one Stage-56 time-integration qualification method."""

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

from hyqmom_fp import DynamicHighOrderState, realizability_margin_35  # noqa: E402
from riemann35_patch.stage54_heat_flux_third_order.run_heat_flux_method import (  # noqa: E402
    oblique_heat_flux_state,
)
from riemann35_patch.stage55_closure_source_audit.run_closure_method import (  # noqa: E402
    TAIL_INDICES,
    _invariants,
    _run_qmc_replicate,
    exact_initial_tail,
)
from riemann35_patch.stage56_tail_time_integrator.integrator import (  # noqa: E402
    legacy_lie_projected_step,
    strang_exact_projected_step,
)


METHODS = (
    "qmc_reference",
    "legacy_lie_h0",
    "strang_h0",
    "strang_h1",
    "strang_h2",
    "strang_h3",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=float, default=2.5e-2)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--points-per-component", type=int, default=32768)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20_260_822)
    parser.add_argument("--tail-relaxation-time", type=float, default=1.0e-2)
    parser.add_argument("--quadrature-nodes", type=int, default=5)
    return parser.parse_args()


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    value = int(round(numerator / denominator))
    if value < 1 or not np.isclose(value * denominator, numerator, rtol=0.0, atol=2.0e-13):
        raise ValueError(f"{name} must be a positive integer multiple of dt")
    return value


def _run_candidate(
    target: np.ndarray,
    components: tuple,
    *,
    dt: float,
    steps: int,
    sample_steps: tuple[int, ...],
    tau: float,
    prandtl: float,
    integrator: str,
    tail_relaxation_time: float,
    quadrature_nodes: int,
) -> dict[str, object]:
    state = DynamicHighOrderState(
        moments=target.copy(),
        tail_moments=exact_initial_tail(components),
        maximum_order=6,
    )
    histories = [state.moments.copy()]
    tails = [state.tail_moments.copy()]
    sample_set = set(sample_steps[1:])
    minimum_margin = float(realizability_margin_35(state.moments))
    minimum_limiter = 1.0
    minimum_weight = float("inf")
    maximum_negative_mass = 0.0
    maximum_target_residual = 0.0
    maximum_projection_distance = 0.0
    start = time.perf_counter()
    stepper = legacy_lie_projected_step if integrator == "legacy_lie" else strang_exact_projected_step
    for step in range(1, steps + 1):
        state, diagnostics = stepper(
            state,
            dt,
            tau,
            prandtl=prandtl,
            tail_relaxation_time=tail_relaxation_time,
            quadrature_nodes=quadrature_nodes,
        )
        minimum_margin = min(minimum_margin, diagnostics.realizability_margin)
        minimum_limiter = min(minimum_limiter, diagnostics.limiter_fraction)
        minimum_weight = min(minimum_weight, diagnostics.minimum_weight)
        maximum_negative_mass = max(
            maximum_negative_mass, diagnostics.maximum_negative_mass_fraction
        )
        maximum_target_residual = max(
            maximum_target_residual, diagnostics.maximum_target_relative_residual
        )
        maximum_projection_distance = max(
            maximum_projection_distance, diagnostics.maximum_projection_distance
        )
        if step in sample_set:
            histories.append(state.moments.copy())
            tails.append(state.tail_moments.copy())
        if step == steps or step % max(steps // 8, 1) == 0:
            print(
                f"[stage56] integrator={integrator} step={step}/{steps} "
                f"limiter={diagnostics.limiter_fraction:.3e}",
                flush=True,
            )
    return {
        "histories": np.asarray(histories),
        "sources": np.empty((0, 35)),
        "tails": np.asarray(tails),
        "minimum_weight": float(minimum_weight),
        "minimum_H2_margin": float(minimum_margin),
        "minimum_limiter": float(minimum_limiter),
        "maximum_negative_mass_fraction": float(maximum_negative_mass),
        "maximum_target_relative_residual": float(maximum_target_residual),
        "maximum_tail_projection_distance": float(maximum_projection_distance),
        "persistent_state_scalars": 35 + len(TAIL_INDICES),
        "integrator": integrator,
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
    if min(args.dt, args.final_time, args.sample_interval, args.tau, args.tail_relaxation_time) <= 0.0:
        raise ValueError("all time scales must be positive")
    steps = _integer_ratio(args.final_time, args.dt, "final-time")
    sample_every = _integer_ratio(args.sample_interval, args.dt, "sample-interval")
    sample_steps = tuple([0, *range(sample_every, steps + 1, sample_every)])
    if sample_steps[-1] != steps:
        sample_steps = (*sample_steps, steps)
    initial = oblique_heat_flux_state()
    components = tuple(initial["components"])
    target = np.asarray(initial["moments"], dtype=float)
    args.output.mkdir(parents=True, exist_ok=True)
    failure_path = args.output / f"stage56_{args.method}_failure.json"
    print(
        f"[stage56] method={args.method} dt={args.dt} steps={steps} "
        f"started={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
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
            integrator = "positive_full_qmc"
        else:
            integrator = "legacy_lie" if args.method == "legacy_lie_h0" else "strang_exact"
            results = [
                _run_candidate(
                    target,
                    components,
                    dt=args.dt,
                    steps=steps,
                    sample_steps=sample_steps,
                    tau=args.tau,
                    prandtl=args.prandtl,
                    integrator=integrator,
                    tail_relaxation_time=args.tail_relaxation_time,
                    quadrature_nodes=args.quadrature_nodes,
                )
            ]
    except Exception as error:
        failure = {
            "schema": "riemann35-stage56-method-failure-v1",
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
        args.output / f"stage56_{args.method}.npz",
        times=times,
        histories=histories,
        sources=sources,
        tails=tails,
        tail_indices=np.asarray(TAIL_INDICES, dtype=int),
    )
    summary = {
        "schema": "riemann35-stage56-time-method-v1",
        "method": args.method,
        "integrator": integrator,
        "controls": {
            "dt_over_tau": args.dt / args.tau,
            "final_time_over_tau": args.final_time / args.tau,
            "sample_interval_over_tau": args.sample_interval / args.tau,
            "points_per_component": args.points_per_component if args.method == "qmc_reference" else None,
            "replicates": len(results),
            "prandtl": args.prandtl,
            "tail_relaxation_time_over_tau": None if args.method == "qmc_reference" else args.tail_relaxation_time / args.tau,
            "quadrature_nodes": None if args.method == "qmc_reference" else args.quadrature_nodes,
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
    (args.output / f"stage56_{args.method}_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    if failure_path.exists():
        failure_path.unlink()
    print(json.dumps(_jsonable(summary), indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()

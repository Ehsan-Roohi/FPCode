#!/usr/bin/env python3
"""Run one Stage-57 positive persistent-population method."""

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

from hyqmom_fp import realizability_margin_35  # noqa: E402
from hyqmom_fp.two_population import _gauss_hermite_mixture_nodes  # noqa: E402
from riemann35_patch.stage54_heat_flux_third_order.run_heat_flux_method import (  # noqa: E402
    oblique_heat_flux_state,
)
from riemann35_patch.stage55_closure_source_audit.run_closure_method import (  # noqa: E402
    _direct_node_source,
    _invariants,
    _raw_tail_from_nodes,
    _run_qmc_replicate,
    central_source_components,
)
from riemann35_patch.stage56_tail_time_integrator.run_time_method import (  # noqa: E402
    _run_candidate as _run_stage56_candidate,
)
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import (  # noqa: E402
    initialize_persistent_gaussian_mixture,
    persistent_gaussian_mixture_fp_step,
    persistent_gaussian_mixture_moments,
    stored_scalar_count,
)


METHODS = (
    "qmc_reference",
    "stage56_strang_control",
    "persistent4_h0",
    "persistent4_h1",
    "persistent4_h2",
    "persistent4_h3",
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
    parser.add_argument("--seed", type=int, default=20_260_823)
    parser.add_argument("--quadrature-nodes", type=int, default=5)
    return parser.parse_args()


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    value = int(round(numerator / denominator))
    if value < 1 or not np.isclose(
        value * denominator, numerator, rtol=0.0, atol=2.0e-13
    ):
        raise ValueError(f"{name} must be a positive integer multiple of dt")
    return value


def _initial_source_audit(
    components: tuple,
    moments: np.ndarray,
    *,
    tau: float,
    prandtl: float,
    quadrature_nodes: int,
) -> dict[str, float]:
    state = initialize_persistent_gaussian_mixture(components)
    coarse_weights, coarse_nodes = _gauss_hermite_mixture_nodes(
        state.probabilities,
        state.means,
        state.covariances,
        state.rho,
        quadrature_nodes,
    )
    fine_weights, fine_nodes = _gauss_hermite_mixture_nodes(
        state.probabilities,
        state.means,
        state.covariances,
        state.rho,
        max(quadrature_nodes + 2, 7),
    )
    coarse_tail = _raw_tail_from_nodes(coarse_nodes, coarse_weights)
    fine_tail = _raw_tail_from_nodes(fine_nodes, fine_weights)
    coarse_source = _direct_node_source(
        moments, coarse_nodes, coarse_weights, tau=tau, prandtl=prandtl
    )
    fine_source = _direct_node_source(
        moments, fine_nodes, fine_weights, tau=tau, prandtl=prandtl
    )
    coarse_third = central_source_components(moments, coarse_source)
    fine_third = central_source_components(moments, fine_source)
    return {
        "initial_moment_relative_residual": float(
            np.linalg.norm(persistent_gaussian_mixture_moments(state) - moments)
            / max(np.linalg.norm(moments), 1.0e-14)
        ),
        "initial_tail_relative_error": float(
            np.linalg.norm(coarse_tail - fine_tail)
            / max(np.linalg.norm(fine_tail), 1.0e-14)
        ),
        "initial_third_source_relative_error": float(
            np.linalg.norm(coarse_third - fine_third)
            / max(np.linalg.norm(fine_third), 1.0e-14)
        ),
        "initial_minimum_quadrature_weight": float(np.min(coarse_weights)),
    }


def _run_persistent_candidate(
    components: tuple,
    *,
    dt: float,
    steps: int,
    sample_steps: tuple[int, ...],
    tau: float,
    prandtl: float,
    quadrature_nodes: int,
) -> dict[str, object]:
    state = initialize_persistent_gaussian_mixture(components)
    moments = persistent_gaussian_mixture_moments(state)
    histories = [moments.copy()]
    sample_set = set(sample_steps[1:])
    minimum_weight = float("inf")
    minimum_covariance = float("inf")
    minimum_margin = float(realizability_margin_35(moments))
    minimum_projection_fraction = 1.0
    maximum_projection_residual = 0.0
    start = time.perf_counter()
    for step in range(1, steps + 1):
        state, moments, diagnostics = persistent_gaussian_mixture_fp_step(
            state,
            dt,
            tau,
            prandtl=prandtl,
            quadrature_nodes=quadrature_nodes,
            enforce_heat_flux_rate=True,
        )
        minimum_weight = min(minimum_weight, diagnostics.minimum_quadrature_weight)
        minimum_covariance = min(
            minimum_covariance, diagnostics.minimum_covariance_eigenvalue
        )
        minimum_margin = min(minimum_margin, diagnostics.realizability_margin)
        minimum_projection_fraction = min(
            minimum_projection_fraction, diagnostics.heat_flux_projection_fraction
        )
        maximum_projection_residual = max(
            maximum_projection_residual, diagnostics.heat_flux_projection_residual
        )
        if step in sample_set:
            histories.append(moments.copy())
        if step == steps or step % max(steps // 8, 1) == 0:
            print(
                f"[stage57] persistent4 step={step}/{steps} "
                f"projection={diagnostics.heat_flux_projection_fraction:.3e}",
                flush=True,
            )
    audit = _initial_source_audit(
        components,
        histories[0],
        tau=tau,
        prandtl=prandtl,
        quadrature_nodes=quadrature_nodes,
    )
    return {
        "histories": np.asarray(histories),
        "minimum_weight": float(minimum_weight),
        "minimum_covariance_eigenvalue": float(minimum_covariance),
        "minimum_H2_margin": float(minimum_margin),
        "minimum_projection_fraction": float(minimum_projection_fraction),
        "maximum_projection_residual": float(maximum_projection_residual),
        "maximum_negative_mass_fraction": 0.0,
        "persistent_state_scalars": stored_scalar_count(state),
        "integrator": "persistent-four-Gaussian-with-physical-heat-flux-projection",
        "elapsed_seconds": time.perf_counter() - start,
        **audit,
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
    if min(args.dt, args.final_time, args.sample_interval, args.tau) <= 0.0:
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
    failure_path = args.output / f"stage57_{args.method}_failure.json"
    print(
        f"[stage57] method={args.method} dt={args.dt} steps={steps} "
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
            with ProcessPoolExecutor(
                max_workers=min(args.workers, args.replicates)
            ) as executor:
                results = list(executor.map(_run_qmc_replicate, tasks))
            integrator = "positive_full_qmc"
        elif args.method == "stage56_strang_control":
            results = [
                _run_stage56_candidate(
                    target,
                    components,
                    dt=args.dt,
                    steps=steps,
                    sample_steps=sample_steps,
                    tau=args.tau,
                    prandtl=args.prandtl,
                    integrator="strang_exact",
                    tail_relaxation_time=1.0e-2,
                    quadrature_nodes=args.quadrature_nodes,
                )
            ]
            integrator = "stage56-exact-Strang-control"
        else:
            results = [
                _run_persistent_candidate(
                    components,
                    dt=args.dt,
                    steps=steps,
                    sample_steps=sample_steps,
                    tau=args.tau,
                    prandtl=args.prandtl,
                    quadrature_nodes=args.quadrature_nodes,
                )
            ]
            integrator = results[0]["integrator"]
    except Exception as error:
        failure = {
            "schema": "riemann35-stage57-method-failure-v1",
            "method": args.method,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise

    histories = np.asarray([item["histories"] for item in results])
    times = np.asarray(sample_steps, dtype=float) * args.dt / args.tau
    np.savez_compressed(
        args.output / f"stage57_{args.method}.npz",
        times=times,
        histories=histories,
    )
    summary = {
        "schema": "riemann35-stage57-persistent-method-v1",
        "method": args.method,
        "integrator": integrator,
        "controls": {
            "dt_over_tau": args.dt / args.tau,
            "final_time_over_tau": args.final_time / args.tau,
            "sample_interval_over_tau": args.sample_interval / args.tau,
            "points_per_component": (
                args.points_per_component if args.method == "qmc_reference" else None
            ),
            "replicates": len(results),
            "prandtl": args.prandtl,
            "quadrature_nodes_per_population": (
                args.quadrature_nodes if args.method != "qmc_reference" else None
            ),
        },
        "invariants": _invariants(histories),
        "minimum_weight": float(min(item.get("minimum_weight", 0.0) for item in results)),
        "minimum_H2_margin": float(min(item["minimum_H2_margin"] for item in results)),
        "persistent_state_scalars": int(
            max(item.get("persistent_state_scalars", 35) for item in results)
        ),
        "replicate_diagnostics": [
            {key: value for key, value in item.items() if key != "histories"}
            for item in results
        ],
    }
    (args.output / f"stage57_{args.method}_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    if failure_path.exists():
        failure_path.unlink()
    print(json.dumps(_jsonable(summary), indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()

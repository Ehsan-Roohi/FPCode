#!/usr/bin/env python3
"""Run one frozen Stage-56 time/node refinement candidate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import HYQMOM_35_INDICES  # noqa: E402
from riemann35_patch.stage54_heat_flux_third_order.run_heat_flux_method import (  # noqa: E402
    oblique_heat_flux_state,
)
from riemann35_patch.stage55_closure_source_audit.run_closure_method import (  # noqa: E402
    exact_initial_tail,
)
from riemann35_patch.stage56_time_consistent_closure.time_consistent import (  # noqa: E402
    h3_margin,
    pack_degree_six,
    time_consistent_degree_six_step,
)


METHODS = (
    "q4_dt2500",
    "q5_dt2500",
    "q5_dt1250",
    "q5_dt0625",
    "q5_dt03125",
    "q6_dt0625",
)
POSITION = {index: position for position, index in enumerate(HYQMOM_35_INDICES)}
INVARIANT_INDICES = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)
ENERGY_INDICES = ((2, 0, 0), (0, 2, 0), (0, 0, 2))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--quadrature-nodes", type=int, required=True)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=float, default=0.025)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--tail-relaxation-time", type=float, default=0.01)
    parser.add_argument("--h3-floor", type=float, default=-1.0e-12)
    return parser.parse_args()


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    value = int(round(numerator / denominator))
    if value < 1 or not np.isclose(
        value * denominator,
        numerator,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"{name} must be a positive integer multiple of dt")
    return value


def _invariant_drift(histories: np.ndarray) -> dict[str, float]:
    mass = histories[:, POSITION[(0, 0, 0)]]
    momentum = histories[
        :,
        [
            POSITION[(1, 0, 0)],
            POSITION[(0, 1, 0)],
            POSITION[(0, 0, 1)],
        ],
    ]
    energy = sum(histories[:, POSITION[index]] for index in ENERGY_INDICES)
    return {
        "maximum_mass_drift": float(np.max(np.abs(mass - mass[0]))),
        "maximum_momentum_drift": float(
            np.max(np.linalg.norm(momentum - momentum[0], axis=-1))
        ),
        "maximum_energy_trace_drift": float(np.max(np.abs(energy - energy[0]))),
    }


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    args = arguments()
    if min(
        args.dt,
        args.final_time,
        args.sample_interval,
        args.tau,
        args.tail_relaxation_time,
    ) <= 0.0:
        raise ValueError("all time scales must be positive")
    steps = _integer_ratio(args.final_time, args.dt, "final-time")
    sample_every = _integer_ratio(args.sample_interval, args.dt, "sample-interval")
    sample_steps = tuple([0, *range(sample_every, steps + 1, sample_every)])
    if sample_steps[-1] != steps:
        sample_steps = (*sample_steps, steps)

    initial = oblique_heat_flux_state()
    moments = np.asarray(initial["moments"], dtype=float)
    tail = exact_initial_tail(tuple(initial["components"]))
    packed = pack_degree_six(moments, tail)
    initial_h3 = h3_margin(packed)
    histories = [moments.copy()]
    tails = [tail.copy()]
    h3_samples = [initial_h3]
    minimum_h3 = initial_h3
    minimum_nonlinear_limiter = 1.0
    minimum_projection_limiter = 1.0
    active_nonlinear_limiter_steps = 0
    active_projection_limiter_steps = 0
    minimum_projection_weight = float("inf")
    maximum_projection_residual = 0.0
    maximum_source_residual = 0.0
    maximum_source_norm = 0.0
    sample_set = set(sample_steps[1:])
    args.output.mkdir(parents=True, exist_ok=True)
    failure_path = args.output / f"stage56_{args.method}_failure.json"
    start = time.perf_counter()
    print(
        f"[stage56] method={args.method} dt={args.dt} q={args.quadrature_nodes} "
        f"steps={steps} initial_h3={initial_h3:.6e}",
        flush=True,
    )
    try:
        for step in range(1, steps + 1):
            packed, diagnostics = time_consistent_degree_six_step(
                packed,
                dt=args.dt,
                tau=args.tau,
                prandtl=args.prandtl,
                tail_relaxation_time=args.tail_relaxation_time,
                quadrature_nodes=args.quadrature_nodes,
                h3_floor=args.h3_floor,
            )
            minimum_h3 = min(minimum_h3, diagnostics.minimum_h3_margin)
            minimum_nonlinear_limiter = min(
                minimum_nonlinear_limiter,
                diagnostics.nonlinear_limiter,
            )
            minimum_projection_limiter = min(
                minimum_projection_limiter,
                diagnostics.projection_limiter,
            )
            active_nonlinear_limiter_steps += int(
                diagnostics.nonlinear_limiter < 1.0 - 1.0e-12
            )
            active_projection_limiter_steps += int(
                diagnostics.projection_limiter < 1.0 - 1.0e-12
            )
            minimum_projection_weight = min(
                minimum_projection_weight,
                diagnostics.minimum_projection_weight,
            )
            maximum_projection_residual = max(
                maximum_projection_residual,
                diagnostics.maximum_projection_moment_residual,
            )
            maximum_source_residual = max(
                maximum_source_residual,
                diagnostics.maximum_source_quadrature_residual,
            )
            maximum_source_norm = max(maximum_source_norm, diagnostics.source_norm)
            if step in sample_set:
                histories.append(packed[:35].copy())
                tails.append(packed[35:].copy())
                h3_samples.append(h3_margin(packed))
            if step == steps or step % max(steps // 8, 1) == 0:
                print(
                    f"[stage56] method={args.method} step={step}/{steps} "
                    f"h3={diagnostics.minimum_h3_margin:.3e} "
                    f"L_N={diagnostics.nonlinear_limiter:.3e} "
                    f"L_P={diagnostics.projection_limiter:.3e}",
                    flush=True,
                )
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

    histories_array = np.asarray(histories)
    tails_array = np.asarray(tails)
    times = np.asarray(sample_steps, dtype=float) * args.dt / args.tau
    np.savez_compressed(
        args.output / f"stage56_{args.method}.npz",
        times=times,
        histories=histories_array[None, ...],
        tails=tails_array[None, ...],
        h3_margins=np.asarray(h3_samples),
    )
    summary = {
        "schema": "riemann35-stage56-time-consistent-method-v1",
        "method": args.method,
        "controls": {
            "dt_over_tau": args.dt / args.tau,
            "final_time_over_tau": args.final_time / args.tau,
            "sample_interval_over_tau": args.sample_interval / args.tau,
            "prandtl": args.prandtl,
            "tail_relaxation_time_over_tau": args.tail_relaxation_time / args.tau,
            "quadrature_nodes": args.quadrature_nodes,
            "integrator": "P(dt/2)-OU_exact(dt/2)-SSPRK2_N(dt)-OU_exact(dt/2)-P(dt/2)",
            "persistent_state_scalars": 84,
        },
        "initial_h3_margin": initial_h3,
        "minimum_h3_margin": minimum_h3,
        "minimum_nonlinear_limiter": minimum_nonlinear_limiter,
        "minimum_projection_limiter": minimum_projection_limiter,
        "active_nonlinear_limiter_steps": active_nonlinear_limiter_steps,
        "active_projection_limiter_steps": active_projection_limiter_steps,
        "minimum_projection_weight": minimum_projection_weight,
        "maximum_projection_moment_residual": maximum_projection_residual,
        "maximum_source_quadrature_residual": maximum_source_residual,
        "maximum_source_norm": maximum_source_norm,
        "invariants": _invariant_drift(histories_array),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output / f"stage56_{args.method}_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    if failure_path.exists():
        failure_path.unlink()
    print(json.dumps(_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()

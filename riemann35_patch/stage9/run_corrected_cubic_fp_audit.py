#!/usr/bin/env python3
"""Stage-9 long-time audit of the corrected cubic-FP/HyQMOM coupling."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    finite_gaussian_mixture_fp_step,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
    realizability_margin_35,
    reconstruct_gaussian_mixture_quadrature,
)
from hyqmom_fp.moments import central_moment  # noqa: E402
from riemann35_patch.stage7.run_stage7 import case_components  # noqa: E402


CASES = ("symmetric", "asymmetric", "correlated", "leptokurtic", "rare_hot")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--t-final", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def dimensionless_shape(moments: np.ndarray) -> dict[str, object]:
    state = macroscopic_state(moments)
    excess = []
    for axis in range(3):
        index = [0, 0, 0]
        index[axis] = 4
        fourth = central_moment(moments, tuple(index)) / state.rho
        excess.append(float(fourth / state.covariance[axis, axis] ** 2 - 3.0))
    return {
        "excess_kurtosis_axes": excess,
        "heat_flux_norm": float(
            np.linalg.norm(state.heat_flux) / (state.rho * state.theta**1.5)
        ),
        "realizability_margin": realizability_margin_35(moments),
    }


def simulate(task: tuple[str, float, float]) -> tuple[str, float, dict[str, object], np.ndarray]:
    case, dt, final_time = task
    steps = int(round(final_time / dt))
    if not np.isclose(steps * dt, final_time):
        raise ValueError("final time must be an integer multiple of dt")
    moments = mixture_of_gaussians_moments_35(case_components(case))
    initial = moments.copy()
    initial_quadrature = reconstruct_gaussian_mixture_quadrature(initial)
    minimum_margin = realizability_margin_35(moments)
    maximum_residual = initial_quadrature.relative_moment_residual
    status = "REACHED_FINAL_TIME"
    failure = None
    for step in range(1, steps + 1):
        try:
            moments, diagnostics = finite_gaussian_mixture_fp_step(
                moments, dt, 1.0
            )
        except Exception as error:
            status = "FAILED"
            failure = {
                "step": step,
                "time": step * dt,
                "message": f"{type(error).__name__}: {error}",
            }
            break
        minimum_margin = min(minimum_margin, diagnostics.realizability_margin)
        maximum_residual = max(
            maximum_residual, diagnostics.reconstruction_relative_residual
        )
        if diagnostics.realizability_margin < -1.0e-10:
            status = "LEFT_REALIZABILITY_CONE"
            failure = {
                "step": step,
                "time": step * dt,
                "margin": diagnostics.realizability_margin,
            }
            break
    initial_state = macroscopic_state(initial)
    final_state = macroscopic_state(moments)
    result: dict[str, object] = {
        "status": status,
        "dt": dt,
        "requested_final_time": final_time,
        "reached_time": step * dt,
        "initial_shape": dimensionless_shape(initial),
        "final_shape": dimensionless_shape(moments),
        "initial_marginal_branches": [
            marginal.branch for marginal in initial_quadrature.marginals
        ],
        "initial_reconstruction_relative_residual": (
            initial_quadrature.relative_moment_residual
        ),
        "maximum_reconstruction_relative_residual": maximum_residual,
        "minimum_realizability_margin": minimum_margin,
        "mass_drift": abs(final_state.rho - initial_state.rho),
        "momentum_drift": float(
            np.linalg.norm(final_state.velocity - initial_state.velocity)
        ),
        "temperature_drift": abs(final_state.theta - initial_state.theta),
    }
    if failure is not None:
        result["failure"] = failure
    return case, dt, result, moments


def main() -> None:
    args = parse_arguments()
    if args.dt <= 0.0 or args.t_final <= 0.0 or args.workers <= 0:
        raise SystemExit("dt, final time, and worker count must be positive")
    refinement_cases = ("symmetric", "correlated")
    tasks = [(case, args.dt, args.t_final) for case in CASES]
    tasks += [
        (case, factor * args.dt, args.t_final)
        for case in refinement_cases
        for factor in (2.0, 0.5)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        raw_results = list(executor.map(simulate, tasks))

    indexed = {(case, dt): (record, moments) for case, dt, record, moments in raw_results}
    cases = {case: indexed[(case, args.dt)][0] for case in CASES}
    rare_hot_initial = np.asarray(
        cases["rare_hot"]["initial_shape"]["excess_kurtosis_axes"]
    )
    rare_hot_final = np.asarray(
        cases["rare_hot"]["final_shape"]["excess_kurtosis_axes"]
    )
    rare_hot_exact = rare_hot_initial * np.exp(-4.0 * args.t_final)
    rare_hot_analytic_check = {
        "reason": (
            "isotropic stress-free, zero-heat-flux state reduces exactly to OU"
        ),
        "exact_excess_kurtosis_axes": rare_hot_exact.tolist(),
        "computed_excess_kurtosis_axes": rare_hot_final.tolist(),
        "maximum_absolute_error": float(
            np.max(np.abs(rare_hot_final - rare_hot_exact))
        ),
    }
    refinement: dict[str, object] = {}
    for case in refinement_cases:
        coarse = indexed[(case, 2.0 * args.dt)][1]
        base = indexed[(case, args.dt)][1]
        fine = indexed[(case, 0.5 * args.dt)][1]
        refinement[case] = {
            "dt_values": [2.0 * args.dt, args.dt, 0.5 * args.dt],
            "relative_final_difference_coarse_to_base": float(
                np.linalg.norm(coarse - base) / max(np.linalg.norm(base), 1.0e-15)
            ),
            "relative_final_difference_base_to_fine": float(
                np.linalg.norm(base - fine) / max(np.linalg.norm(fine), 1.0e-15)
            ),
        }
    summary = {
        "schema": "riemann35-cubic-fp-stage9-corrected-v1",
        "coefficient_model": "physical 9x9 cubic-FP solve",
        "unresolved_residual_model": "exact OU moment propagation",
        "cases": cases,
        "rare_hot_exact_ou_check": rare_hot_analytic_check,
        "time_step_refinement": refinement,
        "all_cases_reached_final_time": all(
            record["status"] == "REACHED_FINAL_TIME" for record in cases.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for case, record in cases.items():
        print(
            f"{case:>12}: {record['status']} "
            f"margin={record['minimum_realizability_margin']:.3e} "
            f"q*={record['final_shape']['heat_flux_norm']:.3e}"
        )
    for case, record in refinement.items():
        print(
            f"refinement {case:>10}: "
            f"{record['relative_final_difference_coarse_to_base']:.3e} -> "
            f"{record['relative_final_difference_base_to_fine']:.3e}"
        )
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()

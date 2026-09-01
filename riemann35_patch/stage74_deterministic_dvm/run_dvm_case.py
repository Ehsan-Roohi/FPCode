#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyqmom_fp import DVMGrid, dvm_cubic_fp_step, initialize_diagonal_gaussian_mixture
from riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux import (
    central_third_components,
    irreducible_decomposition,
    symmetric_tensor,
)
from riemann35_patch.stage57_persistent_four_population.run_persistent_method import (
    _invariants,
    _run_persistent_candidate,
)
from riemann35_patch.stage71_harder_unseen.hard_cases import hard_case
from riemann35_patch.stage72_density_jacobian_fix.fixed_projection import (
    install_density_consistent_projection,
)

CASES = ("rare_beam_3d", "dense_hot_extreme", "dilute_broad")


def _ratio(a: float, b: float) -> int:
    n = int(round(a / b))
    if n < 1 or not np.isclose(n * b, a, rtol=0.0, atol=2.0e-13):
        raise ValueError("time interval must be an integer multiple of dt")
    return n


def _grid_for_case(components, cells: int, margin_sigma: float) -> DVMGrid:
    means = np.asarray([item[1] for item in components], dtype=float)
    covs = np.asarray([item[2] for item in components], dtype=float)
    if not np.allclose(covs, np.asarray([np.diag(np.diag(c)) for c in covs]), atol=1.0e-12):
        raise ValueError("Stage74 clean-room DVM gate is restricted to diagonal component covariances")
    sigmas = np.sqrt(np.diagonal(covs, axis1=1, axis2=2))
    lower = np.min(means - margin_sigma * sigmas, axis=0)
    upper = np.max(means + margin_sigma * sigmas, axis=0)
    padding = 0.02 * np.maximum(upper - lower, 1.0)
    lower -= padding
    upper += padding
    return DVMGrid(tuple(lower), tuple(upper), (cells, cells, cells))


def _sample_steps(final_time: float, sample_interval: float, dt: float) -> tuple[int, ...]:
    steps = _ratio(final_time, dt)
    every = _ratio(sample_interval, dt)
    out = tuple([0, *range(every, steps + 1, every)])
    return out if out[-1] == steps else (*out, steps)


def _run_dvm(components, *, cells: int, dt: float, final_time: float, sample_interval: float, tau: float, prandtl: float, margin_sigma: float):
    grid = _grid_for_case(components, cells, margin_sigma)
    state, init_projection = initialize_diagonal_gaussian_mixture(grid, components, match_exact_moments=True)
    steps = _ratio(final_time, dt)
    samples = _sample_steps(final_time, sample_interval, dt)
    sample_set = set(samples[1:])
    histories = [state.moments()]
    max_mass = max_momentum = max_energy = 0.0
    min_mass = float(np.min(state.masses))
    max_projection_residual = 0.0
    max_projection_iterations = 0
    start = time.perf_counter()
    for step in range(1, steps + 1):
        state, diag = dvm_cubic_fp_step(
            state,
            dt,
            tau,
            prandtl=prandtl,
            guided=True,
            projection_tolerance=1.0e-9,
        )
        max_mass = max(max_mass, abs(diag.mass_drift))
        max_momentum = max(max_momentum, abs(diag.momentum_drift))
        max_energy = max(max_energy, abs(diag.energy_drift))
        min_mass = min(min_mass, float(np.min(state.masses)))
        if diag.projection is not None:
            max_projection_residual = max(max_projection_residual, diag.projection.relative_moment_residual)
            max_projection_iterations = max(max_projection_iterations, int(diag.projection.iterations))
        if step in sample_set:
            histories.append(state.moments())
        if step == steps or step % max(steps // 5, 1) == 0:
            print(f"[stage74] dvm cells={cells} step={step}/{steps}", flush=True)
    return {
        "histories": np.asarray(histories),
        "grid": {"lower": list(grid.lower), "upper": list(grid.upper), "shape": list(grid.shape)},
        "dt": dt,
        "elapsed_seconds": time.perf_counter() - start,
        "initial_projection_relative_residual": None if init_projection is None else init_projection.relative_moment_residual,
        "maximum_step_projection_relative_residual": max_projection_residual,
        "maximum_step_projection_iterations": max_projection_iterations,
        "minimum_cell_mass": min_mass,
        "maximum_mass_drift": max_mass,
        "maximum_momentum_drift": max_momentum,
        "maximum_energy_trace_drift": max_energy,
    }


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--final-time", type=float, default=0.10)
    p.add_argument("--sample-interval", type=float, default=0.025)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    p.add_argument("--coarse-cells", type=int, default=25)
    p.add_argument("--fine-cells", type=int, default=33)
    p.add_argument("--coarse-dt", type=float, default=2.5e-3)
    p.add_argument("--fine-dt", type=float, default=1.25e-3)
    p.add_argument("--closure-dt", type=float, default=3.125e-4)
    p.add_argument("--margin-sigma", type=float, default=7.0)
    a = p.parse_args()

    case = hard_case(a.case)
    install_density_consistent_projection()
    coarse_samples = _sample_steps(a.final_time, a.sample_interval, a.coarse_dt)
    fine_samples = _sample_steps(a.final_time, a.sample_interval, a.fine_dt)
    closure_samples = _sample_steps(a.final_time, a.sample_interval, a.closure_dt)

    coarse = _run_dvm(case.components, cells=a.coarse_cells, dt=a.coarse_dt, final_time=a.final_time, sample_interval=a.sample_interval, tau=a.tau, prandtl=a.prandtl, margin_sigma=a.margin_sigma)
    fine = _run_dvm(case.components, cells=a.fine_cells, dt=a.fine_dt, final_time=a.final_time, sample_interval=a.sample_interval, tau=a.tau, prandtl=a.prandtl, margin_sigma=a.margin_sigma)
    closure = _run_persistent_candidate(
        case.components,
        dt=a.closure_dt,
        steps=_ratio(a.final_time, a.closure_dt),
        sample_steps=closure_samples,
        tau=a.tau,
        prandtl=a.prandtl,
        quadrature_nodes=5,
    )

    times = np.asarray(fine_samples, dtype=float) * a.fine_dt / a.tau
    if not np.allclose(times, np.asarray(coarse_samples) * a.coarse_dt / a.tau, atol=2e-13, rtol=0):
        raise RuntimeError("DVM sample times differ")
    if not np.allclose(times, np.asarray(closure_samples) * a.closure_dt / a.tau, atol=2e-13, rtol=0):
        raise RuntimeError("closure sample times differ")

    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output / f"stage74_{a.case}.npz",
        times=times,
        dvm_coarse_histories=coarse["histories"],
        dvm_fine_histories=fine["histories"],
        closure_histories=np.asarray(closure["histories"]),
    )
    summary = {
        "schema": "riemann35-stage74-deterministic-dvm-case-v1",
        "case": a.case,
        "stage71_fingerprint": case.fingerprint,
        "configuration": case.configuration,
        "controls": {
            "reference": "independent-clean-room-guided-Scharfetter-Gummel-DVM",
            "qmc_used": False,
            "density_jacobian_fix": True,
            "closure_parameter_refit": False,
            "final_time_over_tau": a.final_time / a.tau,
            "sample_interval_over_tau": a.sample_interval / a.tau,
            "coarse_cells_per_axis": a.coarse_cells,
            "fine_cells_per_axis": a.fine_cells,
            "coarse_dt_over_tau": a.coarse_dt / a.tau,
            "fine_dt_over_tau": a.fine_dt / a.tau,
            "closure_dt_over_tau": a.closure_dt / a.tau,
            "domain_margin_component_sigma": a.margin_sigma,
        },
        "dvm_coarse": {k: v for k, v in coarse.items() if k != "histories"},
        "dvm_fine": {k: v for k, v in fine.items() if k != "histories"},
        "closure": {
            "invariants": _invariants(np.asarray(closure["histories"])[None, ...]),
            "diagnostics": {k: v for k, v in closure.items() if k != "histories"},
        },
    }
    (a.output / f"stage74_{a.case}_summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps({"case": a.case, "fine_grid": fine["grid"], "dvm_fine_seconds": fine["elapsed_seconds"], "closure_seconds": closure["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

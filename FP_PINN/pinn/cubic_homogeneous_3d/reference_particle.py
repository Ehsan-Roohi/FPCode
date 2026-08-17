#!/usr/bin/env python3
"""Independent particle reference for homogeneous Stage-2/V4 cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from cubic_operator import (
    ALL_CASE_NAMES,
    analytic_initial_summary,
    case_default_nu,
    moments_from_samples,
    ou_cubic_step,
    sample_initial,
    solve_closure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=ALL_CASE_NAMES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=250_000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--histogram-bins", type=int, default=200)
    parser.add_argument("--histogram-vmax", type=float, default=5.0)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--print-every", type=int, default=50)
    return parser.parse_args()


def _write_history_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    header = [
        "time", "mean_x", "mean_y", "mean_z", "dm2",
        "p_xx", "p_xy", "p_xz", "p_yy", "p_yz", "p_zz",
        "q_x", "q_y", "q_z", "lambda", "closure_condition",
        "closure_residual",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for index, value in enumerate(arrays["time"]):
            writer.writerow(
                [value, *arrays["mean"][index], arrays["dm2"][index],
                 *arrays["pij"][index], *arrays["q"][index],
                 arrays["cubic_lambda"][index], arrays["closure_condition"][index],
                 arrays["closure_residual"][index]]
            )


def run_reference(args: argparse.Namespace) -> dict[str, float | bool | str]:
    if args.particles < 1_000:
        raise ValueError("Use at least 1000 particles for a meaningful reference")
    steps_float = args.tmax / args.dt
    steps = int(round(steps_float))
    if not np.isclose(steps_float, steps, rtol=0.0, atol=1.0e-12):
        raise ValueError("tmax must be an integer multiple of dt")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    velocity = sample_initial(args.case, args.particles, rng)
    velocity -= np.mean(velocity, axis=0)
    initial_dm2 = float(np.mean(np.sum(velocity * velocity, axis=1)))
    velocity *= np.sqrt(3.0 / initial_dm2)
    target_dm2 = 3.0

    edges = np.linspace(-args.histogram_vmax, args.histogram_vmax, args.histogram_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    records: dict[str, list[np.ndarray | float]] = {
        "time": [], "mean": [], "dm2": [], "pij": [], "q": [],
        "coefficients": [], "cubic_lambda": [], "closure_condition": [],
        "closure_residual": [], "marginal_x": [], "marginal_y": [],
        "marginal_z": [],
    }

    def record(step: int) -> None:
        moments = moments_from_samples(velocity)
        closure = solve_closure(moments, nu=args.nu)
        records["time"].append(step * args.dt)
        records["mean"].append(moments.mean.copy())
        records["dm2"].append(moments.dm2)
        records["pij"].append(moments.pij.copy())
        records["q"].append(moments.q.copy())
        records["coefficients"].append(closure.vector.copy())
        records["cubic_lambda"].append(closure.cubic_lambda)
        records["closure_condition"].append(closure.condition_number)
        records["closure_residual"].append(closure.linear_residual)
        for axis, name in enumerate(("marginal_x", "marginal_y", "marginal_z")):
            density, _ = np.histogram(velocity[:, axis], bins=edges, density=True)
            records[name].append(density)

    started = time.perf_counter()
    record(0)
    for step in range(1, steps + 1):
        moments = moments_from_samples(velocity)
        closure = solve_closure(moments, nu=args.nu)
        velocity = ou_cubic_step(
            velocity, moments, closure, args.dt, rng, nu=args.nu,
            preserve_invariants=True, target_dm2=target_dm2,
        )
        if step % args.save_every == 0 or step == steps:
            record(step)
        if step % args.print_every == 0 or step == steps:
            now = moments_from_samples(velocity)
            print(
                f"reference case={args.case} step={step:4d}/{steps} "
                f"t={step*args.dt:.3f} dm2={now.dm2:.12f} "
                f"|mean|={np.linalg.norm(now.mean):.3e} "
                f"|Q|={np.linalg.norm(now.q):.3e} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays["histogram_centers"] = centers
    arrays["histogram_edges"] = edges
    np.savez_compressed(output / "reference.npz", **arrays)
    _write_history_csv(output / "reference_history.csv", arrays)

    mean_error = np.linalg.norm(arrays["mean"], axis=1)
    energy_error = np.abs(arrays["dm2"] - target_dm2)
    p = arrays["pij"]
    dev_norm = np.sqrt(
        (p[:, 0]-1.0)**2 + (p[:, 3]-1.0)**2 + (p[:, 5]-1.0)**2
        + 2.0*(p[:, 1]**2+p[:, 2]**2+p[:, 4]**2)
    )
    q_norm = np.linalg.norm(arrays["q"], axis=1)
    finite = bool(all(np.all(np.isfinite(value)) for value in arrays.values()))
    metrics: dict[str, float | bool | str] = {
        "case": args.case,
        "case_default_nu": case_default_nu(args.case),
        "analytic_initial_heat_flux_qx": float(
            analytic_initial_summary(args.case)["q"][0]
        ),
        "particles": args.particles,
        "dt": args.dt,
        "saved_times": int(arrays["time"].size),
        "max_mean_norm": float(np.max(mean_error)),
        "max_energy_error": float(np.max(energy_error)),
        "initial_deviatoric_stress_norm": float(dev_norm[0]),
        "final_deviatoric_stress_norm": float(dev_norm[-1]),
        "initial_heat_flux_norm": float(q_norm[0]),
        "final_heat_flux_norm": float(q_norm[-1]),
        "max_closure_condition": float(np.max(arrays["closure_condition"])),
        "max_closure_linear_residual": float(np.max(arrays["closure_residual"])),
        "finite": finite,
        "invariants_passed": bool(
            finite and np.max(mean_error) < 5.0e-12 and np.max(energy_error) < 5.0e-12
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (output / "reference_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (output / "reference_config.json").write_text(
        json.dumps(vars(args) | {"output_dir": str(output)}, indent=2, default=str) + "\n"
    )
    print("REFERENCE_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


def main() -> None:
    metrics = run_reference(parse_args())
    if not metrics["invariants_passed"]:
        raise SystemExit("Reference invariant gate failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the first homogeneous HyQMOM-FP coupling experiment."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    HYQMOM_35_INDICES,
    bgk_collision_source,
    coefficients_from_moments,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
    projected_fp_collision_source,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare homogeneous 35-moment BGK and projected cubic-FP "
            "relaxation from the same two-stream initial state."
        )
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hyqmom_fp_relaxation.csv"),
    )
    return parser.parse_args()


def diagnostic_row(step: int, dt: float, fp: np.ndarray, bgk: np.ndarray) -> dict[str, float]:
    position = {index: i for i, index in enumerate(HYQMOM_35_INDICES)}
    fp_state = macroscopic_state(fp)
    bgk_state = macroscopic_state(bgk)
    return {
        "step": step,
        "time": step * dt,
        "fp_rho": fp_state.rho,
        "fp_energy_trace": fp[position[(2, 0, 0)]]
        + fp[position[(0, 2, 0)]]
        + fp[position[(0, 0, 2)]],
        "fp_M200": fp[position[(2, 0, 0)]],
        "fp_M400": fp[position[(4, 0, 0)]],
        "bgk_rho": bgk_state.rho,
        "bgk_energy_trace": bgk[position[(2, 0, 0)]]
        + bgk[position[(0, 2, 0)]]
        + bgk[position[(0, 0, 2)]],
        "bgk_M200": bgk[position[(2, 0, 0)]],
        "bgk_M400": bgk[position[(4, 0, 0)]],
    }


def main() -> None:
    args = parse_arguments()
    if args.steps <= 0 or args.dt <= 0.0 or args.tau <= 0.0:
        raise SystemExit("steps, dt, and tau must be positive")

    covariance = 0.45 * np.eye(3)
    initial = mixture_of_gaussians_moments_35(
        [
            (0.5, (-0.55, 0.0, 0.0), covariance),
            (0.5, (0.55, 0.0, 0.0), covariance),
        ]
    )
    fp = initial.copy()
    bgk = initial.copy()
    rows = [diagnostic_row(0, args.dt, fp, bgk)]

    for step in range(1, args.steps + 1):
        fp_coefficients = coefficients_from_moments(
            fp, tau=args.tau, prandtl=args.prandtl
        )
        fp += args.dt * projected_fp_collision_source(fp, fp_coefficients)
        bgk += args.dt * bgk_collision_source(bgk, args.tau)
        rows.append(diagnostic_row(step, args.dt, fp, bgk))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    initial_row = rows[0]
    final_row = rows[-1]
    fp_mass_error = abs(final_row["fp_rho"] - initial_row["fp_rho"])
    fp_energy_error = abs(
        final_row["fp_energy_trace"] - initial_row["fp_energy_trace"]
    )
    print(f"wrote {len(rows)} states to {args.output}")
    print(f"projected-FP mass drift:   {fp_mass_error:.3e}")
    print(f"projected-FP energy drift: {fp_energy_error:.3e}")
    print(
        "M400 final (FP, BGK): "
        f"{final_row['fp_M400']:.8e}, {final_row['bgk_M400']:.8e}"
    )


if __name__ == "__main__":
    main()

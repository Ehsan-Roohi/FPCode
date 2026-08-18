#!/usr/bin/env python3
"""Validate the projected 35-moment FP source against FPCode particles."""

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
    coefficients_from_moments,
    macroscopic_state,
    moments_35_from_particles,
    particle_cubic_fp_step,
    projected_fp_collision_source,
    sample_gaussian_mixture,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the closed 35-moment cubic-FP relaxation with the "
            "finite-particle physics update used by FPCode."
        )
    )
    parser.add_argument("--particles", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hyqmom_fp_particle_validation.csv"),
    )
    parser.add_argument(
        "--no-speed-limit",
        action="store_true",
        help="disable FPCode's |c|^2 <= 25 theta limiter",
    )
    return parser.parse_args()


POSITION = {index: i for i, index in enumerate(HYQMOM_35_INDICES)}


def diagnostics(step: int, dt: float, closed: np.ndarray, particle: np.ndarray) -> dict[str, float]:
    closed_state = macroscopic_state(closed)
    particle_state = macroscopic_state(particle)
    row: dict[str, float] = {"step": step, "time": step * dt}
    for label, vector, state in (
        ("closed", closed, closed_state),
        ("particle", particle, particle_state),
    ):
        row[f"{label}_rho"] = state.rho
        row[f"{label}_energy_trace"] = sum(
            vector[POSITION[index]]
            for index in ((2, 0, 0), (0, 2, 0), (0, 0, 2))
        )
        row[f"{label}_M200"] = vector[POSITION[(2, 0, 0)]]
        row[f"{label}_M300"] = vector[POSITION[(3, 0, 0)]]
        row[f"{label}_M400"] = vector[POSITION[(4, 0, 0)]]
        row[f"{label}_stress_norm"] = float(np.linalg.norm(state.stress))
        row[f"{label}_heat_flux_norm"] = float(np.linalg.norm(state.heat_flux))
    return row


def relative_history_l2(rows: list[dict[str, float]], quantity: str) -> float:
    closed = np.asarray([row[f"closed_{quantity}"] for row in rows])
    particle = np.asarray([row[f"particle_{quantity}"] for row in rows])
    scale = max(float(np.linalg.norm(particle)), 1.0e-14)
    return float(np.linalg.norm(closed - particle) / scale)


def main() -> None:
    args = parse_arguments()
    if (
        args.particles < 2
        or args.steps <= 0
        or args.dt <= 0.0
        or args.tau <= 0.0
        or args.sample_every <= 0
    ):
        raise SystemExit("particles, steps, dt, tau, and sample-every must be positive")

    covariance = 0.45 * np.eye(3)
    particles = sample_gaussian_mixture(
        [
            (0.5, (-0.55, 0.0, 0.0), covariance),
            (0.5, (0.55, 0.0, 0.0), covariance),
        ],
        particles=args.particles,
        seed=args.seed,
    )
    # Both models start from the *measured* ensemble moments, removing initial
    # Monte-Carlo mismatch from the validation metric.
    closed = moments_35_from_particles(particles)
    particle_moments = closed.copy()
    rows = [diagnostics(0, args.dt, closed, particle_moments)]
    rng = np.random.default_rng(args.seed + 1)
    maximum_particle_energy_drift = 0.0

    for step in range(1, args.steps + 1):
        coefficients = coefficients_from_moments(
            closed, tau=args.tau, prandtl=args.prandtl
        )
        closed += args.dt * projected_fp_collision_source(closed, coefficients)
        particles, step_diagnostics = particle_cubic_fp_step(
            particles,
            dt=args.dt,
            tau=args.tau,
            rng=rng,
            prandtl=args.prandtl,
            limit_peculiar_speed=not args.no_speed_limit,
            enforce_sample_invariants=True,
        )
        maximum_particle_energy_drift = max(
            maximum_particle_energy_drift, abs(step_diagnostics.energy_drift)
        )
        if step % args.sample_every == 0 or step == args.steps:
            particle_moments = moments_35_from_particles(particles)
            rows.append(diagnostics(step, args.dt, closed, particle_moments))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    initial = rows[0]
    final = rows[-1]
    closed_mass_drift = abs(final["closed_rho"] - initial["closed_rho"])
    closed_energy_drift = abs(
        final["closed_energy_trace"] - initial["closed_energy_trace"]
    )
    m400_relative = abs(final["closed_M400"] - final["particle_M400"]) / max(
        abs(final["particle_M400"]), 1.0e-14
    )

    print(f"wrote {len(rows)} sampled states to {args.output}")
    print(f"particles / steps: {args.particles} / {args.steps}")
    print(f"closed-model mass drift:       {closed_mass_drift:.3e}")
    print(f"closed-model energy drift:     {closed_energy_drift:.3e}")
    print(f"particle max-step energy drift:{maximum_particle_energy_drift: .3e}")
    print(
        "final M400 (closed, particle): "
        f"{final['closed_M400']:.8e}, {final['particle_M400']:.8e}"
    )
    print(f"final M400 relative difference: {m400_relative:.3%}")
    for quantity in ("M200", "M400", "stress_norm", "heat_flux_norm"):
        print(f"history L2 {quantity:>14}: {relative_history_l2(rows, quantity):.3%}")


if __name__ == "__main__":
    main()

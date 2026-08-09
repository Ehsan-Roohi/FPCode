#!/usr/bin/env python3
"""Export the deterministic Stage-2 particle reference and initial M4 state."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    HYQMOM_35_INDICES,
    macroscopic_state,
    moments_35_from_particles,
    particle_cubic_fp_step,
    sample_gaussian_mixture,
)


POSITION = {index: i for i, index in enumerate(HYQMOM_35_INDICES)}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--gamma-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--initial-output", type=Path, required=True)
    parser.add_argument("--history-output", type=Path, required=True)
    return parser.parse_args()


def diagnostic_row(step: int, dt: float, moments: np.ndarray) -> dict[str, float]:
    state = macroscopic_state(moments)
    return {
        "step": step,
        "time": step * dt,
        "rho": state.rho,
        "energy_trace": sum(
            moments[POSITION[index]]
            for index in ((2, 0, 0), (0, 2, 0), (0, 0, 2))
        ),
        "M200": moments[POSITION[(2, 0, 0)]],
        "M300": moments[POSITION[(3, 0, 0)]],
        "M400": moments[POSITION[(4, 0, 0)]],
        "stress_norm": float(np.linalg.norm(state.stress)),
        "heat_flux_norm": float(np.linalg.norm(state.heat_flux)),
    }


def main() -> None:
    args = parse_arguments()
    if (
        args.particles < 2
        or args.steps < 1
        or args.dt <= 0.0
        or args.tau <= 0.0
        or args.sample_every < 1
    ):
        raise SystemExit("invalid positive particle-validation controls")

    covariance = 0.45 * np.eye(3)
    particles = sample_gaussian_mixture(
        [
            (0.5, (-0.55, 0.0, 0.0), covariance),
            (0.5, (0.55, 0.0, 0.0), covariance),
        ],
        particles=args.particles,
        seed=args.seed,
    )
    initial = moments_35_from_particles(particles)

    args.initial_output.parent.mkdir(parents=True, exist_ok=True)
    with args.initial_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([f"M{i}{j}{k}" for i, j, k in HYQMOM_35_INDICES])
        writer.writerow([f"{value:.17g}" for value in initial])

    rows = [diagnostic_row(0, args.dt, initial)]
    rng = np.random.default_rng(args.seed + 1)
    maximum_energy_drift = 0.0
    maximum_momentum_drift = 0.0
    minimum_alpha = np.inf

    for step in range(1, args.steps + 1):
        particles, diagnostics = particle_cubic_fp_step(
            particles,
            dt=args.dt,
            tau=args.tau,
            rng=rng,
            prandtl=args.prandtl,
            gamma_scale=args.gamma_scale,
            limit_peculiar_speed=True,
            enforce_sample_invariants=True,
        )
        maximum_energy_drift = max(
            maximum_energy_drift, abs(diagnostics.energy_drift)
        )
        maximum_momentum_drift = max(
            maximum_momentum_drift, abs(diagnostics.momentum_drift)
        )
        minimum_alpha = min(minimum_alpha, diagnostics.alpha)
        if step % args.sample_every == 0 or step == args.steps:
            rows.append(
                diagnostic_row(step, args.dt, moments_35_from_particles(particles))
            )

    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    with args.history_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"particle reference: {args.particles} particles, {args.steps} steps")
    print(f"particle samples written: {len(rows)}")
    print(f"particle max-step momentum drift: {maximum_momentum_drift:.3e}")
    print(f"particle max-step energy drift:   {maximum_energy_drift:.3e}")
    print(f"particle minimum alpha:           {minimum_alpha:.12g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose the high-skew M400 source against independent particles."""

from __future__ import annotations

import argparse
import csv
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
    HYQMOM_35_INDICES,
    WeightedNodeTailClosure,
    coefficients_from_moments,
    coefficients_from_particles,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
    moments_35_from_particles,
    particle_cubic_fp_step,
    particle_macroscopic_state,
    projected_fp_collision_source,
    reconstruct_gaussian_mixture_quadrature,
    reconstruct_grad_hyqmom_quadrature,
    reconstruct_two_population_quadrature,
    sample_gaussian_mixture,
)
from riemann35_patch.stage10.run_general_realizability_audit import (  # noqa: E402
    deterministic_states,
)


POSITION = {index: position for position, index in enumerate(HYQMOM_35_INDICES)}
M400_POSITION = POSITION[(4, 0, 0)]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=20_260_810)
    parser.add_argument("--seed-stride", type=int, default=104_729)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rare_beam_components():
    states = {state.name: state for state in deterministic_states()}
    return states["rare_beam_ma20"].components


def empirical_m400_source(velocities: np.ndarray) -> float:
    state = particle_macroscopic_state(velocities)
    coefficients = coefficients_from_particles(
        velocities, tau=1.0, prandtl=2.0 / 3.0
    )
    peculiar = velocities - state.velocity
    c2 = np.einsum("ni,ni->n", peculiar, peculiar)
    drift = -peculiar
    drift += peculiar @ coefficients.C.T
    drift += (c2 - 3.0 * state.theta)[:, None] * coefficients.gamma
    drift += coefficients.beta * (
        c2[:, None] * peculiar - 2.0 * state.heat_flux[None, :] / state.rho
    )
    diffusion = state.theta
    vx = velocities[:, 0]
    return float(np.mean(4.0 * vx**3 * drift[:, 0] + 12.0 * diffusion * vx**2))


def particle_task(arguments: tuple) -> tuple[np.ndarray, np.ndarray]:
    components, particles, seed, steps, dt, sample_every = arguments
    velocities = sample_gaussian_mixture(components, particles=particles, seed=seed)
    rng = np.random.default_rng(seed + 1_000_003)
    moments = [moments_35_from_particles(velocities)]
    sources = [empirical_m400_source(velocities)]
    for step in range(1, steps + 1):
        velocities, _ = particle_cubic_fp_step(
            velocities,
            dt=dt,
            tau=1.0,
            rng=rng,
            prandtl=2.0 / 3.0,
            limit_peculiar_speed=False,
            enforce_sample_invariants=True,
        )
        if step % sample_every == 0 or step == steps:
            moments.append(moments_35_from_particles(velocities))
            sources.append(empirical_m400_source(velocities))
    return np.asarray(moments), np.asarray(sources)


def source_from_quadrature(moments: np.ndarray, quadrature) -> float:
    closure = WeightedNodeTailClosure(quadrature.nodes, quadrature.weights)
    coefficients = coefficients_from_moments(
        moments, tau=1.0, prandtl=2.0 / 3.0, closure=closure
    )
    return float(
        projected_fp_collision_source(moments, coefficients, closure=closure)[
            M400_POSITION
        ]
    )


def closure_sources(moments: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    builders = {
        "stage9": lambda: reconstruct_gaussian_mixture_quadrature(moments),
        "grad": lambda: reconstruct_grad_hyqmom_quadrature(moments),
        "two_population_base": lambda: reconstruct_two_population_quadrature(
            moments, minimum_skewness_norm=0.05, residual_correction=False
        ),
        "two_population_residual": lambda: reconstruct_two_population_quadrature(
            moments, minimum_skewness_norm=0.05, residual_correction=True
        ),
    }
    for name, builder in builders.items():
        try:
            result[name] = source_from_quadrature(moments, builder())
        except Exception:
            result[name] = float("nan")
    return result


def main() -> None:
    args = parse_arguments()
    steps = int(round(args.final_time / args.dt))
    if steps <= 0 or args.sample_every <= 0:
        raise SystemExit("invalid integration controls")
    components = rare_beam_components()
    exact_initial = mixture_of_gaussians_moments_35(components)
    seeds = [args.seed_base + args.seed_stride * index for index in range(args.seeds)]
    tasks = [
        (components, args.particles, seed, steps, args.dt, args.sample_every)
        for seed in seeds
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"rare-beam source diagnostic: {args.seeds} seeds x {args.particles} particles",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(particle_task, tasks))
    raw_moments = np.asarray([item[0] for item in results])
    particle_sources = np.asarray([item[1] for item in results])
    aligned_moments = raw_moments - raw_moments[:, :1, :] + exact_initial
    mean_moments = np.mean(aligned_moments, axis=0)
    particle_source_mean = np.mean(particle_sources, axis=0)
    particle_source_sem = np.std(particle_sources, axis=0, ddof=1) / np.sqrt(args.seeds)
    predictions = [closure_sources(row) for row in mean_moments]
    times = np.arange(mean_moments.shape[0]) * args.sample_every * args.dt
    fields = [
        "time",
        "particle_m400",
        "particle_m400_source_mean",
        "particle_m400_source_sem",
        "stage9",
        "grad",
        "two_population_base",
        "two_population_residual",
    ]
    csv_path = args.output / "stage12_rare_beam_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample, current_time in enumerate(times):
            writer.writerow(
                {
                    "time": float(current_time),
                    "particle_m400": float(mean_moments[sample, M400_POSITION]),
                    "particle_m400_source_mean": float(particle_source_mean[sample]),
                    "particle_m400_source_sem": float(particle_source_sem[sample]),
                    **predictions[sample],
                }
            )

    source_errors = {}
    for method in fields[4:]:
        values = np.asarray([row[method] for row in predictions])
        mask = np.isfinite(values)
        source_errors[method] = float(
            np.linalg.norm(values[mask] - particle_source_mean[mask])
            / max(np.linalg.norm(particle_source_mean[mask]), 1.0e-14)
        )
    summary = {
        "schema": "riemann35-stage12-rare-beam-source-diagnostic-v1",
        "particles_per_seed": args.particles,
        "independent_seeds": args.seeds,
        "dt_over_tau": args.dt,
        "final_time_over_tau": args.final_time,
        "m400_source_history_relative_errors": source_errors,
    }
    (args.output / "stage12_rare_beam_source.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.fill_between(
        times,
        particle_source_mean - 2.13145 * particle_source_sem,
        particle_source_mean + 2.13145 * particle_source_sem,
        color="0.88",
        label="Particle 95% seed CI",
    )
    axis.plot(times, particle_source_mean, "o-k", ms=2.5, lw=0.9, label="Particle mean")
    colors = {
        "stage9": "#cc3311",
        "grad": "#0077bb",
        "two_population_base": "#117864",
        "two_population_residual": "#aa4499",
    }
    for method in fields[4:]:
        axis.plot(
            times,
            [row[method] for row in predictions],
            lw=1.2,
            color=colors[method],
            label=method.replace("_", " "),
        )
    axis.set_xlabel(r"Time, $t/\tau$")
    axis.set_ylabel(r"Instantaneous $dM_{400}/dt$")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(args.output / "stage12_rare_beam_source.png", dpi=400)
    plt.close(figure)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

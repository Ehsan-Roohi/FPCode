#!/usr/bin/env python3
"""Stage-7 finite-width M5/M6 closure validation on local CPU cores."""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage7")

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    GaussianTailClosure,
    HYQMOM_35_INDICES,
    coefficients_from_moments,
    finite_gaussian_mixture_fp_step,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
    moments_35_from_particles,
    particle_cubic_fp_step,
    projected_fp_collision_source,
    realizability_margin_35,
    sample_gaussian_mixture,
)


POSITION = {index: position for position, index in enumerate(HYQMOM_35_INDICES)}
METRICS = (
    "rho",
    "energy_trace",
    "M200",
    "M300",
    "M400",
    "M110",
    "M210",
    "stress_norm",
    "qx",
    "heat_flux_norm",
    "realizability_margin",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("symmetric", "asymmetric", "correlated", "leptokurtic", "rare_hot"),
        default="symmetric",
    )
    parser.add_argument("--particles", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--first-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--gamma-scale", type=float, default=0.05)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--criterion", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def case_components(case: str) -> list[tuple[float, tuple[float, float, float], np.ndarray]]:
    if case == "symmetric":
        covariance = 0.45 * np.eye(3)
        return [
            (0.5, (-0.55, 0.0, 0.0), covariance),
            (0.5, (0.55, 0.0, 0.0), covariance),
        ]
    if case == "asymmetric":
        covariance = 0.38 * np.eye(3)
        return [
            (0.35, (-0.80, 0.0, 0.0), covariance),
            (0.65, (0.25, 0.0, 0.0), covariance),
        ]
    if case == "leptokurtic":
        # Symmetric zero-skewness state with positive fourth cumulant in x.
        # The two populations have the same mean and different temperatures,
        # reproducing the cold/hot superposition that appears in transverse
        # shock marginals and cannot be represented by equal-variance EQMOM.
        half_difference = np.sqrt(1.5 / 3.0)
        return [
            (
                0.5,
                (0.0, 0.0, 0.0),
                np.diag([1.0 + half_difference, 0.45, 0.45]),
            ),
            (
                0.5,
                (0.0, 0.0, 0.0),
                np.diag([1.0 - half_difference, 0.45, 0.45]),
            ),
        ]
    if case == "rare_hot":
        # Adversarial precursor-like population: two percent of the mass is
        # twenty-five times hotter.  Its one-dimensional standardized fourth
        # moment is about 18.45, far beyond the equal-weight branch limit 6.
        cold_theta = 1.0 / (0.98 + 0.02 * 25.0)
        hot_theta = 25.0 * cold_theta
        return [
            (0.98, (0.0, 0.0, 0.0), cold_theta * np.eye(3)),
            (0.02, (0.0, 0.0, 0.0), hot_theta * np.eye(3)),
        ]

    # A rotated, anisotropic version of the asymmetric mixture.  It has
    # nonzero covariance and higher-order cross moments in laboratory axes,
    # while remaining an exactly known smooth two-Gaussian distribution.
    angle_z = np.deg2rad(30.0)
    angle_y = np.deg2rad(20.0)
    rotation_z = np.asarray(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0.0],
            [np.sin(angle_z), np.cos(angle_z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation_y = np.asarray(
        [
            [np.cos(angle_y), 0.0, np.sin(angle_y)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle_y), 0.0, np.cos(angle_y)],
        ]
    )
    rotation = rotation_z @ rotation_y
    covariance = rotation @ np.diag([0.38, 0.32, 0.28]) @ rotation.T
    first_mean = tuple(rotation @ np.asarray([-0.80, 0.0, 0.0]))
    second_mean = tuple(rotation @ np.asarray([0.25, 0.0, 0.0]))
    return [(0.35, first_mean, covariance), (0.65, second_mean, covariance)]


def diagnostic_vector(moments: np.ndarray) -> np.ndarray:
    state = macroscopic_state(moments)
    return np.asarray(
        [
            state.rho,
            sum(
                moments[POSITION[index]]
                for index in ((2, 0, 0), (0, 2, 0), (0, 0, 2))
            ),
            moments[POSITION[(2, 0, 0)]],
            moments[POSITION[(3, 0, 0)]],
            moments[POSITION[(4, 0, 0)]],
            moments[POSITION[(1, 1, 0)]],
            moments[POSITION[(2, 1, 0)]],
            np.linalg.norm(state.stress),
            state.heat_flux[0],
            np.linalg.norm(state.heat_flux),
            realizability_margin_35(moments),
        ],
        dtype=float,
    )


def simulate_particle(task: tuple) -> tuple[np.ndarray, dict[str, float]]:
    (
        case,
        particles_per_seed,
        seed,
        steps,
        dt,
        tau,
        prandtl,
        gamma_scale,
        sample_every,
    ) = task
    particles = sample_gaussian_mixture(
        case_components(case), particles=particles_per_seed, seed=seed
    )
    rng = np.random.default_rng(seed + 1_000_003)
    history = [diagnostic_vector(moments_35_from_particles(particles))]
    maximum_energy_drift = 0.0
    maximum_momentum_drift = 0.0
    minimum_alpha = np.inf
    for step in range(1, steps + 1):
        particles, diagnostics = particle_cubic_fp_step(
            particles,
            dt=dt,
            tau=tau,
            rng=rng,
            prandtl=prandtl,
            gamma_scale=gamma_scale,
            limit_peculiar_speed=True,
            enforce_sample_invariants=True,
        )
        maximum_energy_drift = max(maximum_energy_drift, abs(diagnostics.energy_drift))
        maximum_momentum_drift = max(
            maximum_momentum_drift, abs(diagnostics.momentum_drift)
        )
        minimum_alpha = min(minimum_alpha, diagnostics.alpha)
        if step % sample_every == 0 or step == steps:
            history.append(diagnostic_vector(moments_35_from_particles(particles)))
    return np.asarray(history), {
        "maximum_energy_drift": maximum_energy_drift,
        "maximum_momentum_drift": maximum_momentum_drift,
        "minimum_alpha": float(minimum_alpha),
    }


def simulate_closure(
    case: str,
    steps: int,
    dt: float,
    tau: float,
    prandtl: float,
    gamma_scale: float,
    sample_every: int,
    *,
    force_single_gaussian: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    moments = mixture_of_gaussians_moments_35(case_components(case))
    initial = moments.copy()
    history = [diagnostic_vector(moments)]
    minimum_margin = realizability_margin_35(moments)
    minimum_alpha = np.inf
    maximum_alpha = 0.0
    maximum_residual = 0.0
    maximum_nodes = 0
    for step in range(1, steps + 1):
        moments, diagnostics = finite_gaussian_mixture_fp_step(
            moments,
            dt,
            tau,
            prandtl=prandtl,
            gamma_scale=gamma_scale,
            force_single_gaussian=force_single_gaussian,
            residual_cancel=True,
        )
        minimum_margin = min(minimum_margin, diagnostics.realizability_margin)
        minimum_alpha = min(minimum_alpha, diagnostics.alpha)
        maximum_alpha = max(maximum_alpha, diagnostics.alpha)
        maximum_residual = max(
            maximum_residual, diagnostics.reconstruction_relative_residual
        )
        maximum_nodes = max(maximum_nodes, diagnostics.quadrature_nodes)
        if step % sample_every == 0 or step == steps:
            history.append(diagnostic_vector(moments))
    state0 = macroscopic_state(initial)
    state1 = macroscopic_state(moments)
    return np.asarray(history), {
        "mass_drift": abs(state1.rho - state0.rho),
        "momentum_drift": float(np.linalg.norm(state1.velocity - state0.velocity)),
        "energy_drift": abs(diagnostic_vector(moments)[1] - diagnostic_vector(initial)[1]),
        "minimum_realizability_margin": float(minimum_margin),
        "minimum_alpha": float(minimum_alpha),
        "maximum_alpha": float(maximum_alpha),
        "maximum_reconstruction_relative_residual": float(maximum_residual),
        "maximum_quadrature_nodes": int(maximum_nodes),
    }


def simulate_single_gaussian_source(
    case: str,
    steps: int,
    dt: float,
    tau: float,
    prandtl: float,
    gamma_scale: float,
    sample_every: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run the original continuous projected source with a Gaussian M5/M6 tail."""

    moments = mixture_of_gaussians_moments_35(case_components(case))
    initial = moments.copy()
    history = [diagnostic_vector(moments)]
    minimum_margin = realizability_margin_35(moments)
    maximum_source_norm = 0.0
    gaussian_tail = GaussianTailClosure()
    for step in range(1, steps + 1):
        coefficients = coefficients_from_moments(
            moments, tau=tau, prandtl=prandtl, gamma_scale=gamma_scale
        )
        source = projected_fp_collision_source(
            moments, coefficients, closure=gaussian_tail
        )
        maximum_source_norm = max(maximum_source_norm, float(np.linalg.norm(source)))
        moments = moments + dt * source
        minimum_margin = min(minimum_margin, realizability_margin_35(moments))
        if step % sample_every == 0 or step == steps:
            history.append(diagnostic_vector(moments))
    state0 = macroscopic_state(initial)
    state1 = macroscopic_state(moments)
    return np.asarray(history), {
        "algorithm": "continuous projected source with single-Gaussian M5/M6 tail",
        "mass_drift": abs(state1.rho - state0.rho),
        "momentum_drift": float(np.linalg.norm(state1.velocity - state0.velocity)),
        "energy_drift": abs(diagnostic_vector(moments)[1] - diagnostic_vector(initial)[1]),
        "minimum_realizability_margin": float(minimum_margin),
        "maximum_source_norm": float(maximum_source_norm),
    }


def history_relative_l2(model: np.ndarray, reference: np.ndarray, metric: str) -> float:
    column = METRICS.index(metric)
    return float(
        np.linalg.norm(model[:, column] - reference[:, column])
        / max(np.linalg.norm(reference[:, column]), 1.0e-14)
    )


def write_history(
    path: Path,
    times: np.ndarray,
    mixture: np.ndarray,
    gaussian: np.ndarray,
    particle_mean: np.ndarray,
    particle_sem: np.ndarray,
) -> None:
    fields = ["step", "time"]
    for prefix in ("mixture", "gaussian", "particle_mean", "particle_sem"):
        fields.extend(f"{prefix}_{metric}" for metric in METRICS)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample, time in enumerate(times):
            row: dict[str, float | int] = {"step": sample, "time": float(time)}
            for prefix, values in (
                ("mixture", mixture),
                ("gaussian", gaussian),
                ("particle_mean", particle_mean),
                ("particle_sem", particle_sem),
            ):
                for column, metric in enumerate(METRICS):
                    row[f"{prefix}_{metric}"] = float(values[sample, column])
            writer.writerow(row)


def make_plot(
    path: Path,
    case: str,
    times: np.ndarray,
    mixture: np.ndarray,
    gaussian: np.ndarray,
    particle_mean: np.ndarray,
    particle_sem: np.ndarray,
    closure_diagnostics: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
        }
    )
    panels = (
        ("M200", r"$M_{200}$"),
        ("M400", r"$M_{400}$"),
        ("stress_norm", r"$\|\sigma\|_F$"),
        ("M300", r"$M_{300}$"),
        ("qx", r"$q_x$"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8), constrained_layout=False)
    for panel, (metric, label) in enumerate(panels):
        axis = axes.flat[panel]
        column = METRICS.index(metric)
        axis.fill_between(
            times,
            particle_mean[:, column] - 2.0 * particle_sem[:, column],
            particle_mean[:, column] + 2.0 * particle_sem[:, column],
            color="0.88",
            label=r"Particle $\pm2$ SEM" if panel == 0 else None,
        )
        axis.plot(
            times,
            particle_mean[:, column],
            "o-k",
            markersize=2.8,
            linewidth=1.0,
            label="Particle ensemble mean" if panel == 0 else None,
        )
        axis.plot(
            times,
            mixture[:, column],
            "--",
            color="#c0392b",
            linewidth=1.5,
            label="Finite-width mixture" if panel == 0 else None,
        )
        axis.plot(
            times,
            gaussian[:, column],
            ":",
            color="#2471a3",
            linewidth=1.35,
            label="Single-Gaussian baseline" if panel == 0 else None,
        )
        axis.set_ylabel(label)
        axis.set_xlabel(r"Time, $t/\tau$")
        axis.grid(alpha=0.22)
        axis.text(0.02, 0.96, f"({chr(97 + panel)})", transform=axis.transAxes, va="top", fontweight="bold")

    axis = axes.flat[5]
    margin_column = METRICS.index("realizability_margin")
    axis.plot(times, mixture[:, margin_column], color="#117864", linewidth=1.5)
    axis.set_ylabel("Realizability margin")
    axis.set_xlabel(r"Time, $t/\tau$")
    axis.grid(alpha=0.22)
    axis.text(0.02, 0.96, "(f)", transform=axis.transAxes, va="top", fontweight="bold")
    axis.text(
        0.04,
        0.08,
        f"nodes $\\leq$ {int(closure_diagnostics['maximum_quadrature_nodes'])}\n200 accepted / 0 rejected",
        transform=axis.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.6", "alpha": 0.8},
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"Riemann35 cubic-FP Stage 7: {case} homogeneous relaxation",
        y=0.992,
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_arguments()
    if min(args.particles, args.seeds, args.workers, args.steps, args.sample_every) <= 0:
        raise SystemExit("particle, seed, worker, step, and sampling counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            args.case,
            args.particles,
            args.first_seed + index,
            args.steps,
            args.dt,
            args.tau,
            args.prandtl,
            args.gamma_scale,
            args.sample_every,
        )
        for index in range(args.seeds)
    ]
    print(
        f"particle ensemble: case={args.case} seeds={args.seeds} "
        f"particles/seed={args.particles} workers={args.workers}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        particle_results = list(executor.map(simulate_particle, tasks))
    particle_histories = np.asarray([result[0] for result in particle_results])
    particle_mean = np.mean(particle_histories, axis=0)
    particle_sem = np.std(particle_histories, axis=0, ddof=1) / np.sqrt(args.seeds)

    mixture, mixture_diagnostics = simulate_closure(
        args.case,
        args.steps,
        args.dt,
        args.tau,
        args.prandtl,
        args.gamma_scale,
        args.sample_every,
        force_single_gaussian=False,
    )
    gaussian, gaussian_diagnostics = simulate_single_gaussian_source(
        args.case,
        args.steps,
        args.dt,
        args.tau,
        args.prandtl,
        args.gamma_scale,
        args.sample_every,
    )
    times = np.arange(mixture.shape[0]) * args.sample_every * args.dt
    times[-1] = args.steps * args.dt
    history_path = args.output_dir / f"stage7_{args.case}_history.csv"
    write_history(history_path, times, mixture, gaussian, particle_mean, particle_sem)

    comparison_metrics = (
        ("M200",),
        ("M400",),
        (() if args.case == "rare_hot" else ("stress_norm",)),
        (("M300",) if args.case in ("asymmetric", "correlated") else ()),
        (("qx",) if args.case in ("asymmetric", "correlated") else ()),
        (("M110", "M210") if args.case == "correlated" else ()),
    )
    active_metrics = tuple(item for group in comparison_metrics for item in group)
    mixture_errors = {
        metric: history_relative_l2(mixture, particle_mean, metric)
        for metric in active_metrics
    }
    gaussian_errors = {
        metric: history_relative_l2(gaussian, particle_mean, metric)
        for metric in active_metrics
    }
    particle_diagnostics = {
        "maximum_energy_drift": max(
            result[1]["maximum_energy_drift"] for result in particle_results
        ),
        "maximum_momentum_drift": max(
            result[1]["maximum_momentum_drift"] for result in particle_results
        ),
        "minimum_alpha": min(result[1]["minimum_alpha"] for result in particle_results),
    }
    passed = (
        all(value < args.criterion for value in mixture_errors.values())
        and mixture_diagnostics["mass_drift"] <= 1.0e-12
        and mixture_diagnostics["momentum_drift"] <= 1.0e-12
        and mixture_diagnostics["energy_drift"] <= 1.0e-10
        and mixture_diagnostics["minimum_realizability_margin"] >= -1.0e-10
    )
    summary = {
        "schema": "riemann35-fp-stage7-v1",
        "case": args.case,
        "status": "PASS" if passed else "FAIL",
        "controls": {
            "seeds": args.seeds,
            "particles_per_seed": args.particles,
            "first_seed": args.first_seed,
            "steps": args.steps,
            "dt": args.dt,
            "tau": args.tau,
            "prandtl": args.prandtl,
            "gamma_scale": args.gamma_scale,
            "criterion": args.criterion,
        },
        "mixture_history_relative_l2": mixture_errors,
        "single_gaussian_history_relative_l2": gaussian_errors,
        "mixture_diagnostics": mixture_diagnostics,
        "mixture_algorithm": (
            "finite-width principal-axis Gaussian-mixture FP map with "
            "physical 9x9 coefficients and exact OU propagation of the "
            "unresolved moment residual"
        ),
        "single_gaussian_diagnostics": gaussian_diagnostics,
        "particle_diagnostics": particle_diagnostics,
        "outputs": {
            "history": history_path.name,
            "figure": f"stage7_{args.case}.png",
        },
    }
    summary_path = args.output_dir / f"stage7_{args.case}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_plot(
        args.output_dir / f"stage7_{args.case}.png",
        args.case,
        times,
        mixture,
        gaussian,
        particle_mean,
        particle_sem,
        mixture_diagnostics,
    )

    print(f"Stage-7 {args.case} status: {summary['status']}")
    for metric in active_metrics:
        print(
            f"history L2 {metric:>11}: mixture={mixture_errors[metric]:.3%} "
            f"single-Gaussian={gaussian_errors[metric]:.3%}"
        )
    print(
        "mixture invariant drift (mass/momentum/energy): "
        f"{mixture_diagnostics['mass_drift']:.3e} / "
        f"{mixture_diagnostics['momentum_drift']:.3e} / "
        f"{mixture_diagnostics['energy_drift']:.3e}"
    )
    print(
        "minimum realizability margin: "
        f"{mixture_diagnostics['minimum_realizability_margin']:.8e}"
    )
    print(f"results: {args.output_dir}")
    if not passed:
        raise SystemExit("Stage-7 validation gates failed")


if __name__ == "__main__":
    main()

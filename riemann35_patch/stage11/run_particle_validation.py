#!/usr/bin/env python3
"""Stage 11: 16-seed particle validation of both production candidates.

The validation advances the same homogeneous cubic Fokker--Planck dynamics
from six deterministic Gaussian-mixture initial states through ``t/tau = 1``.
It compares

* the finite principal-axis Gaussian-mixture map developed in Stage 9, and
* the exact-OU/guarded Grad--HyQMOM/Gaussian--GQMOM map developed in Stage 10,

against independent particle ensembles.  The primary comparison uses the
unclipped continuous cubic-FP drift for all three paths.  Particle histories
are paired with their own initial sample and shifted to the analytic initial
moment vector.  This removes persistent initial Monte-Carlo offset without
mixing the sixteen statistically independent seeds in time.

Every closure step and every Grad limiter value is exported.  The script makes
no global claim that either closure dominates: errors are reported by physical
case and by observable.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage11")

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    HYQMOM_35_INDICES,
    finite_gaussian_mixture_fp_step,
    grad_hyqmom_fp_step,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
    moments_35_from_particles,
    particle_cubic_fp_step,
    realizability_margin_35,
    sample_gaussian_mixture,
)
from riemann35_patch.stage10.run_general_realizability_audit import (  # noqa: E402
    AuditState,
    deterministic_states,
)


POSITION = {index: position for position, index in enumerate(HYQMOM_35_INDICES)}
SELECTED_CASES = (
    "stage9_correlated",
    "rare_hot_anisotropic_w0.02_r25",
    "counterstream_ma20",
    "crossing_ma20",
    "rare_beam_ma20",
    "counterstream_ma100",
)
METRICS = (
    "rho",
    "momentum_x",
    "momentum_y",
    "momentum_z",
    "energy_trace",
    "M200",
    "M020",
    "M002",
    "M300",
    "M030",
    "M003",
    "M400",
    "M040",
    "M004",
    "M110",
    "M101",
    "M011",
    "M210",
    "stress_norm",
    "qx",
    "qy",
    "qz",
    "heat_flux_norm",
    "realizability_margin",
)
PLOT_METRICS = (
    ("M200", r"$M_{200}$"),
    ("M300", r"$M_{300}$"),
    ("M400", r"$M_{400}$"),
    ("stress_norm", r"$\|\boldsymbol{\sigma}\|_F$"),
    ("heat_flux_norm", r"$\|\mathbf{q}\|$"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=20_260_810)
    parser.add_argument("--seed-stride", type=int, default=104_729)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_states() -> list[AuditState]:
    states = {state.name: state for state in deterministic_states()}
    missing = [name for name in SELECTED_CASES if name not in states]
    if missing:
        raise RuntimeError(f"Stage-10 state definitions missing: {missing}")
    return [states[name] for name in SELECTED_CASES]


def sample_steps(steps: int, sample_every: int) -> list[int]:
    result = [0]
    result.extend(range(sample_every, steps + 1, sample_every))
    if result[-1] != steps:
        result.append(steps)
    return result


def diagnostics_from_moments(moments: Sequence[float]) -> np.ndarray:
    vector = np.asarray(moments, dtype=float)
    state = macroscopic_state(vector)
    energy = sum(
        vector[POSITION[index]]
        for index in ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    )
    return np.asarray(
        [
            state.rho,
            vector[POSITION[(1, 0, 0)]],
            vector[POSITION[(0, 1, 0)]],
            vector[POSITION[(0, 0, 1)]],
            energy,
            vector[POSITION[(2, 0, 0)]],
            vector[POSITION[(0, 2, 0)]],
            vector[POSITION[(0, 0, 2)]],
            vector[POSITION[(3, 0, 0)]],
            vector[POSITION[(0, 3, 0)]],
            vector[POSITION[(0, 0, 3)]],
            vector[POSITION[(4, 0, 0)]],
            vector[POSITION[(0, 4, 0)]],
            vector[POSITION[(0, 0, 4)]],
            vector[POSITION[(1, 1, 0)]],
            vector[POSITION[(1, 0, 1)]],
            vector[POSITION[(0, 1, 1)]],
            vector[POSITION[(2, 1, 0)]],
            np.linalg.norm(state.stress),
            state.heat_flux[0],
            state.heat_flux[1],
            state.heat_flux[2],
            np.linalg.norm(state.heat_flux),
            realizability_margin_35(vector),
        ],
        dtype=float,
    )


def particle_task(task: tuple) -> dict[str, object]:
    (
        state,
        particles_per_seed,
        seed,
        steps,
        dt,
        tau,
        prandtl,
        requested_samples,
    ) = task
    velocities = sample_gaussian_mixture(
        state.components, particles=particles_per_seed, seed=seed
    )
    rng = np.random.default_rng(seed + 1_000_003)
    moments_history = [moments_35_from_particles(velocities)]
    maximum_energy_drift = 0.0
    maximum_momentum_drift = 0.0
    minimum_alpha = np.inf
    requested = set(requested_samples[1:])
    start = time.perf_counter()
    for step in range(1, steps + 1):
        velocities, diagnostics = particle_cubic_fp_step(
            velocities,
            dt=dt,
            tau=tau,
            rng=rng,
            prandtl=prandtl,
            limit_peculiar_speed=False,
            enforce_sample_invariants=True,
        )
        maximum_energy_drift = max(maximum_energy_drift, abs(diagnostics.energy_drift))
        maximum_momentum_drift = max(
            maximum_momentum_drift, abs(diagnostics.momentum_drift)
        )
        minimum_alpha = min(minimum_alpha, diagnostics.alpha)
        if step in requested:
            moments_history.append(moments_35_from_particles(velocities))
    return {
        "case": state.name,
        "seed": seed,
        "moments": np.asarray(moments_history),
        "maximum_energy_drift": maximum_energy_drift,
        "maximum_momentum_drift": maximum_momentum_drift,
        "minimum_alpha": float(minimum_alpha),
        "elapsed_seconds": time.perf_counter() - start,
    }


def closure_task(task: tuple) -> dict[str, object]:
    state, method, steps, dt, tau, prandtl, requested_samples = task
    moments = mixture_of_gaussians_moments_35(state.components)
    initial = moments.copy()
    history = [moments.copy()]
    per_step: list[dict[str, float | int]] = []
    requested = set(requested_samples[1:])
    start = time.perf_counter()
    status = "REACHED_FINAL_TIME"
    message = ""
    for step in range(1, steps + 1):
        try:
            if method == "stage9_finite_map":
                moments, diagnostics = finite_gaussian_mixture_fp_step(
                    moments,
                    dt,
                    tau,
                    prandtl=prandtl,
                    speed_cap=np.inf,
                )
                record = {
                    "step": step,
                    "time": step * dt,
                    "lambda": 1.0,
                    "realizability_margin": float(
                        diagnostics.realizability_margin
                    ),
                    "reconstruction_relative_residual": float(
                        diagnostics.reconstruction_relative_residual
                    ),
                }
            elif method == "guarded_grad_map":
                moments, diagnostics = grad_hyqmom_fp_step(
                    moments, dt, tau, prandtl=prandtl
                )
                record = {
                    "step": step,
                    "time": step * dt,
                    "lambda": float(diagnostics.limiter_fraction),
                    "realizability_margin": float(
                        diagnostics.realizability_margin
                    ),
                    "negative_mass_fraction": float(
                        diagnostics.negative_mass_fraction
                    ),
                    "minimum_hankel_margin": float(
                        diagnostics.minimum_hankel_margin
                    ),
                    "source_norm": float(diagnostics.source_norm),
                    "nonlinear_source_norm": float(
                        diagnostics.nonlinear_source_norm
                    ),
                }
            else:
                raise ValueError(f"unknown method {method}")
            per_step.append(record)
            if step in requested:
                history.append(moments.copy())
        except Exception as error:  # pragma: no cover - exercised by audit failures
            status = "FAILED"
            message = f"{type(error).__name__}: {error}"
            break

    final_state = macroscopic_state(moments)
    initial_state = macroscopic_state(initial)
    nonlinear_work = sum(
        float(row.get("nonlinear_source_norm", 0.0)) * dt for row in per_step
    )
    removed_work = sum(
        (1.0 - float(row["lambda"]))
        * float(row.get("nonlinear_source_norm", 0.0))
        * dt
        for row in per_step
    )
    return {
        "case": state.name,
        "method": method,
        "status": status,
        "message": message,
        "completed_steps": len(per_step),
        "moments": np.asarray(history),
        "per_step": per_step,
        "elapsed_seconds": time.perf_counter() - start,
        "minimum_realizability_margin": float(
            min(
                [realizability_margin_35(initial)]
                + [float(row["realizability_margin"]) for row in per_step]
            )
        ),
        "limited_steps": sum(float(row["lambda"]) < 1.0 - 1.0e-12 for row in per_step),
        "minimum_lambda": float(
            min((float(row["lambda"]) for row in per_step), default=np.nan)
        ),
        "weighted_removed_nonlinear_fraction": float(
            removed_work / max(nonlinear_work, 1.0e-30)
        ),
        "mass_drift": abs(final_state.rho - initial_state.rho),
        "momentum_drift": float(
            np.linalg.norm(final_state.velocity - initial_state.velocity)
        ),
        "temperature_drift": abs(final_state.theta - initial_state.theta),
    }


def dimensionless_moments(history: np.ndarray, initial: np.ndarray) -> np.ndarray:
    state = macroscopic_state(initial)
    scales = np.asarray(
        [state.rho * state.theta ** (sum(index) / 2.0) for index in HYQMOM_35_INDICES]
    )
    return history / np.maximum(scales[None, :], 1.0e-14)


def error_summary(
    model: np.ndarray,
    reference: np.ndarray,
    reference_sem: np.ndarray,
    initial: np.ndarray,
) -> dict[str, object]:
    model_scaled = dimensionless_moments(model, initial)
    reference_scaled = dimensionless_moments(reference, initial)
    difference = model_scaled - reference_scaled
    state_l2 = np.linalg.norm(difference) / max(
        np.linalg.norm(reference_scaled), 1.0e-14
    )
    change = reference_scaled - reference_scaled[0]
    change_l2 = np.linalg.norm(difference) / max(np.linalg.norm(change), 1.0e-14)

    model_diagnostics = np.asarray([diagnostics_from_moments(row) for row in model])
    reference_diagnostics = np.asarray(
        [diagnostics_from_moments(row) for row in reference]
    )
    # Propagate moment SEM to observable SEM only for raw-moment observables.
    raw_metric_positions = {
        "rho": (0, 0, 0),
        "momentum_x": (1, 0, 0),
        "momentum_y": (0, 1, 0),
        "momentum_z": (0, 0, 1),
        "M200": (2, 0, 0),
        "M020": (0, 2, 0),
        "M002": (0, 0, 2),
        "M300": (3, 0, 0),
        "M030": (0, 3, 0),
        "M003": (0, 0, 3),
        "M400": (4, 0, 0),
        "M040": (0, 4, 0),
        "M004": (0, 0, 4),
        "M110": (1, 1, 0),
        "M101": (1, 0, 1),
        "M011": (0, 1, 1),
        "M210": (2, 1, 0),
    }
    metric_results: dict[str, dict[str, float | None]] = {}
    for metric in (
        "M200",
        "M300",
        "M400",
        "M110",
        "M210",
        "stress_norm",
        "heat_flux_norm",
    ):
        column = METRICS.index(metric)
        model_values = model_diagnostics[:, column]
        reference_values = reference_diagnostics[:, column]
        delta = model_values - reference_values
        reference_norm = np.linalg.norm(reference_values)
        relaxation_norm = np.linalg.norm(reference_values - reference_values[0])
        if metric in raw_metric_positions:
            sem_values = reference_sem[:, POSITION[raw_metric_positions[metric]]]
            final_sem = float(sem_values[-1])
            final_z = (
                float(delta[-1] / final_sem) if final_sem > 1.0e-14 else None
            )
        else:
            final_z = None
        metric_results[metric] = {
            "history_relative_l2": float(
                np.linalg.norm(delta) / max(reference_norm, 1.0e-14)
            ),
            "relaxation_change_relative_l2": float(
                np.linalg.norm(delta) / max(relaxation_norm, 1.0e-14)
            ),
            "final_absolute_error": float(delta[-1]),
            "final_relative_error": float(
                delta[-1] / max(abs(reference_values[-1]), 1.0e-14)
            ),
            "final_seed_z_score": final_z,
        }
    return {
        "all_35_dimensionless_history_relative_l2": float(state_l2),
        "all_35_relaxation_change_relative_l2": float(change_l2),
        "physical_observables": metric_results,
    }


def write_history_csv(
    path: Path,
    times: np.ndarray,
    stage9: np.ndarray,
    grad: np.ndarray,
    particle: np.ndarray,
    particle_sem: np.ndarray,
) -> None:
    diagnostic_histories = {
        "stage9": np.asarray([diagnostics_from_moments(row) for row in stage9]),
        "grad": np.asarray([diagnostics_from_moments(row) for row in grad]),
        "particle": np.asarray([diagnostics_from_moments(row) for row in particle]),
    }
    fields = ["sample", "time"]
    for prefix in ("stage9", "grad", "particle"):
        fields.extend(f"{prefix}_{metric}" for metric in METRICS)
    fields.extend(f"particle_sem_M{''.join(map(str, index))}" for index in HYQMOM_35_INDICES)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample, current_time in enumerate(times):
            row: dict[str, float | int] = {"sample": sample, "time": float(current_time)}
            for prefix, values in diagnostic_histories.items():
                for column, metric in enumerate(METRICS):
                    row[f"{prefix}_{metric}"] = float(values[sample, column])
            for position, index in enumerate(HYQMOM_35_INDICES):
                row[f"particle_sem_M{''.join(map(str, index))}"] = float(
                    particle_sem[sample, position]
                )
            writer.writerow(row)


def write_per_step_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_case_plot(
    path: Path,
    case: str,
    times: np.ndarray,
    stage9: np.ndarray,
    grad: np.ndarray,
    particle_seed_diagnostics: np.ndarray,
    particle_mean: np.ndarray,
    grad_per_step: list[dict[str, float | int]],
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "text.color": "black",
            "axes.labelcolor": "black",
        }
    )
    stage9_diag = np.asarray([diagnostics_from_moments(row) for row in stage9])
    grad_diag = np.asarray([diagnostics_from_moments(row) for row in grad])
    particle_diag = np.asarray([diagnostics_from_moments(row) for row in particle_mean])
    t95 = 2.13144955
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.8))
    for panel, (metric, label) in enumerate(PLOT_METRICS):
        axis = axes.flat[panel]
        column = METRICS.index(metric)
        seed_values = particle_seed_diagnostics[:, :, column]
        mean = np.mean(seed_values, axis=0)
        sem = np.std(seed_values, axis=0, ddof=1) / np.sqrt(seed_values.shape[0])
        axis.fill_between(
            times,
            mean - t95 * sem,
            mean + t95 * sem,
            color="0.88",
            label="Particle 95% seed CI" if panel == 0 else None,
        )
        axis.plot(
            times,
            particle_diag[:, column],
            "o-k",
            markersize=2.2,
            linewidth=0.9,
            label="16-seed particle mean" if panel == 0 else None,
        )
        axis.plot(
            times,
            stage9_diag[:, column],
            "--",
            color="#cc3311",
            linewidth=1.35,
            label="Stage-9 finite map" if panel == 0 else None,
        )
        axis.plot(
            times,
            grad_diag[:, column],
            "-",
            color="#0077bb",
            linewidth=1.25,
            label="Guarded Grad/GQMOM" if panel == 0 else None,
        )
        axis.set_xlabel(r"Time, $t/\tau$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
        axis.text(
            0.02,
            0.96,
            f"({chr(97 + panel)})",
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
        )

    axis = axes.flat[5]
    step_times = np.asarray([float(row["time"]) for row in grad_per_step])
    lambdas = np.asarray([float(row["lambda"]) for row in grad_per_step])
    axis.plot(step_times, lambdas, color="#117864", linewidth=1.1)
    axis.set_ylim(-0.03, 1.05)
    axis.set_xlabel(r"Time, $t/\tau$")
    axis.set_ylabel(r"Nonlinear-source fraction, $\lambda$")
    axis.grid(alpha=0.22)
    axis.text(
        0.02,
        0.96,
        "(f)",
        transform=axis.transAxes,
        va="top",
        fontweight="bold",
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(case.replace("_", " "), y=0.915, fontsize=10.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def make_overview_plot(path: Path, summaries: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    cases = list(summaries)
    labels = (
        "Corr.",
        "Rare-hot",
        "CS 20",
        "Cross 20",
        "Rare beam",
        "CS 100",
    )
    stage9_errors = [
        summaries[case]["methods"]["stage9_finite_map"][
            "all_35_dimensionless_history_relative_l2"
        ]
        for case in cases
    ]
    grad_errors = [
        summaries[case]["methods"]["guarded_grad_map"][
            "all_35_dimensionless_history_relative_l2"
        ]
        for case in cases
    ]
    limited = [
        summaries[case]["closure_diagnostics"]["guarded_grad_map"]["limited_steps"]
        / summaries[case]["closure_diagnostics"]["guarded_grad_map"]["completed_steps"]
        for case in cases
    ]
    removed = [
        summaries[case]["closure_diagnostics"]["guarded_grad_map"][
            "weighted_removed_nonlinear_fraction"
        ]
        for case in cases
    ]
    x = np.arange(len(cases))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8))
    width = 0.37
    axes[0].bar(x - width / 2, stage9_errors, width, color="#cc3311", label="Stage 9")
    axes[0].bar(x + width / 2, grad_errors, width, color="#0077bb", label="Guarded Grad")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("35-moment history relative error")
    axes[0].set_xticks(x, labels)
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)
    axes[1].bar(x - width / 2, limited, width, color="#117864", label="limited-step fraction")
    axes[1].bar(x + width / 2, removed, width, color="#7d3c98", label="removed nonlinear work")
    axes[1].set_ylabel("fraction")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_arguments()
    if min(args.particles, args.seeds, args.workers, args.sample_every) <= 0:
        raise SystemExit("particle, seed, worker, and sampling counts must be positive")
    if min(args.dt, args.final_time, args.tau) <= 0.0:
        raise SystemExit("dt, final-time, and tau must be positive")
    steps = int(round(args.final_time / args.dt))
    if not np.isclose(steps * args.dt, args.final_time, rtol=0.0, atol=1.0e-12):
        raise SystemExit("final-time must be an integer multiple of dt")
    requested_samples = sample_steps(steps, args.sample_every)
    times = np.asarray(requested_samples, dtype=float) * args.dt / args.tau
    states = selected_states()
    seeds = [args.seed_base + args.seed_stride * index for index in range(args.seeds)]
    args.output.mkdir(parents=True, exist_ok=True)

    particle_tasks = [
        (
            state,
            args.particles,
            seed,
            steps,
            args.dt,
            args.tau,
            args.prandtl,
            requested_samples,
        )
        for state in states
        for seed in seeds
    ]
    print(
        f"Stage 11 particle reference: {len(states)} cases x {args.seeds} seeds x "
        f"{args.particles} particles, {steps} steps, workers={args.workers}",
        flush=True,
    )
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        particle_results = list(executor.map(particle_task, particle_tasks))
    print(f"particle ensemble completed in {time.perf_counter() - start:.1f} s", flush=True)

    closure_tasks = [
        (state, method, steps, args.dt, args.tau, args.prandtl, requested_samples)
        for state in states
        for method in ("stage9_finite_map", "guarded_grad_map")
    ]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(closure_tasks))) as executor:
        closure_results = list(executor.map(closure_task, closure_tasks))
    print(f"closure paths completed in {time.perf_counter() - start:.1f} s", flush=True)

    particles_by_case: dict[str, list[dict[str, object]]] = {state.name: [] for state in states}
    for result in particle_results:
        particles_by_case[str(result["case"])].append(result)
    closures_by_case: dict[str, dict[str, dict[str, object]]] = {
        state.name: {} for state in states
    }
    for result in closure_results:
        closures_by_case[str(result["case"])][str(result["method"])] = result

    summary_cases: dict[str, object] = {}
    raw_seed_archive: dict[str, np.ndarray] = {}
    for state in states:
        initial = mixture_of_gaussians_moments_35(state.components)
        seed_rows = sorted(particles_by_case[state.name], key=lambda row: int(row["seed"]))
        raw_histories = np.asarray([row["moments"] for row in seed_rows])
        aligned_histories = raw_histories - raw_histories[:, :1, :] + initial[None, None, :]
        particle_mean = np.mean(aligned_histories, axis=0)
        particle_sem = np.std(aligned_histories, axis=0, ddof=1) / np.sqrt(args.seeds)
        particle_seed_diagnostics = np.asarray(
            [
                [diagnostics_from_moments(moment_vector) for moment_vector in history]
                for history in aligned_histories
            ]
        )
        raw_seed_archive[f"{state.name}_aligned_moments"] = aligned_histories
        raw_seed_archive[f"{state.name}_raw_moments"] = raw_histories

        case_closures = closures_by_case[state.name]
        for method, result in case_closures.items():
            if result["status"] != "REACHED_FINAL_TIME":
                raise RuntimeError(
                    f"{state.name} {method} failed: {result['message']}"
                )
            if np.asarray(result["moments"]).shape != particle_mean.shape:
                raise RuntimeError(f"{state.name} {method} history length mismatch")

        method_summaries = {
            method: error_summary(
                np.asarray(result["moments"]),
                particle_mean,
                particle_sem,
                initial,
            )
            for method, result in case_closures.items()
        }
        closure_diagnostics = {
            method: {
                key: value
                for key, value in result.items()
                if key not in ("moments", "per_step", "case", "method")
            }
            for method, result in case_closures.items()
        }
        particle_diagnostics = {
            "maximum_energy_drift": float(
                max(float(row["maximum_energy_drift"]) for row in seed_rows)
            ),
            "maximum_momentum_drift": float(
                max(float(row["maximum_momentum_drift"]) for row in seed_rows)
            ),
            "minimum_alpha": float(min(float(row["minimum_alpha"]) for row in seed_rows)),
            "elapsed_seconds_sum": float(
                sum(float(row["elapsed_seconds"]) for row in seed_rows)
            ),
        }
        summary_cases[state.name] = {
            "family": state.family,
            "components": [
                {
                    "weight": float(weight),
                    "mean": np.asarray(mean).tolist(),
                    "covariance": np.asarray(covariance).tolist(),
                }
                for weight, mean, covariance in state.components
            ],
            "initial_realizability_margin": realizability_margin_35(initial),
            "methods": method_summaries,
            "closure_diagnostics": closure_diagnostics,
            "particle_diagnostics": particle_diagnostics,
        }

        write_history_csv(
            args.output / f"stage11_{state.name}_history.csv",
            times,
            np.asarray(case_closures["stage9_finite_map"]["moments"]),
            np.asarray(case_closures["guarded_grad_map"]["moments"]),
            particle_mean,
            particle_sem,
        )
        for method, result in case_closures.items():
            write_per_step_csv(
                args.output / f"stage11_{state.name}_{method}_per_step.csv",
                list(result["per_step"]),
            )
        make_case_plot(
            args.output / f"stage11_{state.name}.png",
            state.name,
            times,
            np.asarray(case_closures["stage9_finite_map"]["moments"]),
            np.asarray(case_closures["guarded_grad_map"]["moments"]),
            particle_seed_diagnostics,
            particle_mean,
            list(case_closures["guarded_grad_map"]["per_step"]),
        )

    np.savez_compressed(args.output / "stage11_particle_seed_histories.npz", **raw_seed_archive)
    make_overview_plot(args.output / "stage11_overview.png", summary_cases)
    summary = {
        "schema": "riemann35-cubic-fp-stage11-particle-validation-v1",
        "model": {
            "drift": "continuous cubic Fokker-Planck without particle speed clipping",
            "particle_integrator": (
                "exact OU factor plus explicit nonlinear drift, antithetic noise, "
                "and sample invariant projection"
            ),
            "stage9_speed_cap": "disabled (infinite) for model consistency",
            "grad_guard": "exact OU base plus maximal realizable nonlinear fraction",
        },
        "controls": {
            "cases": list(SELECTED_CASES),
            "particles_per_seed": args.particles,
            "independent_seeds": args.seeds,
            "seed_values": seeds,
            "workers": args.workers,
            "steps": steps,
            "dt_over_tau": args.dt / args.tau,
            "final_time_over_tau": args.final_time / args.tau,
            "sample_every_steps": args.sample_every,
            "confidence_interval": "Student-t 95% interval over independent seed histories",
            "particle_alignment": (
                "paired change from each seed's own t=0 sample, shifted to the exact "
                "analytic initial moment vector"
            ),
        },
        "cases": summary_cases,
    }
    (args.output / "stage11_particle_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Stage 11 case-wise 35-moment history errors:")
    for case, result in summary_cases.items():
        stage9 = result["methods"]["stage9_finite_map"][
            "all_35_dimensionless_history_relative_l2"
        ]
        grad = result["methods"]["guarded_grad_map"][
            "all_35_dimensionless_history_relative_l2"
        ]
        guard = result["closure_diagnostics"]["guarded_grad_map"]
        print(
            f"  {case:38s} Stage9={stage9:.3%} Grad={grad:.3%} "
            f"limited={guard['limited_steps']}/{guard['completed_steps']} "
            f"lambda_min={guard['minimum_lambda']:.3e}",
            flush=True,
        )
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()

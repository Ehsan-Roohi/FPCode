#!/usr/bin/env python3
"""Stage 15: particle-count refinement for the rare-beam kinetic reference."""

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

from hyqmom_fp import mixture_of_gaussians_moments_35  # noqa: E402
from riemann35_patch.stage10.run_general_realizability_audit import (  # noqa: E402
    deterministic_states,
)
from riemann35_patch.stage11.run_particle_validation import (  # noqa: E402
    POSITION,
    particle_task,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage11", type=Path, required=True)
    parser.add_argument("--stage14", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=31_415_927)
    return parser.parse_args()


def mean_sem(histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.mean(histories, axis=0),
        np.std(histories, axis=0, ddof=1) / np.sqrt(histories.shape[0]),
    )


def comparison_row(
    label: str,
    mean: np.ndarray,
    sem: np.ndarray,
    qmc_mean: np.ndarray,
    qmc_sem: np.ndarray,
) -> dict[str, float | str]:
    position = POSITION[(4, 0, 0)]
    history_error = float(
        np.linalg.norm(mean[:, position] - qmc_mean[:, position])
        / max(np.linalg.norm(qmc_mean[:, position]), 1.0e-14)
    )
    difference = float(mean[-1, position] - qmc_mean[-1, position])
    combined_sem = float(np.hypot(sem[-1, position], qmc_sem[-1, position]))
    return {
        "particle_ensemble": label,
        "M400_history_error_vs_qmc": history_error,
        "final_M400_particle_mean": float(mean[-1, position]),
        "final_M400_particle_sem": float(sem[-1, position]),
        "final_M400_qmc_mean": float(qmc_mean[-1, position]),
        "final_M400_qmc_sem": float(qmc_sem[-1, position]),
        "final_M400_difference": difference,
        "final_M400_combined_z": difference / max(combined_sem, 1.0e-14),
    }


def main() -> None:
    args = arguments()
    state = {state.name: state for state in deterministic_states()}["rare_beam_ma20"]
    steps = int(round(args.final_time / args.dt))
    requested = [0, *range(args.sample_every, steps + 1, args.sample_every)]
    seeds = [args.seed_base + 104_729 * index for index in range(args.seeds)]
    tasks = [
        (
            state,
            args.particles,
            seed,
            steps,
            args.dt,
            1.0,
            2.0 / 3.0,
            requested,
        )
        for seed in seeds
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(particle_task, tasks))
    analytic_initial = mixture_of_gaussians_moments_35(state.components)
    raw = np.asarray([result["moments"] for result in results])
    aligned = raw - raw[:, :1, :] + analytic_initial[None, None, :]
    refined_mean, refined_sem = mean_sem(aligned)

    stage11_archive = np.load(args.stage11 / "stage11_particle_seed_histories.npz")
    baseline = stage11_archive["rare_beam_ma20_aligned_moments"]
    baseline_mean, baseline_sem = mean_sem(baseline)
    qmc_archive = np.load(args.stage14 / "stage14_qmc_scramble_histories.npz")
    qmc_mean = qmc_archive["rare_beam_ma20_mean"]
    qmc_sem = qmc_archive["rare_beam_ma20_sem"]
    rows = [
        comparison_row("16x100k", baseline_mean, baseline_sem, qmc_mean, qmc_sem),
        comparison_row(
            f"{args.seeds}x{args.particles // 1000}k",
            refined_mean,
            refined_sem,
            qmc_mean,
            qmc_sem,
        ),
    ]
    summary = {
        "schema": "riemann35-stage15-particle-count-refinement-v1",
        "case": "rare_beam_ma20",
        "controls": {
            "dt_over_tau": args.dt,
            "final_time_over_tau": args.final_time,
            "refined_particles_per_seed": args.particles,
            "refined_independent_seeds": args.seeds,
            "seed_values": seeds,
        },
        "comparisons": rows,
        "diagnostics": {
            "maximum_energy_drift": max(result["maximum_energy_drift"] for result in results),
            "maximum_momentum_drift": max(result["maximum_momentum_drift"] for result in results),
            "minimum_alpha": min(result["minimum_alpha"] for result in results),
            "elapsed_seconds_sum": sum(result["elapsed_seconds"] for result in results),
        },
    }
    (args.output / "stage15_particle_count_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "stage15_particle_count_histories.npz",
        refined_aligned_histories=aligned,
        refined_mean=refined_mean,
        refined_sem=refined_sem,
        baseline_mean=baseline_mean,
        baseline_sem=baseline_sem,
        qmc_mean=qmc_mean,
        qmc_sem=qmc_sem,
    )
    with (args.output / "stage15_particle_count_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stage 15: particle-count refinement",
        "",
        "The Stage-11 rare-beam reference averages sixteen independent 100k-particle paths. Because the cubic-FP coefficients are nonlinear functions of empirical moments, averaging many small ensembles need not equal one continuum solution. Eight independent 200k-particle paths test this finite-ensemble bias at fixed total work order.",
        "",
        "| Ensemble | M400 history vs QMC | final particle mean | final particle SEM | final QMC mean | combined z |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['particle_ensemble']} | {row['M400_history_error_vs_qmc']:.2%} | "
            f"{row['final_M400_particle_mean']:.6g} | {row['final_M400_particle_sem']:.3e} | "
            f"{row['final_M400_qmc_mean']:.6g} | {row['final_M400_combined_z']:.2f} |"
        )
    lines.extend(
        [
            "",
            "A systematic movement toward the positive-QMC result under particles-per-seed refinement indicates nonlinear finite-ensemble bias; unchanged means with shrinking SEM would instead flag a remaining QMC discretization bias.",
        ]
    )
    (args.output / "STAGE15_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

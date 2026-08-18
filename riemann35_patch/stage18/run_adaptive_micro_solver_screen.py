#!/usr/bin/env python3
"""Stage 18: size a positive kinetic micro-solver for troubled cells."""

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
    error_summary,
)
from riemann35_patch.stage14.run_positive_qmc_reference import run_qmc  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage11", type=Path, required=True)
    parser.add_argument("--stage14", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_810)
    parser.add_argument("--replicates", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    state = {state.name: state for state in deterministic_states()}["rare_beam_ma20"]
    point_counts = (512, 1024, 2048, 4096, 8192)
    tasks = []
    for points in point_counts:
        for replicate in range(args.replicates):
            tasks.append(
                (
                    state.name,
                    state.components,
                    f"micro_{points}_replicate_{replicate}",
                    points,
                    1,
                    args.dt,
                    args.final_time,
                    args.sample_every,
                    args.seed + 15_485_863 * replicate,
                )
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(run_qmc, tasks))

    particle_archive = np.load(args.stage11 / "stage11_particle_seed_histories.npz")
    particle_seeds = particle_archive["rare_beam_ma20_aligned_moments"]
    particle = np.mean(particle_seeds, axis=0)
    particle_sem = np.std(particle_seeds, axis=0, ddof=1) / np.sqrt(particle_seeds.shape[0])
    qmc_archive = np.load(args.stage14 / "stage14_qmc_scramble_histories.npz")
    qmc = qmc_archive["rare_beam_ma20_mean"]
    qmc_sem = qmc_archive["rare_beam_ma20_sem"]
    initial = mixture_of_gaussians_moments_35(state.components)
    rows = []
    histories = {}
    for points in point_counts:
        point_results = [
            result for result in results if result["points_per_component"] == points
        ]
        replicate_histories = np.asarray(
            [result["aligned_history"] for result in point_results]
        )
        history = np.mean(replicate_histories, axis=0)
        histories[f"micro_{points}_replicates"] = replicate_histories
        histories[f"micro_{points}_mean"] = history
        against_particle = error_summary(history, particle, particle_sem, initial)
        against_qmc = error_summary(history, qmc, qmc_sem, initial)
        particle_error = against_particle["physical_observables"]["M400"]["history_relative_l2"]
        qmc_error = against_qmc["physical_observables"]["M400"]["history_relative_l2"]
        position = POSITION[(4, 0, 0)]
        m400_replicates = replicate_histories[:, :, position]
        scramble_spread = float(
            np.linalg.norm(np.std(m400_replicates, axis=0, ddof=1))
            / max(np.linalg.norm(history[:, position]), 1.0e-14)
        )
        rows.append(
            {
                "points_per_component": points,
                "total_positive_nodes": point_results[0]["total_positive_nodes"],
                "independent_scramblings": args.replicates,
                "elapsed_seconds_sum": sum(result["elapsed_seconds"] for result in point_results),
                "M400_error_vs_particle": particle_error,
                "M400_error_vs_qmc": qmc_error,
                "M400_scramble_relative_spread": scramble_spread,
                "passes_3pct_reference_envelope": max(particle_error, qmc_error, scramble_spread) < 0.03,
                "minimum_realizability_margin": min(result["minimum_raw_realizability_margin"] for result in point_results),
                "maximum_energy_drift": max(result["maximum_energy_drift"] for result in point_results),
            }
        )
    rows.sort(key=lambda row: row["points_per_component"])
    passing = [row for row in rows if row["passes_3pct_reference_envelope"]]
    selected = min(passing, key=lambda row: row["total_positive_nodes"]) if passing else None
    summary = {
        "schema": "riemann35-stage18-adaptive-micro-solver-screen-v1",
        "case": "rare_beam_ma20",
        "interpretation": "kinetic microstate sizing, not an instantaneous 35-moment closure",
        "gate": "M400 history error and independent-scrambling spread all < 3%",
        "candidates": rows,
        "selected_minimum": selected,
    }
    (args.output / "stage18_micro_solver_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output / "stage18_micro_solver_histories.npz", **histories)
    with (args.output / "stage18_micro_solver_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stage 18: adaptive positive micro-solver sizing",
        "",
        "Stage 17 proves that the missing tail is not identifiable from the instantaneous 35-moment state. This screen therefore retains a positive kinetic microstate only in troubled cells. It measures the smallest Sobol ensemble that meets the 3% rare-beam gate; it is not presented as a new algebraic closure.",
        "",
        "| points/component | total nodes | scrambles | runtime sum | M400 vs particle | M400 vs QMC | scramble spread | min margin | gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['points_per_component']} | {row['total_positive_nodes']} | {row['independent_scramblings']} | {row['elapsed_seconds_sum']:.2f}s | "
            f"{row['M400_error_vs_particle']:.2%} | {row['M400_error_vs_qmc']:.2%} | "
            f"{row['M400_scramble_relative_spread']:.2%} | "
            f"{row['minimum_realizability_margin']:.3e} | {'PASS' if row['passes_3pct_reference_envelope'] else 'FAIL'} |"
        )
    if selected is None:
        lines.extend(["", "No tested microstate size passes; the adaptive path is not yet viable."])
    else:
        lines.extend(
            [
                "",
                f"The smallest passing proof-of-concept uses {selected['total_positive_nodes']} positive nodes. The next engineering step is a troubled-cell activation/deactivation rule and conservative projection between this persistent microstate and HyQMOM-35. Spatial publication tests remain blocked until that coupling is implemented and benchmarked.",
            ]
        )
    (args.output / "STAGE18_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

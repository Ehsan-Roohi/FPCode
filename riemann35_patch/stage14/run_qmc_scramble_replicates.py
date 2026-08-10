#!/usr/bin/env python3
"""Independent scrambling control for the finest Stage-14 QMC reference."""

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
from riemann35_patch.stage14.run_positive_qmc_reference import (  # noqa: E402
    CASES,
    run_qmc,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage11", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--points-per-component", type=int, default=32768)
    parser.add_argument("--dt", type=float, default=1.25e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-every-base", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_810)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.replicates < 2:
        raise ValueError("at least two independent scramblings are required")
    states = {state.name: state for state in deterministic_states()}
    # run_qmc expects a base dt and an integer refinement.  Use refinement=2
    # so that its output sampling remains on the Stage-11 0.025-tau grid.
    base_dt = 2.0 * args.dt
    tasks = []
    for case in CASES:
        for replicate in range(args.replicates):
            tasks.append(
                (
                    case,
                    states[case].components,
                    f"replicate_{replicate}",
                    args.points_per_component,
                    2,
                    base_dt,
                    args.final_time,
                    args.sample_every_base,
                    args.seed + 15_485_863 * replicate,
                )
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(run_qmc, tasks))

    particle_archive = np.load(args.stage11 / "stage11_particle_seed_histories.npz")
    summary = {
        "schema": "riemann35-stage14-qmc-scramble-control-v1",
        "replicates": args.replicates,
        "points_per_component": args.points_per_component,
        "dt_over_tau": args.dt,
        "cases": {},
    }
    archive = {}
    rows = []
    for case in CASES:
        case_results = sorted(
            (result for result in results if result["case"] == case),
            key=lambda result: result["label"],
        )
        histories = np.asarray([result["aligned_history"] for result in case_results])
        mean = np.mean(histories, axis=0)
        sem = np.std(histories, axis=0, ddof=1) / np.sqrt(args.replicates)
        particle_seeds = particle_archive[f"{case}_aligned_moments"]
        particle_mean = np.mean(particle_seeds, axis=0)
        particle_sem = np.std(particle_seeds, axis=0, ddof=1) / np.sqrt(particle_seeds.shape[0])
        initial = mixture_of_gaussians_moments_35(states[case].components)
        comparison = error_summary(mean, particle_mean, particle_sem, initial)
        position = POSITION[(4, 0, 0)]
        final_difference = float(mean[-1, position] - particle_mean[-1, position])
        combined_sem = float(
            np.hypot(sem[-1, position], particle_sem[-1, position])
        )
        individual_m400 = histories[:, :, position]
        history_scramble_cv = float(
            np.linalg.norm(np.std(individual_m400, axis=0, ddof=1))
            / max(np.linalg.norm(mean[:, position]), 1.0e-14)
        )
        entry = {
            "M400_history_error_vs_stage11_particle": comparison["physical_observables"]["M400"]["history_relative_l2"],
            "M400_history_scramble_relative_spread": history_scramble_cv,
            "final_M400_qmc_mean": float(mean[-1, position]),
            "final_M400_qmc_sem": float(sem[-1, position]),
            "final_M400_particle_mean": float(particle_mean[-1, position]),
            "final_M400_particle_sem": float(particle_sem[-1, position]),
            "final_M400_difference": final_difference,
            "final_M400_combined_z": final_difference / max(combined_sem, 1.0e-14),
            "minimum_raw_realizability_margin": min(
                result["minimum_raw_realizability_margin"] for result in case_results
            ),
            "maximum_energy_drift": max(
                result["maximum_energy_drift"] for result in case_results
            ),
        }
        summary["cases"][case] = entry
        rows.append({"case": case, **entry})
        archive[f"{case}_replicate_histories"] = histories
        archive[f"{case}_mean"] = mean
        archive[f"{case}_sem"] = sem

    (args.output / "stage14_qmc_scramble_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output / "stage14_qmc_scramble_histories.npz", **archive)
    with (args.output / "stage14_qmc_scramble_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stage 14 independent-scrambling control",
        "",
        f"The finest positive QMC rule was repeated with {args.replicates} independent Sobol scramblings. Uncertainty below is computed across scrambling replicates, separately from the Stage-11 particle-seed uncertainty.",
        "",
        "| Case | M400 history vs particle | QMC scramble spread | final QMC SEM | final particle SEM | combined z | min margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['M400_history_error_vs_stage11_particle']:.2%} | "
            f"{row['M400_history_scramble_relative_spread']:.2%} | {row['final_M400_qmc_sem']:.3e} | "
            f"{row['final_M400_particle_sem']:.3e} | {row['final_M400_combined_z']:.2f} | "
            f"{row['minimum_raw_realizability_margin']:.3e} |"
        )
    lines.extend(
        [
            "",
            "The positive QMC mean is accepted as the current kinetic reference only when the node/time refinement differences and the independent-scrambling spread are both small relative to the closure error being measured.",
        ]
    )
    (args.output / "STAGE14_SCRAMBLE_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

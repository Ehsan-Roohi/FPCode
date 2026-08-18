#!/usr/bin/env python3
"""Stage 14: positive low-discrepancy kinetic-reference convergence audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage14")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    mixture_of_gaussians_moments_35,
    moments_35_from_qmc,
    qmc_cubic_fp_step,
    realizability_margin_35,
    sample_gaussian_mixture_qmc,
)
from riemann35_patch.stage10.run_general_realizability_audit import (  # noqa: E402
    deterministic_states,
)
from riemann35_patch.stage11.run_particle_validation import (  # noqa: E402
    POSITION,
    error_summary,
)
from riemann35_patch.stage11.summarize_stage11 import history_error  # noqa: E402


CASES = ("rare_beam_ma20", "rare_hot_anisotropic_w0.02_r25")
CONFIGURATIONS = (
    ("qmc_8k_dt", 8192, 1),
    ("qmc_32k_dt", 32768, 1),
    ("qmc_32k_half_dt", 32768, 2),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage11", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_810)
    return parser.parse_args()


def selected_states():
    states = {state.name: state for state in deterministic_states()}
    return {name: states[name] for name in CASES}


def run_qmc(task: tuple) -> dict[str, object]:
    case, components, label, points_per_component, refinement, base_dt, final_time, sample_every, seed = task
    dt = base_dt / refinement
    steps = int(round(final_time / dt))
    requested_stride = sample_every * refinement
    velocities, weights = sample_gaussian_mixture_qmc(
        components,
        points_per_component=points_per_component,
        seed=seed,
    )
    raw_history = [moments_35_from_qmc(velocities, weights)]
    minimum_margin = realizability_margin_35(raw_history[0])
    minimum_alpha = np.inf
    maximum_energy_drift = 0.0
    maximum_momentum_drift = 0.0
    start = time.perf_counter()
    for step in range(1, steps + 1):
        velocities, diagnostics = qmc_cubic_fp_step(
            velocities,
            weights,
            dt=dt,
            tau=1.0,
            seed=seed + 1_000_003 + 104_729 * step,
            prandtl=2.0 / 3.0,
        )
        minimum_alpha = min(minimum_alpha, diagnostics.alpha)
        maximum_energy_drift = max(maximum_energy_drift, abs(diagnostics.energy_drift))
        maximum_momentum_drift = max(
            maximum_momentum_drift, abs(diagnostics.momentum_drift)
        )
        if step % requested_stride == 0 or step == steps:
            moments = moments_35_from_qmc(velocities, weights)
            raw_history.append(moments)
            minimum_margin = min(minimum_margin, realizability_margin_35(moments))
    raw = np.asarray(raw_history)
    analytic_initial = mixture_of_gaussians_moments_35(components)
    aligned = raw - raw[0] + analytic_initial
    return {
        "case": case,
        "label": label,
        "points_per_component": points_per_component,
        "total_positive_nodes": len(weights),
        "minimum_weight": float(np.min(weights)),
        "dt_over_tau": dt,
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - start,
        "minimum_raw_realizability_margin": float(minimum_margin),
        "minimum_alpha": float(minimum_alpha),
        "maximum_energy_drift": float(maximum_energy_drift),
        "maximum_momentum_drift": float(maximum_momentum_drift),
        "raw_history": raw,
        "aligned_history": aligned,
    }


def m400_relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    position = POSITION[(4, 0, 0)]
    return float(
        np.linalg.norm(model[:, position] - reference[:, position])
        / max(np.linalg.norm(reference[:, position]), 1.0e-14)
    )


def make_plot(path: Path, times: np.ndarray, cases: dict[str, dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "text.color": "black",
            "axes.labelcolor": "black",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.45))
    styles = {
        "qmc_8k_dt": ("--", "#cc3311"),
        "qmc_32k_dt": ("-.", "#0077bb"),
        "qmc_32k_half_dt": ("-", "#117864"),
    }
    for panel, case in enumerate(CASES):
        axis = axes[panel]
        data = cases[case]
        particle = np.asarray(data["particle_seed_histories"])
        particle_m400 = particle[:, :, POSITION[(4, 0, 0)]]
        mean = np.mean(particle_m400, axis=0)
        sem = np.std(particle_m400, axis=0, ddof=1) / np.sqrt(particle_m400.shape[0])
        axis.fill_between(times, mean - 2.13145 * sem, mean + 2.13145 * sem, color="0.87", label="Particle 95% seed CI")
        axis.plot(times, mean, "o-k", markersize=2.0, linewidth=0.8, label="16-seed particle mean")
        for label, history in data["histories"].items():
            linestyle, color = styles[label]
            axis.plot(
                times,
                history[:, POSITION[(4, 0, 0)]],
                linestyle,
                color=color,
                linewidth=1.25,
                label=label.replace("qmc_", "QMC ").replace("_", " "),
            )
        axis.set_xlabel(r"Time, $t/\tau$")
        axis.set_ylabel(r"$M_{400}$")
        axis.set_title(case.replace("_", " "))
        axis.grid(alpha=0.22)
        axis.text(0.02, 0.96, f"({chr(97 + panel)})", transform=axis.transAxes, va="top", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = arguments()
    states = selected_states()
    tasks = []
    for case, state in states.items():
        for label, points, refinement in CONFIGURATIONS:
            tasks.append(
                (
                    case,
                    state.components,
                    label,
                    points,
                    refinement,
                    args.dt,
                    args.final_time,
                    args.sample_every,
                    args.seed,
                )
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        raw_results = list(executor.map(run_qmc, tasks))

    particle_archive = np.load(args.stage11 / "stage11_particle_seed_histories.npz")
    case_data: dict[str, dict[str, object]] = {}
    summary: dict[str, object] = {
        "schema": "riemann35-stage14-positive-qmc-reference-v1",
        "controls": {
            "base_dt_over_tau": args.dt,
            "final_time_over_tau": args.final_time,
            "positive_measure": True,
            "initial_and_noise_rule": "scrambled Sobol with exact affine mean/covariance correction and seeded exchangeable noise assignment",
            "alignment": "paired QMC change shifted to analytic initial moments",
        },
        "cases": {},
    }
    histories_for_archive = {}
    rows = []
    for case, state in states.items():
        particle_seeds = particle_archive[f"{case}_aligned_moments"]
        particle_mean = np.mean(particle_seeds, axis=0)
        particle_sem = np.std(particle_seeds, axis=0, ddof=1) / np.sqrt(particle_seeds.shape[0])
        initial = mixture_of_gaussians_moments_35(state.components)
        results = {result["label"]: result for result in raw_results if result["case"] == case}
        finest = np.asarray(results["qmc_32k_half_dt"]["aligned_history"])
        entries = {}
        histories = {}
        for label, result in results.items():
            aligned = np.asarray(result["aligned_history"])
            histories[label] = aligned
            histories_for_archive[f"{case}_{label}_aligned"] = aligned
            histories_for_archive[f"{case}_{label}_raw"] = np.asarray(result["raw_history"])
            against_particle = error_summary(aligned, particle_mean, particle_sem, initial)
            convergence = history_error(aligned, finest, initial)
            entry = {
                **{
                    key: value
                    for key, value in result.items()
                    if key not in ("raw_history", "aligned_history")
                },
                "M400_error_vs_stage11_particle": against_particle["physical_observables"]["M400"]["history_relative_l2"],
                "M400_difference_vs_finest_qmc": m400_relative_l2(aligned, finest),
                "degree4_rmse_vs_finest_qmc": convergence["fourth_order_componentwise_dimensionless_rmse"],
                "final_M400_particle_z_score": against_particle["physical_observables"]["M400"]["final_seed_z_score"],
            }
            entries[label] = entry
            rows.append(entry)
        summary["cases"][case] = {"configurations": entries}
        case_data[case] = {
            "particle_seed_histories": particle_seeds,
            "histories": histories,
        }

    (args.output / "stage14_qmc_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output / "stage14_qmc_histories.npz", **histories_for_archive)
    with (args.output / "stage14_qmc_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    times = np.linspace(0.0, args.final_time, next(iter(case_data.values()))["histories"]["qmc_8k_dt"].shape[0])
    make_plot(args.output / "stage14_qmc_M400_convergence.png", times, case_data)

    lines = [
        "# Stage 14: positive low-discrepancy kinetic reference",
        "",
        "A positive weighted Sobol ensemble was advanced with the same unclipped cubic-FP coefficient solve and finite collision map used by the Stage-11 particles. The component probabilities are exact, and the QMC mean/covariance are corrected exactly. Higher moments are accepted only after node-count and time-step convergence checks.",
        "",
        "| Case | Configuration | Positive nodes | dt/tau | M400 vs particle | M400 vs finest QMC | degree-4 RMSE vs finest | min margin |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        for label, _, _ in CONFIGURATIONS:
            entry = summary["cases"][case]["configurations"][label]
            lines.append(
                f"| {case} | {label} | {entry['total_positive_nodes']} | {entry['dt_over_tau']:.6g} | "
                f"{entry['M400_error_vs_stage11_particle']:.2%} | {entry['M400_difference_vs_finest_qmc']:.2%} | "
                f"{entry['degree4_rmse_vs_finest_qmc']:.3e} | {entry['minimum_raw_realizability_margin']:.3e} |"
            )
    lines.extend(
        [
            "",
            "The QMC path is a positive kinetic discretization, not a closed 35-moment model. Agreement under both node refinement and time-step refinement is the acceptance test. Differences from the Stage-11 seed mean are reported separately and must be interpreted with that reference's seed uncertainty.",
        ]
    )
    (args.output / "STAGE14_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

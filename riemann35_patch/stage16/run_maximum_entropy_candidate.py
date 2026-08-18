#!/usr/bin/env python3
"""Stage 16: screen a positive maximum-entropy M5/M6 closure."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage16")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    maximum_entropy_fp_step,
    mixture_of_gaussians_moments_35,
)
from riemann35_patch.stage10.run_general_realizability_audit import (  # noqa: E402
    deterministic_states,
)
from riemann35_patch.stage11.run_particle_validation import (  # noqa: E402
    POSITION,
    error_summary,
)
from riemann35_patch.stage11.summarize_stage11 import history_error  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage11", type=Path, required=True)
    parser.add_argument("--stage14", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=10)
    return parser.parse_args()


def run_candidate(task: tuple) -> dict[str, object]:
    nodes_per_dimension, components, dt, final_time, sample_every = task
    steps = int(round(final_time / dt))
    moments = mixture_of_gaussians_moments_35(components)
    history = [moments.copy()]
    minimum_margin = np.inf
    minimum_lambda = 1.0
    limited_steps = 0
    maximum_iterations = 0
    maximum_constraint_residual = 0.0
    maximum_moment_residual = 0.0
    minimum_probability = np.inf
    status = "REACHED_FINAL_TIME"
    message = ""
    dual_parameters = None
    start = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            moments, diagnostics = maximum_entropy_fp_step(
                moments,
                dt,
                1.0,
                nodes_per_dimension=nodes_per_dimension,
                initial_parameters=dual_parameters,
            )
        except Exception as error:  # pragma: no cover - audit failure path
            status = "FAILED"
            message = f"{type(error).__name__}: {error}"
            break
        dual_parameters = diagnostics.dual_parameters
        minimum_margin = min(minimum_margin, diagnostics.realizability_margin)
        minimum_lambda = min(minimum_lambda, diagnostics.limiter_fraction)
        limited_steps += int(diagnostics.limiter_fraction < 1.0 - 1.0e-12)
        maximum_iterations = max(maximum_iterations, diagnostics.iterations)
        maximum_constraint_residual = max(
            maximum_constraint_residual, diagnostics.scaled_constraint_residual
        )
        maximum_moment_residual = max(
            maximum_moment_residual, diagnostics.relative_moment_residual
        )
        minimum_probability = min(minimum_probability, diagnostics.minimum_probability)
        if step % sample_every == 0 or step == steps:
            history.append(moments.copy())
    return {
        "nodes_per_dimension": nodes_per_dimension,
        "total_support_nodes": 2 * nodes_per_dimension**3,
        "status": status,
        "message": message,
        "completed_steps": step if status == "REACHED_FINAL_TIME" else step - 1,
        "elapsed_seconds": time.perf_counter() - start,
        "minimum_realizability_margin": float(minimum_margin),
        "limited_steps": limited_steps,
        "minimum_lambda": float(minimum_lambda),
        "maximum_newton_iterations": maximum_iterations,
        "maximum_scaled_constraint_residual": float(maximum_constraint_residual),
        "maximum_raw_moment_residual": float(maximum_moment_residual),
        "minimum_probability": float(minimum_probability),
        "history": np.asarray(history),
    }


def make_plot(path: Path, times: np.ndarray, histories: dict[str, np.ndarray], particle: np.ndarray, particle_sem: np.ndarray, qmc: np.ndarray, qmc_sem: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    position = POSITION[(4, 0, 0)]
    axis.fill_between(times, particle[:, position] - 2.13145 * particle_sem[:, position], particle[:, position] + 2.13145 * particle_sem[:, position], color="0.88", label="Particle 95% seed CI")
    axis.plot(times, particle[:, position], "o-k", markersize=2.0, linewidth=0.8, label="Stage-11 particle mean")
    axis.fill_between(times, qmc[:, position] - 2.0 * qmc_sem[:, position], qmc[:, position] + 2.0 * qmc_sem[:, position], color="#ccebc5", alpha=0.6, label="QMC scramble band")
    axis.plot(times, qmc[:, position], color="#117864", linewidth=1.2, label="Positive QMC mean")
    colors = {"maxent_n4": "#cc3311", "maxent_n6": "#0077bb", "maxent_n8": "#aa4499"}
    for label, history in histories.items():
        axis.plot(times[: history.shape[0]], history[:, position], "--", color=colors[label], linewidth=1.2, label=label.replace("_", " "))
    axis.set_xlabel(r"Time, $t/\tau$")
    axis.set_ylabel(r"$M_{400}$")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=7.5, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = arguments()
    state = {state.name: state for state in deterministic_states()}["rare_beam_ma20"]
    tasks = [
        (nodes, state.components, args.dt, args.final_time, args.sample_every)
        for nodes in (4, 6, 8)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(run_candidate, tasks))

    particle_archive = np.load(args.stage11 / "stage11_particle_seed_histories.npz")
    particle_seeds = particle_archive["rare_beam_ma20_aligned_moments"]
    particle = np.mean(particle_seeds, axis=0)
    particle_sem = np.std(particle_seeds, axis=0, ddof=1) / np.sqrt(particle_seeds.shape[0])
    qmc_archive = np.load(args.stage14 / "stage14_qmc_scramble_histories.npz")
    qmc = qmc_archive["rare_beam_ma20_mean"]
    qmc_sem = qmc_archive["rare_beam_ma20_sem"]
    initial = mixture_of_gaussians_moments_35(state.components)
    entries = {}
    rows = []
    histories = {}
    for result in results:
        label = f"maxent_n{result['nodes_per_dimension']}"
        history = np.asarray(result["history"])
        histories[label] = history
        entry = {key: value for key, value in result.items() if key != "history"}
        if result["status"] == "REACHED_FINAL_TIME":
            particle_error = error_summary(history, particle, particle_sem, initial)
            qmc_error = error_summary(history, qmc, qmc_sem, initial)
            degree4 = history_error(history, qmc, initial)
            entry.update(
                {
                    "M400_error_vs_particle": particle_error["physical_observables"]["M400"]["history_relative_l2"],
                    "M400_error_vs_qmc": qmc_error["physical_observables"]["M400"]["history_relative_l2"],
                    "degree4_rmse_vs_qmc": degree4["fourth_order_componentwise_dimensionless_rmse"],
                }
            )
            entry["passes_3pct_gate"] = max(entry["M400_error_vs_particle"], entry["M400_error_vs_qmc"]) < 0.03
        else:
            entry.update({"M400_error_vs_particle": None, "M400_error_vs_qmc": None, "degree4_rmse_vs_qmc": None, "passes_3pct_gate": False})
        entries[label] = entry
        rows.append(entry)
    summary = {
        "schema": "riemann35-stage16-positive-maximum-entropy-v1",
        "case": "rare_beam_ma20",
        "gate": "max(M400 history error vs particle, vs QMC) < 3%",
        "candidates": entries,
    }
    (args.output / "stage16_maxent_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output / "stage16_maxent_histories.npz", **histories)
    with (args.output / "stage16_maxent_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    times = np.linspace(0.0, args.final_time, particle.shape[0])
    make_plot(args.output / "stage16_maxent_M400.png", times, histories, particle, particle_sem, qmc, qmc_sem)
    lines = [
        "# Stage 16: positive maximum-entropy closure",
        "",
        "The candidate uses an adaptive two-population quadrature only as positive support, then solves the discrete entropy dual so that every retained moment through degree four is matched. M5/M6 are evaluated from the resulting positive weights. Promotion requires M400 error below 3% against both independent reference constructions.",
        "",
        "| Candidate | status | support | M400 vs particle | M400 vs QMC | degree-4 RMSE vs QMC | limited | min margin | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, entry in entries.items():
        p = "--" if entry["M400_error_vs_particle"] is None else f"{entry['M400_error_vs_particle']:.2%}"
        q = "--" if entry["M400_error_vs_qmc"] is None else f"{entry['M400_error_vs_qmc']:.2%}"
        d4 = "--" if entry["degree4_rmse_vs_qmc"] is None else f"{entry['degree4_rmse_vs_qmc']:.3e}"
        lines.append(f"| {label} | {entry['status']} | {entry['total_support_nodes']} | {p} | {q} | {d4} | {entry['limited_steps']} | {entry['minimum_realizability_margin']:.3e} | {'PASS' if entry['passes_3pct_gate'] else 'FAIL'} |")
    lines.extend(["", "No candidate is promoted unless it passes the reference-envelope gate and remains positive, conservative, and realizable for the full collision time."])
    (args.output / "STAGE16_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

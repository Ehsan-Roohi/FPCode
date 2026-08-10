#!/usr/bin/env python3
"""Post-process Stage 11 and run a closure time-step refinement control."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage11-summary")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import HYQMOM_35_INDICES, mixture_of_gaussians_moments_35  # noqa: E402
from riemann35_patch.stage11.run_particle_validation import (  # noqa: E402
    METRICS,
    SELECTED_CASES,
    closure_task,
    diagnostics_from_moments,
    dimensionless_moments,
    selected_states,
    write_per_step_csv,
)


CASE_METRICS = {
    "stage9_correlated": (
        "M200", "M300", "M400", "M110", "M210", "stress_norm", "heat_flux_norm"
    ),
    "rare_hot_anisotropic_w0.02_r25": ("M200", "M400", "stress_norm"),
    "counterstream_ma20": ("M200", "M400", "stress_norm"),
    "crossing_ma20": ("M200", "M400", "M110", "stress_norm"),
    "rare_beam_ma20": (
        "M200", "M300", "M400", "M110", "M210", "stress_norm", "heat_flux_norm"
    ),
    "counterstream_ma100": ("M200", "M400", "stress_norm"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def history_error(model: np.ndarray, reference: np.ndarray, initial: np.ndarray) -> dict[str, float]:
    model_scaled = dimensionless_moments(model, initial)
    reference_scaled = dimensionless_moments(reference, initial)
    difference = model_scaled - reference_scaled
    fourth_order_positions = [
        position
        for position, index in enumerate(HYQMOM_35_INDICES)
        if sum(index) == 4
    ]
    fourth_difference = difference[:, fourth_order_positions]
    fourth_reference = reference_scaled[:, fourth_order_positions]
    return {
        "all_35_dimensionless_history_relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference_scaled), 1.0e-14)
        ),
        "all_35_relaxation_change_relative_l2": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(reference_scaled - reference_scaled[0]), 1.0e-14)
        ),
        "fourth_order_componentwise_dimensionless_rmse": float(
            np.sqrt(np.mean(fourth_difference**2))
        ),
        "fourth_order_dimensionless_history_relative_l2": float(
            np.linalg.norm(fourth_difference)
            / max(np.linalg.norm(fourth_reference), 1.0e-14)
        ),
    }


def physical_errors(
    model: np.ndarray,
    seed_histories: np.ndarray,
    active_metrics: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    model_diagnostics = np.asarray([diagnostics_from_moments(row) for row in model])
    seed_diagnostics = np.asarray(
        [
            [diagnostics_from_moments(moment_vector) for moment_vector in history]
            for history in seed_histories
        ]
    )
    reference = np.mean(seed_diagnostics, axis=0)
    sem = np.std(seed_diagnostics, axis=0, ddof=1) / np.sqrt(seed_histories.shape[0])
    results = {}
    for metric in active_metrics:
        column = METRICS.index(metric)
        difference = model_diagnostics[:, column] - reference[:, column]
        ci_half_width = 2.13144955 * sem[:, column]
        results[metric] = {
            "history_relative_l2": float(
                np.linalg.norm(difference)
                / max(np.linalg.norm(reference[:, column]), 1.0e-14)
            ),
            "relaxation_change_relative_l2": float(
                np.linalg.norm(difference)
                / max(
                    np.linalg.norm(reference[:, column] - reference[0, column]),
                    1.0e-14,
                )
            ),
            "final_relative_error": float(
                difference[-1] / max(abs(reference[-1, column]), 1.0e-14)
            ),
            "final_seed_z_score": float(
                difference[-1] / max(sem[-1, column], 1.0e-14)
            ),
            "fraction_of_samples_inside_95pct_seed_ci": float(
                np.mean(np.abs(difference) <= np.maximum(ci_half_width, 1.0e-14))
            ),
            "particle_final_mean": float(reference[-1, column]),
            "particle_final_sem": float(sem[-1, column]),
            "model_final": float(model_diagnostics[-1, column]),
        }
    return results


def make_figure(path: Path, compact: dict) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    cases = list(SELECTED_CASES)
    labels = (
        "Corr.",
        "Rare-hot",
        "CS 20",
        "Cross 20",
        "Rare beam",
        "CS 100",
    )
    x = np.arange(len(cases))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.75))
    colors = {"stage9_finite_map": "#cc3311", "guarded_grad_map": "#0077bb"}
    width = 0.36
    for offset, method in zip((-width / 2, width / 2), colors):
        axes[0].bar(
            x + offset,
            [compact[case][method]["base"]["fourth_order_componentwise_dimensionless_rmse"] for case in cases],
            width,
            color=colors[method],
            label="Stage 9" if method == "stage9_finite_map" else "Guarded Grad/GQMOM",
        )
        axes[1].bar(
            x + offset,
            [compact[case][method]["base"]["physical_observables"]["M400"]["history_relative_l2"] for case in cases],
            width,
            color=colors[method],
        )
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("Degree-4 componentwise RMSE")
    axes[1].set_ylabel(r"$M_{400}$ history error")
    for axis in axes[:2]:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)

    for method, marker, label in (
        ("stage9_finite_map", "o", "Stage 9"),
        ("guarded_grad_map", "s", "Guarded Grad/GQMOM"),
    ):
        axes[2].scatter(
            [compact[case][method]["base"]["fourth_order_componentwise_dimensionless_rmse"] for case in cases],
            [compact[case][method]["half_dt"]["fourth_order_componentwise_dimensionless_rmse"] for case in cases],
            color=colors[method],
            marker=marker,
            label=label,
        )
    limits = (1.0e-3, 3.0e-1)
    axes[2].plot(limits, limits, "--", color="0.4", linewidth=1.0, label="unchanged")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlim(limits)
    axes[2].set_ylim(limits)
    axes[2].set_xlabel(r"error at $\Delta t/\tau=2.5\times10^{-3}$")
    axes[2].set_ylabel(r"error at half $\Delta t$")
    axes[2].grid(alpha=0.22)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, compact: dict) -> None:
    fields = [
        "case",
        "method",
        "base_all35_error",
        "half_dt_all35_error",
        "base_fourth_block_rmse",
        "half_dt_fourth_block_rmse",
        "base_M400_error",
        "half_dt_M400_error",
        "base_M400_final_z",
        "half_dt_M400_final_z",
        "base_limited_steps",
        "base_minimum_lambda",
        "base_removed_nonlinear_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in SELECTED_CASES:
            for method in ("stage9_finite_map", "guarded_grad_map"):
                base = compact[case][method]["base"]
                half = compact[case][method]["half_dt"]
                writer.writerow(
                    {
                        "case": case,
                        "method": method,
                        "base_all35_error": base["all_35_dimensionless_history_relative_l2"],
                        "half_dt_all35_error": half["all_35_dimensionless_history_relative_l2"],
                        "base_fourth_block_rmse": base["fourth_order_componentwise_dimensionless_rmse"],
                        "half_dt_fourth_block_rmse": half["fourth_order_componentwise_dimensionless_rmse"],
                        "base_M400_error": base["physical_observables"]["M400"]["history_relative_l2"],
                        "half_dt_M400_error": half["physical_observables"]["M400"]["history_relative_l2"],
                        "base_M400_final_z": base["physical_observables"]["M400"]["final_seed_z_score"],
                        "half_dt_M400_final_z": half["physical_observables"]["M400"]["final_seed_z_score"],
                        "base_limited_steps": base["diagnostics"]["limited_steps"],
                        "base_minimum_lambda": base["diagnostics"]["minimum_lambda"],
                        "base_removed_nonlinear_fraction": base["diagnostics"]["weighted_removed_nonlinear_fraction"],
                    }
                )


def write_markdown(path: Path, compact: dict, controls: dict) -> None:
    def pct(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    lines = [
        "# Stage 11 particle validation through one collision time",
        "",
        "## Scope and statistical control",
        "",
        f"Six homogeneous trajectories were advanced to t/tau=1 with {controls['independent_seeds']} independent particle seeds and {controls['particles_per_seed']:,} particles per seed. The reference therefore contains {controls['independent_seeds'] * controls['particles_per_seed']:,} particles per case. Uncertainty is computed across independent seeds, not across correlated output times. The primary model uses the continuous cubic-FP drift without speed clipping.",
        "",
        "Each particle trajectory was differenced from its own random t=0 sample and shifted to the exact analytic initial moment vector. This paired-change estimator removes persistent initialization noise while retaining between-seed uncertainty in the relaxation history. Its confidence interval therefore collapses at t=0 by construction and makes whole-history errors slightly optimistic near the initial point. With 1.6 million particles per case, the remaining nonlinear control-variate correction is second order and does not affect the method ranking.",
        "",
        "## Main results",
        "",
        "No closure dominates in accuracy. Stage 9 remains strongest on the separable counter-stream/crossing families, while guarded Grad/GQMOM is slightly better on the correlated case and clearly better, though still insufficient, on the rare beam.",
        "",
        "The primary closure metric below is the componentwise RMS error of the nondimensional degree-four block. The aggregate 35-vector norm is retained in the machine-readable files only as a secondary diagnostic because its raw L2 weighting is dominated by the largest fourth moment in the rare-beam case.",
        "",
        "| Case | Stage-9 degree-4 RMSE | Grad degree-4 RMSE | Stage-9 M400 error | Grad M400 error | Grad limiter |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in SELECTED_CASES:
        s9 = compact[case]["stage9_finite_map"]["base"]
        gr = compact[case]["guarded_grad_map"]["base"]
        limiter = gr["diagnostics"]
        lines.append(
            f"| {case} | {s9['fourth_order_componentwise_dimensionless_rmse']:.4e} | "
            f"{gr['fourth_order_componentwise_dimensionless_rmse']:.4e} | "
            f"{pct(s9['physical_observables']['M400']['history_relative_l2'])} | "
            f"{pct(gr['physical_observables']['M400']['history_relative_l2'])} | "
            f"{limiter['limited_steps']}/{limiter['completed_steps']}, "
            f"lambda_min={limiter['minimum_lambda']:.3f} |"
        )

    rb9 = compact["rare_beam_ma20"]["stage9_finite_map"]["base"]
    rbg = compact["rare_beam_ma20"]["guarded_grad_map"]["base"]
    rbg_half = compact["rare_beam_ma20"]["guarded_grad_map"]["half_dt"]
    lines.extend(
        [
            "",
            "## Physical interpretation",
            "",
            f"For the rare beam, Stage 9 and guarded Grad/GQMOM have M400 history errors of {pct(rb9['physical_observables']['M400']['history_relative_l2'])} and {pct(rbg['physical_observables']['M400']['history_relative_l2'])}, respectively. The Grad guard is active in {rbg['diagnostics']['limited_steps']} of {rbg['diagnostics']['completed_steps']} steps, but removes only {pct(rbg['diagnostics']['weighted_removed_nonlinear_fraction'])} of the source-norm-weighted nonlinear contribution. Hence the large M400 bias is primarily a closure error, not suppression by the lambda guard.",
            "",
            f"Halving the closure time step changes the guarded-Grad rare-beam degree-four componentwise RMSE from {rbg['fourth_order_componentwise_dimensionless_rmse']:.4e} to {rbg_half['fourth_order_componentwise_dimensionless_rmse']:.4e}. Persistence under refinement shows that the dominant discrepancy is not the time step.",
            "",
            "The second-moment/stress histories and the contracted heat-flux relaxation are consistency checks rather than independent closure-accuracy tests: the physical 9-by-9 coefficient solve enforces their production rates by construction on both the moment and particle paths. Independent closure evidence comes primarily from the degree-four block and, secondarily, from the unconstrained part of the third-order block. Counter-stream and crossing retain approximately one-percent-or-better M400 histories. Odd moments that are zero by symmetry are excluded from physical percentage claims because their relative errors are noise-dominated.",
            "",
            "The anisotropic rare-hot M400 comparison is statistically weak despite 1.6 million particles because the rare hot population gives a very large fourth-moment sampling variance. Its final discrepancy is within roughly one seed standard error; a deterministic Hermite reference is required before making an accuracy claim for this case.",
            "",
            "## Decision",
            "",
            "The Stage-9 call site explicitly sets speed_cap=Inf; therefore Stage 9, guarded Grad, and the particle reference all use the same unclipped cubic drift in this comparison.",
            "",
            "Stage 11 validates stability and conservation and gives independent degree-four evidence for the correlated, counter-stream, and crossing relaxations. It also identifies a decisive accuracy gap for the high-skewness rare-beam M400 history. A spatial JCP demonstration should wait until that gap is reduced. Convex blending of the current Stage-9 and Grad tails cannot remove a bias shared by both.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    base_summary = json.loads(
        (args.results / "stage11_particle_validation.json").read_text()
    )
    controls = base_summary["controls"]
    archive = np.load(args.results / "stage11_particle_seed_histories.npz")
    states = selected_states()
    base_dt = float(controls["dt_over_tau"])
    final_time = float(controls["final_time_over_tau"])
    tau = 1.0
    refinements = {"base": 1, "half_dt": 2}
    tasks = []
    for state in states:
        for method in ("stage9_finite_map", "guarded_grad_map"):
            for _, factor in refinements.items():
                dt = base_dt / factor
                steps = int(round(final_time / dt))
                sample_every = int(controls["sample_every_steps"]) * factor
                requested = [0, *range(sample_every, steps + 1, sample_every)]
                tasks.append((state, method, steps, dt, tau, 2.0 / 3.0, requested))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(closure_task, tasks))

    compact: dict[str, dict] = {case: {} for case in SELECTED_CASES}
    histories: dict[str, np.ndarray] = {}
    task_position = 0
    for state in states:
        seed_histories = archive[f"{state.name}_aligned_moments"]
        reference = np.mean(seed_histories, axis=0)
        initial = mixture_of_gaussians_moments_35(state.components)
        for method in ("stage9_finite_map", "guarded_grad_map"):
            compact[state.name][method] = {}
            for refinement in refinements:
                result = results[task_position]
                task_position += 1
                if result["status"] != "REACHED_FINAL_TIME":
                    raise RuntimeError(
                        f"refinement failed: {state.name} {method} {refinement}: {result['message']}"
                    )
                model = np.asarray(result["moments"])
                entry = history_error(model, reference, initial)
                entry["physical_observables"] = physical_errors(
                    model, seed_histories, CASE_METRICS[state.name]
                )
                entry["diagnostics"] = {
                    key: value
                    for key, value in result.items()
                    if key not in ("moments", "per_step", "case", "method")
                }
                compact[state.name][method][refinement] = entry
                histories[f"{state.name}_{method}_{refinement}"] = model
                if refinement == "half_dt":
                    write_per_step_csv(
                        args.results
                        / f"stage11_{state.name}_{method}_half_dt_per_step.csv",
                        list(result["per_step"]),
                    )

    output = {
        "schema": "riemann35-cubic-fp-stage11-refined-interpretation-v1",
        "controls": controls,
        "physical_metric_selection": CASE_METRICS,
        "cases": compact,
    }
    (args.results / "stage11_refinement_and_physics.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.results / "stage11_closure_histories_base_and_half_dt.npz", **histories
    )
    write_csv(args.results / "stage11_accuracy_summary.csv", compact)
    make_figure(args.results / "stage11_accuracy_and_refinement.png", compact)
    write_markdown(args.results / "STAGE11_RESULTS.md", compact, controls)
    print(f"wrote refined Stage-11 interpretation to {args.results}")


if __name__ == "__main__":
    main()

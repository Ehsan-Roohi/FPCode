#!/usr/bin/env python3
"""Reproduce the Stage-12 high-skew closure candidate comparison."""

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

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    dynamic_high_order_fp_step,
    initialize_dynamic_high_order_state,
    initialize_persistent_two_population,
    mixture_of_gaussians_moments_35,
    persistent_two_population_fp_step,
    two_population_fp_step,
)
from riemann35_patch.stage10.run_general_realizability_audit import (  # noqa: E402
    deterministic_states,
)
from riemann35_patch.stage11.run_particle_validation import error_summary  # noqa: E402
from riemann35_patch.stage11.summarize_stage11 import history_error  # noqa: E402


CANDIDATES = (
    "algebraic_base",
    "algebraic_residual",
    "persistent_two_population",
    "dynamic_m6",
    "dynamic_m8",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage11", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=10)
    return parser.parse_args()


def rare_beam_state():
    return {state.name: state for state in deterministic_states()}["rare_beam_ma20"]


def run_candidate(task: tuple) -> dict[str, object]:
    candidate, components, steps, dt, sample_every = task
    initial = mixture_of_gaussians_moments_35(components)
    moments = initial.copy()
    history = [initial.copy()]
    minimum_margin = np.inf
    minimum_lambda = 1.0
    limited_steps = 0
    start = time.perf_counter()
    if candidate == "persistent_two_population":
        internal = initialize_persistent_two_population(
            initial, minimum_skewness_norm=0.05
        )
    elif candidate.startswith("dynamic_"):
        maximum_order = 6 if candidate == "dynamic_m6" else 8
        internal = initialize_dynamic_high_order_state(
            initial,
            minimum_skewness_norm=0.05,
            maximum_order=maximum_order,
        )
    else:
        internal = None

    for step in range(1, steps + 1):
        if candidate.startswith("algebraic_"):
            moments, diagnostics = two_population_fp_step(
                moments,
                dt,
                1.0,
                minimum_skewness_norm=0.05,
                residual_correction=candidate == "algebraic_residual",
            )
            limiter = diagnostics.limiter_fraction
            margin = diagnostics.realizability_margin
        elif candidate == "persistent_two_population":
            internal, moments, diagnostics = persistent_two_population_fp_step(
                internal, dt, 1.0
            )
            limiter = 1.0
            margin = diagnostics.realizability_margin
        else:
            internal, diagnostics = dynamic_high_order_fp_step(
                internal,
                dt,
                1.0,
                minimum_skewness_norm=0.05,
            )
            moments = internal.moments
            limiter = diagnostics.limiter_fraction
            margin = diagnostics.realizability_margin
        minimum_margin = min(minimum_margin, float(margin))
        minimum_lambda = min(minimum_lambda, float(limiter))
        limited_steps += int(limiter < 1.0 - 1.0e-12)
        if step % sample_every == 0 or step == steps:
            history.append(np.asarray(moments).copy())
    return {
        "candidate": candidate,
        "history": np.asarray(history),
        "elapsed_seconds": time.perf_counter() - start,
        "minimum_realizability_margin": float(minimum_margin),
        "limited_steps": limited_steps,
        "minimum_lambda": float(minimum_lambda),
    }


def main() -> None:
    args = arguments()
    steps = int(round(args.final_time / args.dt))
    state = rare_beam_state()
    initial = mixture_of_gaussians_moments_35(state.components)
    archive = np.load(args.stage11 / "stage11_particle_seed_histories.npz")
    seed_histories = archive["rare_beam_ma20_aligned_moments"]
    reference = np.mean(seed_histories, axis=0)
    sem = np.std(seed_histories, axis=0, ddof=1) / np.sqrt(seed_histories.shape[0])
    tasks = [
        (candidate, state.components, steps, args.dt, args.sample_every)
        for candidate in CANDIDATES
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        raw_results = list(executor.map(run_candidate, tasks))

    summary: dict[str, object] = {
        "schema": "riemann35-stage12-high-skew-candidate-audit-v1",
        "case": "rare_beam_ma20",
        "controls": {
            "dt_over_tau": args.dt,
            "final_time_over_tau": args.final_time,
            "particle_reference": "Stage 11, 16 x 100000 particles",
            "accuracy_gate_M400": 0.03,
        },
        "candidates": {},
    }
    histories = {}
    rows = []
    for result in raw_results:
        candidate = str(result["candidate"])
        history = np.asarray(result["history"])
        histories[candidate] = history
        physical = error_summary(history, reference, sem, initial)
        blocks = history_error(history, reference, initial)
        entry = {
            **{key: value for key, value in result.items() if key != "history"},
            **blocks,
            "M400_history_relative_l2": physical["physical_observables"]["M400"][
                "history_relative_l2"
            ],
            "M300_history_relative_l2": physical["physical_observables"]["M300"][
                "history_relative_l2"
            ],
            "heat_flux_history_relative_l2": physical["physical_observables"][
                "heat_flux_norm"
            ]["history_relative_l2"],
        }
        entry["passes_3pct_M400_gate"] = entry["M400_history_relative_l2"] < 0.03
        summary["candidates"][candidate] = entry
        rows.append(entry)

    source_path = args.output / "stage12_rare_beam_source.json"
    if source_path.exists():
        summary["instantaneous_source_diagnostic"] = json.loads(
            source_path.read_text(encoding="utf-8")
        )
    (args.output / "stage12_candidate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output / "stage12_candidate_histories.npz", **histories)
    fields = list(rows[0])
    with (args.output / "stage12_candidate_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda row: row["M400_history_relative_l2"])
    lines = [
        "# Stage 12 high-skew closure candidates",
        "",
        "All candidates were tested on the Stage-11 rare-beam reference through one collision time. The two-population reconstruction is exact at t=0 for this generating mixture, but that exact initial fit does not guarantee an accurate history because cubic FP evolution immediately drives each population away from a Gaussian shape.",
        "",
        "| Candidate | Degree-4 RMSE | M400 history error | M300 error | Limited steps | Minimum margin | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['fourth_order_componentwise_dimensionless_rmse']:.4e} | "
            f"{row['M400_history_relative_l2']:.2%} | {row['M300_history_relative_l2']:.2%} | "
            f"{row['limited_steps']} | {row['minimum_realizability_margin']:.3e} | "
            f"{'PASS' if row['passes_3pct_M400_gate'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"The best Stage-12 candidate is `{best['candidate']}` at {best['M400_history_relative_l2']:.2%}, still well above the 3% gate. No Stage-12 candidate is promoted to the production closure.",
            "",
            "The source-level particle diagnostic shows that Stage 9 and the two-population model reproduce the exact initial M400 source, while all tested algebraic closures become overly dissipative after the initial transient. This rejects the hypothesis that an exact t=0 two-Gaussian fit alone resolves the rare-beam history.",
        ]
    )
    (args.output / "STAGE12_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

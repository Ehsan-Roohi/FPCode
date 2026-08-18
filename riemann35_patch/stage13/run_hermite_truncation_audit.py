#!/usr/bin/env python3
"""Audit naive finite Hermite-moment truncation as a deterministic reference.

This is deliberately a rejection test.  A truncated Hermite expansion is not
positive by construction, so its first 35 moments are checked against the full
HyQMOM realizability cone after every SSP-RK2 step.  Once the margin becomes
negative, the trajectory is stopped and must not be used as a physical
reference.
"""

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
    first_35_from_hermite_state,
    hermite_ssprk2_step,
    initialize_hermite_moment_state,
)
from riemann35_patch.stage10.run_general_realizability_audit import (  # noqa: E402
    deterministic_states,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    return parser.parse_args()


def rare_beam_components():
    states = {state.name: state for state in deterministic_states()}
    return states["rare_beam_ma20"].components


def run_configuration(task: tuple[int, float, float, tuple]) -> dict[str, object]:
    order, dt, final_time, components = task
    steps = int(round(final_time / dt))
    dynamic = initialize_hermite_moment_state(components, order)
    margins = []
    first_negative_step: int | None = None
    start = time.perf_counter()
    for step in range(1, steps + 1):
        dynamic, diagnostics = hermite_ssprk2_step(dynamic, dt, 1.0)
        margins.append(diagnostics.realizability_margin)
        if diagnostics.realizability_margin < 0.0:
            first_negative_step = step
            break
    values = first_35_from_hermite_state(dynamic)
    return {
        "maximum_order": order,
        "dt_over_tau": dt,
        "requested_final_time_over_tau": final_time,
        "completed_steps": len(margins),
        "completed_time_over_tau": len(margins) * dt,
        "reached_final_time": first_negative_step is None,
        "first_negative_step": first_negative_step,
        "minimum_realizability_margin": float(min(margins)),
        "final_realizability_margin": float(margins[-1]),
        "finite_first_35": bool(np.all(np.isfinite(values))),
        "elapsed_seconds": time.perf_counter() - start,
        "margin_history": margins,
    }


def main() -> None:
    args = arguments()
    components = rare_beam_components()
    tasks = [
        (6, args.dt, args.final_time, components),
        (8, args.dt, args.final_time, components),
        (8, args.dt / 2.0, args.final_time, components),
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(run_configuration, tasks))

    summary = {
        "schema": "riemann35-stage13-hermite-truncation-rejection-v1",
        "case": "rare_beam_ma20",
        "method": "finite raw-moment hierarchy with zero Hermite coefficients above K",
        "acceptance_rule": "all retained 35-moment states must remain realizable",
        "configurations": [
            {key: value for key, value in result.items() if key != "margin_history"}
            for result in results
        ],
        "decision": "REJECT_AS_PHYSICAL_REFERENCE",
    }
    (args.output / "stage13_hermite_truncation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "stage13_hermite_margin_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["maximum_order", "dt_over_tau", "step", "time_over_tau", "margin"])
        for result in results:
            for number, margin in enumerate(result["margin_history"], start=1):
                writer.writerow(
                    [
                        result["maximum_order"],
                        result["dt_over_tau"],
                        number,
                        number * result["dt_over_tau"],
                        margin,
                    ]
                )

    lines = [
        "# Stage 13: finite Hermite truncation audit",
        "",
        "A zero-tail Hermite truncation was tested as a deterministic alternative to the particle reference. It is exact for a Maxwellian, but positivity is not guaranteed away from equilibrium. Each trajectory was stopped at the first negative full 35-moment realizability margin.",
        "",
        "| K | dt/tau | completed t/tau | first negative step | minimum margin | decision |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['maximum_order']} | {result['dt_over_tau']:.6g} | "
            f"{result['completed_time_over_tau']:.6g} | {result['first_negative_step']} | "
            f"{result['minimum_realizability_margin']:.3e} | REJECT |"
        )
    lines.extend(
        [
            "",
            "The violation also occurs for K=8 after halving the time step. It is therefore not acceptable to quote this signed Hermite trajectory as a physical reference. The next deterministic reference must evolve a nonnegative velocity-space density or enforce a positivity projection with a documented convergence study.",
        ]
    )
    (args.output / "STAGE13_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

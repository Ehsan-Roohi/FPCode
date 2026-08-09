#!/usr/bin/env python3
"""Compare Julia CHyQMOM-M6 homogeneous relaxation with FPCode particles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


QUANTITIES = ("M200", "M300", "M400", "stress_norm", "heat_flux_norm")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--julia", type=Path, required=True)
    parser.add_argument("--particle", type=Path, required=True)
    parser.add_argument("--julia-metrics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline-final-m400", type=float, default=0.09702)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if not rows:
        raise ValueError(f"no rows in {path}")
    if not all(all(math.isfinite(value) for value in row.values()) for row in rows):
        raise ValueError(f"non-finite value in {path}")
    return rows


def read_metrics(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["key"]: row["value"] for row in csv.DictReader(stream)}


def metric_bool(metrics: dict[str, str], key: str) -> bool:
    return metrics[key].strip().lower() == "true"


def history_l2(julia_rows, particle_rows, quantity: str) -> float:
    squared_error = sum(
        (julia[quantity] - particle[quantity]) ** 2
        for julia, particle in zip(julia_rows, particle_rows)
    )
    squared_reference = sum(row[quantity] ** 2 for row in particle_rows)
    return math.sqrt(squared_error / max(squared_reference, 1.0e-28))


def main() -> None:
    args = parse_arguments()
    julia_rows = read_rows(args.julia)
    particle_rows = read_rows(args.particle)
    julia_metrics = read_metrics(args.julia_metrics)
    if len(julia_rows) != len(particle_rows):
        raise SystemExit("Julia and particle histories have different sample counts")
    for julia, particle in zip(julia_rows, particle_rows):
        if int(julia["step"]) != int(particle["step"]) or not math.isclose(
            julia["time"], particle["time"], rel_tol=0.0, abs_tol=1.0e-14
        ):
            raise SystemExit("Julia and particle histories are not time-aligned")

    initial_julia = julia_rows[0]
    final_julia = julia_rows[-1]
    final_particle = particle_rows[-1]
    final_m400_relative = abs(final_julia["M400"] - final_particle["M400"]) / max(
        abs(final_particle["M400"]), 1.0e-14
    )
    raw_reached_final_time = metric_bool(julia_metrics, "raw_reached_final_time")
    projection_count = int(float(julia_metrics["projection_count"]))
    scientific_status = (
        "RAW_CLOSURE_REACHED_FINAL_TIME"
        if raw_reached_final_time
        else "RAW_CLOSURE_FAILED_REALIZABILITY_PROJECTED_DIAGNOSTIC_ONLY"
    )
    summary = {
        "schema": "riemann35-fp-stage2-v2",
        "scientific_status": scientific_status,
        "samples": len(julia_rows),
        "final_time": final_julia["time"],
        "julia_mass_drift": abs(final_julia["rho"] - initial_julia["rho"]),
        "julia_energy_drift": abs(
            final_julia["energy_trace"] - initial_julia["energy_trace"]
        ),
        "final_m400_relative_difference": final_m400_relative,
        "gaussian_tail_baseline_final_m400_difference": args.baseline_final_m400,
        "m400_improved_over_gaussian_tail_baseline": (
            final_m400_relative < args.baseline_final_m400
        ),
        "raw_closure": {
            "reached_final_time": raw_reached_final_time,
            "failure_step": int(float(julia_metrics["raw_failure_step"])),
            "failure_state_margin": float(
                julia_metrics["raw_failure_state_margin"]
            ),
        },
        "realizability_correction": {
            "projection_count": projection_count,
            "projection_fraction": float(julia_metrics["projection_fraction"]),
            "maximum_relative_projection": float(
                julia_metrics["maximum_relative_projection"]
            ),
            "maximum_interior_weight": float(
                julia_metrics["maximum_interior_weight"]
            ),
            "minimum_trial_margin": float(
                julia_metrics["minimum_trial_margin"]
            ),
            "minimum_accepted_margin": float(
                julia_metrics["minimum_accepted_margin"]
            ),
            "final_margin": float(julia_metrics["final_margin"]),
        },
        "history_relative_l2": {
            quantity: history_l2(julia_rows, particle_rows, quantity)
            for quantity in QUANTITIES
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Stage-2 Julia CHyQMOM-M6 vs particle comparison")
    print(f"scientific status:     {scientific_status}")
    print(
        "raw failure step:      "
        f"{summary['raw_closure']['failure_step']} "
        f"(reached_final={raw_reached_final_time})"
    )
    print(
        "projection count:      "
        f"{projection_count} ({summary['realizability_correction']['projection_fraction']:.1%})"
    )
    print(f"samples / final time: {summary['samples']} / {summary['final_time']:.6g}")
    print(f"Julia mass drift:     {summary['julia_mass_drift']:.3e}")
    print(f"Julia energy drift:   {summary['julia_energy_drift']:.3e}")
    print(f"final M400 relative:  {final_m400_relative:.3%}")
    print(
        "Gaussian-tail baseline: "
        f"{args.baseline_final_m400:.3%} (improved={summary['m400_improved_over_gaussian_tail_baseline']})"
    )
    for quantity, value in summary["history_relative_l2"].items():
        print(f"history L2 {quantity:>14}: {value:.3%}")

    if summary["julia_mass_drift"] > 1.0e-12:
        raise SystemExit("FAIL: Julia mass conservation gate")
    if summary["julia_energy_drift"] > 1.0e-10:
        raise SystemExit("FAIL: Julia energy conservation gate")
    print("PASS: Stage-2 operational gates completed.")
    if not raw_reached_final_time:
        print(
            "SCIENTIFIC DIAGNOSTIC: the raw M5/M6 closure is not realizability-"
            "preserving; particle-error metrics use the explicitly projected path."
        )


if __name__ == "__main__":
    main()

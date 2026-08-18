#!/usr/bin/env python3
"""Compare an adaptive Julia CHyQMOM source with FPCode particles."""

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
    parser.add_argument("--stage", type=int, choices=(2, 3, 4, 5, 6), default=None)
    parser.add_argument("--require-m400-improvement", action="store_true")
    parser.add_argument("--max-history-m200", type=float, default=None)
    parser.add_argument("--max-history-stress", type=float, default=None)
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


def finite_metric_or_none(metrics: dict[str, str], key: str) -> float | None:
    value = float(metrics[key])
    return value if math.isfinite(value) else None


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
    adaptive_reached = metric_bool(julia_metrics, "adaptive_reached_final_time")
    source_mode = julia_metrics.get("source_mode", "raw")
    if source_mode not in {"raw", "bounded", "finite", "finite_subcycled"}:
        raise SystemExit(f"unknown Julia source mode: {source_mode}")
    mode_label = source_mode.upper()
    scientific_status = f"{mode_label}_CLOSURE_{'REACHED_FINAL_TIME' if adaptive_reached else 'ADAPTIVE_INTEGRATION_FAILED'}"
    inferred_stage = (
        6
        if source_mode == "finite_subcycled"
        else (4 if source_mode == "finite" else (3 if source_mode == "bounded" else 2))
    )
    stage_number = args.stage if args.stage is not None else inferred_stage
    summary = {
        "schema": (
            "riemann35-fp-stage6-v1"
            if stage_number == 6
            else (
                "riemann35-fp-stage5-v1"
                if stage_number == 5
                else (
                    "riemann35-fp-stage4-v1"
                    if source_mode == "finite"
                    else (
                        "riemann35-fp-stage3-v1"
                        if source_mode == "bounded"
                        else "riemann35-fp-stage2-v4"
                    )
                )
            )
        ),
        "source_mode": source_mode,
        "scientific_status": scientific_status,
        "samples": len(julia_rows),
        "final_time": final_julia["time"],
        "julia_mass_drift": abs(final_julia["rho"] - initial_julia["rho"]),
        "julia_energy_drift": abs(
            final_julia["energy_trace"] - initial_julia["energy_trace"]
        ),
        "julia_momentum_drift": float(julia_metrics["julia_momentum_drift"]),
        "final_m400_relative_difference": final_m400_relative,
        "gaussian_tail_baseline_final_m400_difference": args.baseline_final_m400,
        "m400_improved_over_gaussian_tail_baseline": (
            final_m400_relative < args.baseline_final_m400
        ),
        "legacy_integrator": {
            "max_substeps": 256,
            "failure_step": int(float(julia_metrics["legacy_cap_failure_step"])),
            "failure_state_margin": float(
                julia_metrics["legacy_failure_state_margin"]
            ),
        },
        "adaptive_integrator": {
            "reached_final_time": adaptive_reached,
            "failure_step": int(float(julia_metrics["adaptive_failure_step"])),
            "accepted_microsteps": int(
                float(julia_metrics["adaptive_accepted_microsteps"])
            ),
            "rejected_microsteps": int(
                float(julia_metrics["adaptive_rejected_microsteps"])
            ),
            "minimum_h": float(julia_metrics["adaptive_minimum_h"]),
            "minimum_h_over_dt": float(
                julia_metrics["adaptive_minimum_h_over_dt"]
            ),
            "maximum_source_norm": float(
                julia_metrics["adaptive_maximum_source_norm"]
            ),
            "minimum_trial_margin": finite_metric_or_none(
                julia_metrics, "minimum_trial_margin"
            ),
            "minimum_accepted_margin": float(
                julia_metrics["minimum_accepted_margin"]
            ),
            "final_margin": float(julia_metrics["final_margin"]),
        },
        "finite_map": (
            {
                "minimum_alpha": float(julia_metrics["finite_minimum_alpha"]),
                "maximum_alpha": float(julia_metrics["finite_maximum_alpha"]),
                "maximum_node_c2_over_theta": float(
                    julia_metrics["finite_maximum_node_c2_over_theta"]
                ),
                "maximum_quadrature_residual_norm": finite_metric_or_none(
                    julia_metrics, "finite_maximum_quadrature_residual_norm"
                ),
                "maximum_collision_increment_norm": finite_metric_or_none(
                    julia_metrics, "finite_maximum_collision_increment_norm"
                ),
            }
            if source_mode in {"finite", "finite_subcycled"}
            else None
        ),
        "finite_subcycling": (
            {
                "max_substeps_limit": int(
                    float(julia_metrics["finite_max_substeps_limit"])
                ),
                "maximum_substeps_per_macro": int(
                    float(julia_metrics["finite_maximum_substeps_per_macro"])
                ),
                "retried_macrosteps": int(
                    float(julia_metrics["finite_retried_macrosteps"])
                ),
                "total_restarts": int(
                    float(julia_metrics["finite_total_restarts"])
                ),
                "first_step_ladder": [
                    {
                        "substeps": int(key.split("_")[4]),
                        "success": metric_bool(julia_metrics, key),
                        "attempted_substeps": int(
                            float(julia_metrics[key.replace("_success", "_attempted")])
                        ),
                        "failure_substep": int(
                            float(
                                julia_metrics[
                                    key.replace("_success", "_failure_substep")
                                ]
                            )
                        ),
                        "minimum_margin": finite_metric_or_none(
                            julia_metrics,
                            key.replace("_success", "_minimum_margin"),
                        ),
                    }
                    for key in sorted(
                        (
                            name
                            for name in julia_metrics
                            if name.startswith("finite_first_step_nsub_")
                            and name.endswith("_success")
                        ),
                        key=lambda name: int(name.split("_")[4]),
                    )
                ],
            }
            if source_mode == "finite_subcycled"
            else None
        ),
        "history_relative_l2": {
            quantity: history_l2(julia_rows, particle_rows, quantity)
            for quantity in QUANTITIES
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    adaptive = summary["adaptive_integrator"]
    stage = f"Stage-{stage_number}"
    print(f"{stage} adaptive {source_mode} Julia CHyQMOM vs particle comparison")
    print(f"scientific status:       {scientific_status}")
    print(f"legacy cap failure step: {summary['legacy_integrator']['failure_step']}")
    print(
        "adaptive accepted/rejected: "
        f"{adaptive['accepted_microsteps']} / {adaptive['rejected_microsteps']}"
    )
    print(f"minimum h/dt:           {adaptive['minimum_h_over_dt']:.8e}")
    if source_mode in {"finite", "finite_subcycled"}:
        finite_map = summary["finite_map"]
        print(
            "finite alpha min/max:  "
            f"{finite_map['minimum_alpha']:.8e} / {finite_map['maximum_alpha']:.8e}"
        )
        print(
            "maximum node c2/theta: "
            f"{finite_map['maximum_node_c2_over_theta']:.8e}"
        )
        if finite_map["maximum_quadrature_residual_norm"] is not None:
            print(
                "maximum quadrature residual: "
                f"{finite_map['maximum_quadrature_residual_norm']:.8e}"
            )
        if finite_map["maximum_collision_increment_norm"] is not None:
            print(
                "maximum collision increment: "
                f"{finite_map['maximum_collision_increment_norm']:.8e}"
            )
    if source_mode == "finite_subcycled":
        subcycling = summary["finite_subcycling"]
        print(
            "subcycling max nsub:   "
            f"{subcycling['maximum_substeps_per_macro']} "
            f"(limit={subcycling['max_substeps_limit']})"
        )
        print(
            "retried macros/restarts: "
            f"{subcycling['retried_macrosteps']} / {subcycling['total_restarts']}"
        )
    print(f"samples / final time:   {summary['samples']} / {summary['final_time']:.6g}")
    print(f"Julia mass drift:       {summary['julia_mass_drift']:.3e}")
    print(f"Julia momentum drift:   {summary['julia_momentum_drift']:.3e}")
    print(f"Julia energy drift:     {summary['julia_energy_drift']:.3e}")
    print(f"final M400 relative:    {final_m400_relative:.3%}")
    print(
        "Gaussian-tail baseline: "
        f"{args.baseline_final_m400:.3%} "
        f"(improved={summary['m400_improved_over_gaussian_tail_baseline']})"
    )
    for quantity, value in summary["history_relative_l2"].items():
        print(f"history L2 {quantity:>14}: {value:.3%}")

    if not adaptive_reached:
        raise SystemExit(f"FAIL: adaptive {source_mode} closure did not reach final time")
    if summary["julia_mass_drift"] > 1.0e-12:
        raise SystemExit("FAIL: Julia mass conservation gate")
    if summary["julia_momentum_drift"] > 1.0e-12:
        raise SystemExit("FAIL: Julia momentum conservation gate")
    if summary["julia_energy_drift"] > 1.0e-10:
        raise SystemExit("FAIL: Julia energy conservation gate")
    if args.require_m400_improvement and not summary[
        "m400_improved_over_gaussian_tail_baseline"
    ]:
        raise SystemExit("FAIL: final M400 did not improve over Gaussian-tail baseline")
    history = summary["history_relative_l2"]
    if args.max_history_m200 is not None and history["M200"] > args.max_history_m200:
        raise SystemExit(
            "FAIL: M200 history error "
            f"{history['M200']:.3%} exceeds {args.max_history_m200:.3%}"
        )
    if (
        args.max_history_stress is not None
        and history["stress_norm"] > args.max_history_stress
    ):
        raise SystemExit(
            "FAIL: stress history error "
            f"{history['stress_norm']:.3%} exceeds {args.max_history_stress:.3%}"
        )
    print(f"PASS: adaptive {source_mode}-closure comparison completed.")


if __name__ == "__main__":
    main()

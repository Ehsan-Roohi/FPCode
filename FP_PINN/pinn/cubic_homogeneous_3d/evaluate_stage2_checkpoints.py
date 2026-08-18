#!/usr/bin/env python3
"""Independently validate every Stage-2 checkpoint and select a balanced best model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import shutil
import traceback
from typing import Any

import numpy as np
import tensorflow as tf

from heat_flux_g0 import (
    HEAT_FLUX_CONTINUATION_L2,
    HEAT_FLUX_PRIMARY_L2,
    HEAT_FLUX_PUBLICATION_L2,
)
from train_stage2 import Config, DensityModel, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-output", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--strict-gate", action="store_true",
        help="Return exit status 2 unless the selected checkpoint passes 5% G0",
    )
    return parser.parse_args()


def selection_score(metrics: dict[str, Any]) -> float:
    """Prefer analytic heat-flux accuracy while penalizing conservation drift."""
    heat = float(metrics["heat_flux_analytic_history_relative_l2"])
    marginal = float(metrics["marginal_relative_l2"])
    mass = float(metrics["max_mass_error"])
    momentum = float(metrics["max_momentum_norm"])
    energy = float(metrics["max_energy_error"])
    return (
        heat
        + 0.25 * marginal
        + 2.0 * max(0.0, mass - 0.01)
        + max(0.0, momentum - 0.01)
        + 2.0 * max(0.0, energy - 0.01)
    )


def smoke_admissible(metrics: dict[str, Any]) -> bool:
    checks = metrics["gate_checks"]
    return bool(all(
        passed for name, passed in checks.items() if name != "case_relaxation_error"
    ))


def load_config(case_output: Path, reference: Path) -> Config:
    raw = json.loads((case_output / "config.json").read_text())
    allowed = {item.name for item in fields(Config)}
    values = {key: value for key, value in raw.items() if key in allowed}
    values["output_dir"] = str(case_output)
    values["reference"] = str(reference)
    values["evaluate_only"] = True
    return Config(**values)


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "checkpoint", "smoke_admissible", "selection_score",
        "heat_flux_analytic_history_relative_l2",
        "particle_heat_flux_analytic_relative_l2",
        "heat_flux_active_axis_relative_l2", "heat_flux_history_relative_l2",
        "marginal_relative_l2", "max_mass_error", "max_momentum_norm",
        "max_energy_error", "max_transverse_heat_flux_relative",
        "continuation_pass", "primary_pass", "publication_pass",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    case_output = Path(args.case_output).resolve()
    reference = Path(args.reference).resolve()
    config = load_config(case_output, reference)
    if config.case != "heat_flux":
        raise SystemExit("Checkpoint sweep is currently defined for the heat_flux case")

    checkpoints = sorted((case_output / "checkpoints_h5").glob("epoch-*.weights.h5"))
    if not checkpoints:
        final = case_output / "stage2_final.weights.h5"
        if final.is_file():
            checkpoints = [final]
    if not checkpoints:
        raise SystemExit(f"No portable checkpoint found under {case_output}")

    sweep_root = case_output / "checkpoint_evaluation"
    sweep_root.mkdir(parents=True, exist_ok=True)
    model = DensityModel(config)
    model.log_density(tf.zeros((1, 1)), tf.zeros((1, 3)))
    rows: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        label = checkpoint.name.removesuffix(".weights.h5")
        destination = sweep_root / label
        destination.mkdir(parents=True, exist_ok=True)
        try:
            model.load_weights(checkpoint)
            metrics = evaluate(model, config, destination)
            row = {
                "checkpoint": checkpoint.relative_to(case_output).as_posix(),
                **metrics,
            }
            row["selection_score"] = selection_score(metrics)
            row["smoke_admissible"] = smoke_admissible(metrics)
            rows.append(row)
            (destination / "metrics.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n"
            )
            print(
                "CHECKPOINT_METRICS "
                + json.dumps(
                    {
                        "checkpoint": row["checkpoint"],
                        "qx_analytic_l2": row["heat_flux_analytic_history_relative_l2"],
                        "qx_particle_l2": row["heat_flux_active_axis_relative_l2"],
                        "energy": row["max_energy_error"],
                        "score": row["selection_score"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception:
            (destination / "evaluation_error.txt").write_text(traceback.format_exc())
            print(f"CHECKPOINT_FAILED {checkpoint}", flush=True)

    if not rows:
        raise SystemExit("Every checkpoint evaluation failed")
    write_table(case_output / "checkpoint_sweep.csv", rows)
    best = min(
        rows,
        key=lambda row: (
            not bool(row["smoke_admissible"]),
            float(row["selection_score"]),
        ),
    )
    best_source = case_output / str(best["checkpoint"])
    best_target = case_output / "stage2_best.weights.h5"
    shutil.copy2(best_source, best_target)
    summary = {
        "selection_rule": (
            "base-admissible first; then minimum analytic-Qx/marginal/conservation composite"
        ),
        "continuation_target_qx_l2": HEAT_FLUX_CONTINUATION_L2,
        "primary_target_qx_l2": HEAT_FLUX_PRIMARY_L2,
        "publication_target_qx_l2": HEAT_FLUX_PUBLICATION_L2,
        "continuation_target_passed": bool(best["continuation_pass"]),
        "primary_target_passed": bool(best["primary_pass"]),
        "publication_target_passed": bool(best["publication_pass"]),
        "best_checkpoint": best["checkpoint"],
        "best_weights": best_target.name,
        "best_metrics": best,
    }
    (case_output / "checkpoint_sweep.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("BEST_CHECKPOINT " + json.dumps(summary, sort_keys=True), flush=True)
    if args.strict_gate and not summary["continuation_target_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

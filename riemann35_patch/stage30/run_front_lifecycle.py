#!/usr/bin/env python3
"""Run the Stage-30 complete moving kinetic-front lifecycle qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from riemann35_patch.stage28.run_localized_pocket import _jsonable
from riemann35_patch.stage29.run_advecting_front import (
    M400,
    REPOSITORY_ROOT,
    configuration as stage29_configuration,
    run as run_stage29_case,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "workstation"), default="workstation"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    return parser.parse_args()


def configuration(mode: str, steps: int | None = None) -> dict[str, object]:
    default_steps = 36 if mode == "smoke" else 48
    values = stage29_configuration(
        mode, default_steps if steps is None else steps
    )
    values["sensor_interval_steps"] = 8
    values["release_sensor_interval_steps"] = 4
    values["causal_activation_candidates_only"] = True
    values["release_persistence_steps"] = 4
    values["plot_title"] = "Stage 30 complete front-following lifecycle"
    values["progress_event"] = "STAGE30_PROGRESS"
    return values


def _persistent_trailing_releases(
    history: list[dict[str, object]], persistence_steps: int
) -> list[dict[str, int]]:
    qualified: list[dict[str, int]] = []
    for index, event in enumerate(history):
        active_before = [int(cell) for cell in event["active_cells_before"]]
        leading_cell = max(active_before) if active_before else -1
        for released in event["release_cells"]:
            cell = int(released)
            if cell >= leading_cell:
                continue
            later = history[index + 1 : index + 1 + persistence_steps]
            if len(later) < persistence_steps:
                continue
            if all(cell not in item["active_cells_after"] for item in later):
                qualified.append(
                    {
                        "step": int(event["step"]),
                        "cell": cell,
                        "leading_cell_at_release": leading_cell,
                        "verified_inactive_steps": persistence_steps,
                    }
                )
    return qualified


def run(config: dict[str, object], output: Path) -> dict[str, object]:
    summary = run_stage29_case(config, output)
    history = summary["activation_history"]
    persistence = int(config["release_persistence_steps"])
    releases = [
        {"step": int(item["step"]), "cell": int(cell)}
        for item in history
        for cell in item["release_cells"]
    ]
    trailing = _persistent_trailing_releases(history, persistence)
    baseline_contracts = dict(summary["contracts"])
    complete_lifecycle_pass = bool(
        baseline_contracts["nontrivial_kinetic_front_birth_pass"] and trailing
    )
    workstation = config["mode"] == "workstation"
    baseline_pass = bool(
        baseline_contracts["all_qualification_gates_pass"]
        if workstation
        else baseline_contracts["numerical_accuracy_and_invariants_pass"]
        and baseline_contracts["kinetic_fraction_below_50_percent_pass"]
    )
    passed = baseline_pass and complete_lifecycle_pass
    decision = (
        "WORKSTATION_PASS" if workstation and passed
        else "WORKSTATION_HOLD" if workstation
        else "SMOKE_PASS" if passed
        else "SMOKE_HOLD"
    )

    summary["stage"] = "30"
    summary["case"] = "complete_advecting_kinetic_front_lifecycle"
    summary["decision"] = decision
    summary["configuration"] = {
        **summary["configuration"],
        "release_persistence_steps": persistence,
    }
    summary["primary_metrics"] = {
        **summary["primary_metrics"],
        "total_releases": len(releases),
        "persistent_trailing_releases": len(trailing),
    }
    summary["contracts"] = {
        **baseline_contracts,
        "persistent_trailing_release_pass": bool(trailing),
        "complete_front_following_lifecycle_pass": complete_lifecycle_pass,
        "smoke_gates_pass": bool(passed) if not workstation else None,
        "all_qualification_gates_pass": bool(passed) if workstation else False,
    }
    summary["release_history"] = releases
    summary["qualified_trailing_releases"] = trailing

    history_source = output / "stage29_advecting_front_histories.npz"
    history_destination = output / "stage30_front_lifecycle_histories.npz"
    if history_source.exists():
        with np.load(history_source) as archive:
            history_payload = {
                "x": archive["x"].astype(np.float32),
                "dt": archive["dt"].astype(np.float32),
                "refined_M400": archive["refined_moments"][:, :, M400].astype(
                    np.float32
                ),
                "coarse_M400": archive["coarse_moments"][:, :, M400].astype(
                    np.float32
                ),
                "adaptive_M400": archive["adaptive_moments"][:, :, M400].astype(
                    np.float32
                ),
                "refined_M420": archive["refined_M420"].astype(np.float32),
                "coarse_M420": archive["coarse_M420"].astype(np.float32),
                "adaptive_M420": archive["adaptive_M420"].astype(np.float32),
                "adaptive_active": archive["adaptive_active"],
                "activation_counts": archive["activation_counts"],
            }
        np.savez_compressed(history_destination, **history_payload)
        history_source.unlink()

    renames = {"stage29_advecting_front.png": "stage30_front_lifecycle.png"}
    for source, destination in renames.items():
        source_path = output / source
        if source_path.exists():
            source_path.replace(output / destination)
    old_summary = output / "stage29_advecting_front_summary.json"
    if old_summary.exists():
        old_summary.unlink()
    old_report = output / "STAGE29_ADVECTING_FRONT_RESULT.md"
    if old_report.exists():
        old_report.unlink()

    with (output / "stage30_front_lifecycle_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_jsonable(summary), stream, indent=2, allow_nan=False)

    metrics = summary["primary_metrics"]
    errors = metrics["final_errors"]
    space_time = metrics["space_time_errors"]
    contracts = summary["contracts"]
    report = [
        "# Stage 30 complete front-following lifecycle result",
        "",
        f"- Decision: **{decision}**",
        f"- Causal births / persistent trailing releases: {metrics['causal_births']} / {len(trailing)}",
        f"- Total release events: {len(releases)}",
        f"- Mean/peak/final kinetic fraction: {metrics['mean_active_fraction_percent']:.3f}% / {metrics['peak_active_fraction_percent']:.3f}% / {metrics['final_active_fraction_percent']:.3f}%",
        f"- Final adaptive M400 error vs refined DVM: {errors['adaptive_vs_refined_M400_percent']:.6f}%",
        f"- Final adaptive M420 error vs refined DVM: {errors['adaptive_vs_refined_M420_percent']:.6f}%",
        f"- Space-time adaptive M400/M420 errors: {space_time['adaptive_vs_refined_M400_percent']:.6f}% / {space_time['adaptive_vs_refined_M420_percent']:.6f}%",
        f"- Expensive sensors / no-donor skips: {metrics['expensive_sensor_evaluations']} / {metrics['activation_sensor_skips_no_donor']}",
        f"- Adaptive/coarse-DVM wall-time ratio: {metrics['adaptive_over_coarse_dvm_wall_time']:.3f}x",
        f"- Measured speedup: {metrics['measured_speedup_factor']:.3f}x",
        f"- Maximum finite-volume balance residual: {contracts['maximum_balance_residual']:.3e}",
        f"- Maximum micro/macro sync residual: {contracts['maximum_micro_macro_sync_residual']:.3e}",
        "",
        "The moving pocket both created positive causal kinetic memory ahead",
        "and retired kinetic memory behind it without changing the frozen",
        "Stage-25 thresholds.  Independent MD/DSMC validation remains necessary",
        "before making a physical-fidelity claim.",
    ]
    (output / "STAGE30_FRONT_LIFECYCLE_RESULT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = arguments()
    config = configuration(args.mode, args.steps)
    output = args.output
    if output is None:
        output = REPOSITORY_ROOT / "results" / "riemann35_stage30" / args.mode
    summary = run(config, output)
    print(
        json.dumps(
            _jsonable(
                {
                    "decision": summary["decision"],
                    "primary_metrics": summary["primary_metrics"],
                    "contracts": summary["contracts"],
                    "qualified_trailing_releases": summary[
                        "qualified_trailing_releases"
                    ],
                    "timing_seconds": summary["timing_seconds"],
                    "output": str(output),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

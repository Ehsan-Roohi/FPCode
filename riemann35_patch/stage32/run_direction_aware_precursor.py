#!/usr/bin/env python3
"""Develop and audit the Stage-32 direction-aware causal precursor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyqmom_fp import stage25_hysteresis
from riemann35_patch.stage28.run_localized_pocket import _jsonable
from riemann35_patch.stage31.run_heldout_shock import (
    configuration as stage31_configuration,
)
from riemann35_patch.stage31.run_heldout_shock import run as run_shock_gate


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FRONT_OBSERVABLES = ("mass", "stress_xx", "heat_flux_x", "M420")
BLIND_VALIDATION_MACH = 2.5


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "workstation"), default="workstation"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    return parser.parse_args()


def configuration(mode: str, steps: int | None = None) -> dict[str, object]:
    """Return Stage 31's frozen lifecycle plus the new causal sensor view."""

    values = stage31_configuration(mode, steps)
    values.update(
        {
            "stage": "32",
            "case": "direction_aware_precursor_ma2_development",
            "result_stem": "stage32_direction_aware_precursor",
            "report_filename": "STAGE32_DIRECTION_AWARE_PRECURSOR_RESULT.md",
            "report_title": "Stage 32 direction-aware Mach-2 precursor result",
            "figure_title": (
                "Mach-2 development shock — direction-aware causal precursor"
            ),
            "progress_event": "STAGE32_PROGRESS",
            "workstation_pass_label": "DEVELOPMENT_PASS",
            "workstation_hold_label": "DEVELOPMENT_HOLD",
            "smoke_pass_label": "DEVELOPMENT_SMOKE_PASS",
            "smoke_hold_label": "DEVELOPMENT_SMOKE_HOLD",
            "case_contract_name": "mach2_development_not_blind_qualification_pass",
            "case_contract_pass": True,
            "case_summary": (
                "Mach 2 development case; Stage 31 remains WORKSTATION_HOLD"
            ),
            "scope": (
                "Mach-2 direction-aware precursor development; not a blind "
                "cross-case qualification and not MD/DSMC evidence"
            ),
            # The original normalized half-range mass discrepancy is retained.
            # Stress, heat-flux, and M420 merely provide additional causal views
            # of the same already-active positive donor and fixed t=0 carrier.
            "kinetic_front_observables": FRONT_OBSERVABLES,
            # Reuse the already-frozen minimum kinetic dwell as the causal
            # prediction horizon; no new fitted horizon or threshold is added.
            "directional_front_lookahead_steps": (
                stage25_hysteresis().minimum_active_steps
            ),
            "minimum_left_neighbor_front_births": 1,
            "minimum_weighted_only_front_births": 1,
            "blind_validation_mach": BLIND_VALIDATION_MACH,
            "blind_validation_executed": False,
            "stage31_decision_preserved": "WORKSTATION_HOLD",
        }
    )
    if float(values["kinetic_front_on"]) != stage25_hysteresis().tail_on:
        raise RuntimeError("Stage 32 must keep the frozen 0.40 front threshold")
    if float(values["mach"]) != 2.0:
        raise RuntimeError("Stage 32 development must remain on Mach 2")
    return values


def run(config: dict[str, object], output: Path) -> dict[str, object]:
    """Run the development case without touching the reserved blind Mach."""

    if bool(config.get("blind_validation_executed", False)):
        raise ValueError("Stage 32 must not execute the reserved blind case")
    return run_shock_gate(config, output)


def main() -> None:
    args = arguments()
    config = configuration(args.mode, args.steps)
    output = args.output
    if output is None:
        output = REPOSITORY_ROOT / "results" / "riemann35_stage32" / args.mode
    summary = run(config, output)
    print(
        json.dumps(
            _jsonable(
                {
                    "decision": summary["decision"],
                    "primary_metrics": summary["primary_metrics"],
                    "contracts": summary["contracts"],
                    "timing_seconds": summary["timing_seconds"],
                    "output": str(output),
                    "reserved_blind_validation_mach": BLIND_VALIDATION_MACH,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

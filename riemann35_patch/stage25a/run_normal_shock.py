#!/usr/bin/env python3
"""Run the Stage-25A Mach-3 one-dimensional normal-shock gate.

The default is a small development smoke run.  It verifies that the spatial
contracts execute but is explicitly not publication evidence.  The frozen
qualification dimensions are available through ``--mode qualification`` and
are intended for a batch/HPC allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    AdaptiveSpatialState,
    DVMGrid,
    SpatialDVMState,
    SpatialGrid1D,
    adaptive_shock_step,
    full_dvm_shock_step,
    initialize_adaptive_normal_shock,
    initialize_normal_shock_dvm,
    initialize_normal_shock_moments,
    macro_shock_step,
    normal_shock_rankine_hugoniot,
    normalized_profile_error,
    shock_profiles,
)
from hyqmom_fp.moments import HYQMOM_35_INDICES  # noqa: E402


FOURTH_POSITIONS = np.asarray(
    [sum(index) == 4 for index in HYQMOM_35_INDICES], dtype=bool
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "qualification"), default="smoke"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="write a transactional restart checkpoint every N completed steps",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="resume from a Stage-25A restart or failure checkpoint",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="emit a flushed JSON progress record every N completed steps",
    )
    return parser.parse_args()


def configuration(mode: str, steps_override: int | None) -> dict[str, object]:
    if mode == "smoke":
        config: dict[str, object] = {
            "scope": "development smoke; not publication evidence",
            "x_lower": -4.0,
            "x_upper": 4.0,
            "spatial_cells": 8,
            "v_lower": (-12.0, -10.0, -10.0),
            "v_upper": (14.0, 10.0, 10.0),
            "velocity_shape": (17, 13, 13),
            "cfl": 0.06,
            "steps": 3,
            "initial_active_half_width": 1,
        }
    else:
        config = {
            "scope": "frozen Stage-25A qualification",
            "x_lower": -20.0,
            "x_upper": 20.0,
            "spatial_cells": 160,
            "v_lower": (-12.0, -10.0, -10.0),
            "v_upper": (14.0, 10.0, 10.0),
            "velocity_shape": (61, 33, 33),
            "cfl": 0.35,
            "steps": 2400,
            "initial_active_half_width": 2,
        }
    if steps_override is not None:
        if steps_override < 1:
            raise ValueError("steps override must be positive")
        config["steps"] = steps_override
        config["scope"] = f"{config['scope']}; step count overridden for diagnostics"
    return config


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _fourth_order_error(model: np.ndarray, reference: np.ndarray) -> float:
    return normalized_profile_error(
        model[:, FOURTH_POSITIONS], reference[:, FOURTH_POSITIONS]
    )


def _transport_summary(diagnostics) -> dict[str, float]:
    return {
        "cfl": diagnostics.cfl,
        "minimum_mass": diagnostics.minimum_mass,
        "mass_balance_residual": diagnostics.mass_balance_residual,
        "momentum_balance_residual": diagnostics.momentum_balance_residual,
        "energy_balance_residual": diagnostics.energy_balance_residual,
    }


def _write_checkpoint(
    path: Path,
    *,
    config: dict[str, object],
    dt: float,
    completed_steps: int,
    reference: SpatialDVMState,
    macro: np.ndarray,
    adaptive: AdaptiveSpatialState,
    reference_history: list[np.ndarray],
    macro_history: list[np.ndarray],
    adaptive_history: list[np.ndarray],
    active_history: list[np.ndarray],
    reference_diagnostics: list[dict[str, object]],
    macro_diagnostics: list[dict[str, object]],
    adaptive_diagnostics: list[dict[str, object]],
    reference_seconds: float,
    macro_seconds: float,
    adaptive_seconds: float,
    failure_phase: str | None = None,
    failure_message: str | None = None,
) -> None:
    """Atomically preserve a consistent pre-step or completed-step state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "stage25a-restart-v1",
        "configuration": _jsonable(config),
        "dt": dt,
        "completed_steps": completed_steps,
        "reference_diagnostics": reference_diagnostics,
        "macro_diagnostics": macro_diagnostics,
        "adaptive_diagnostics": adaptive_diagnostics,
        "timing_seconds": {
            "full_dvm": reference_seconds,
            "macro": macro_seconds,
            "adaptive": adaptive_seconds,
        },
        "failure_phase": failure_phase,
        "failure_message": failure_message,
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            metadata=np.asarray(json.dumps(_jsonable(metadata))),
            reference_masses=reference.masses,
            macro=macro,
            adaptive_moments=adaptive.moments,
            adaptive_micro_masses=adaptive.micro_masses,
            adaptive_active=adaptive.active,
            adaptive_active_steps=adaptive.active_steps,
            adaptive_release_counter=adaptive.release_counter,
            adaptive_global_step=np.asarray(adaptive.global_step),
            adaptive_transition_count=np.asarray(adaptive.transition_count),
            adaptive_blocked_births=np.asarray(adaptive.blocked_births),
            reference_history=np.asarray(reference_history),
            macro_history=np.asarray(macro_history),
            adaptive_history=np.asarray(adaptive_history),
            active_history=np.asarray(active_history),
        )
    os.replace(temporary, path)


def _restore_checkpoint(
    path: Path,
    *,
    config: dict[str, object],
    dt: float,
    xgrid: SpatialGrid1D,
    vgrid: DVMGrid,
):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        if metadata.get("format") != "stage25a-restart-v1":
            raise ValueError("unsupported Stage-25A checkpoint format")
        if metadata.get("configuration") != _jsonable(config):
            raise ValueError("checkpoint configuration does not match this run")
        if not np.isclose(float(metadata.get("dt")), dt, rtol=0.0, atol=1.0e-15):
            raise ValueError("checkpoint time step does not match this run")
        reference = SpatialDVMState(
            xgrid, vgrid, saved["reference_masses"].copy()
        )
        macro = saved["macro"].copy()
        adaptive = AdaptiveSpatialState(
            spatial_grid=xgrid,
            velocity_grid=vgrid,
            moments=saved["adaptive_moments"].copy(),
            micro_masses=saved["adaptive_micro_masses"].copy(),
            active=saved["adaptive_active"].copy(),
            active_steps=saved["adaptive_active_steps"].copy(),
            release_counter=saved["adaptive_release_counter"].copy(),
            global_step=int(saved["adaptive_global_step"]),
            transition_count=int(saved["adaptive_transition_count"]),
            blocked_births=int(saved["adaptive_blocked_births"]),
        )
        histories = tuple(
            [item.copy() for item in saved[name]]
            for name in (
                "reference_history",
                "macro_history",
                "adaptive_history",
                "active_history",
            )
        )
    timings = metadata["timing_seconds"]
    return (
        int(metadata["completed_steps"]),
        reference,
        macro,
        adaptive,
        *histories,
        list(metadata["reference_diagnostics"]),
        list(metadata["macro_diagnostics"]),
        list(metadata["adaptive_diagnostics"]),
        float(timings["full_dvm"]),
        float(timings["macro"]),
        float(timings["adaptive"]),
    )


def run(
    config: dict[str, object],
    output: Path,
    *,
    checkpoint_every: int = 0,
    resume_checkpoint: Path | None = None,
    progress_every: int = 1,
) -> dict[str, object]:
    if checkpoint_every < 0 or progress_every < 1:
        raise ValueError("checkpoint_every must be nonnegative and progress_every positive")
    output.mkdir(parents=True, exist_ok=True)
    shock = normal_shock_rankine_hugoniot(3.0)
    xgrid = SpatialGrid1D(
        float(config["x_lower"]),
        float(config["x_upper"]),
        int(config["spatial_cells"]),
    )
    vgrid = DVMGrid(
        tuple(config["v_lower"]),
        tuple(config["v_upper"]),
        tuple(config["velocity_shape"]),
    )
    maximum_speed = float(np.max(np.abs(vgrid.centers()[:, 0])))
    dt = float(config["cfl"]) * xgrid.width / maximum_speed
    steps = int(config["steps"])
    tau = 1.0

    reference, left_inflow, right_inflow = initialize_normal_shock_dvm(
        xgrid, vgrid, shock
    )
    macro = initialize_normal_shock_moments(xgrid, shock)
    adaptive, adaptive_left, adaptive_right = initialize_adaptive_normal_shock(
        xgrid,
        vgrid,
        shock,
        initial_active_half_width=int(config["initial_active_half_width"]),
    )

    if resume_checkpoint is None:
        start_step = 0
        reference_history = [reference.moments()]
        macro_history = [macro.copy()]
        adaptive_history = [adaptive.moments.copy()]
        active_history = [adaptive.active.copy()]
        reference_diagnostics = []
        macro_diagnostics = []
        adaptive_diagnostics = []
        reference_seconds = 0.0
        macro_seconds = 0.0
        adaptive_seconds = 0.0
    else:
        (
            start_step,
            reference,
            macro,
            adaptive,
            reference_history,
            macro_history,
            adaptive_history,
            active_history,
            reference_diagnostics,
            macro_diagnostics,
            adaptive_diagnostics,
            reference_seconds,
            macro_seconds,
            adaptive_seconds,
        ) = _restore_checkpoint(
            resume_checkpoint,
            config=config,
            dt=dt,
            xgrid=xgrid,
            vgrid=vgrid,
        )
        if start_step >= steps:
            raise ValueError("checkpoint has already completed all requested steps")
        print(
            json.dumps(
                {
                    "event": "STAGE25A_RESUME",
                    "completed_steps": start_step,
                    "remaining_steps": steps - start_step,
                    "checkpoint": str(resume_checkpoint),
                }
            ),
            flush=True,
        )

    for step in range(start_step, steps):
        phase_name = "macro"
        try:
            phase = perf_counter()
            next_macro, macro_diag = macro_shock_step(
                macro,
                xgrid,
                dt,
                tau,
                shock.upstream_moments,
                shock.downstream_moments,
            )
            macro_step_seconds = perf_counter() - phase

            phase_name = "adaptive"
            phase = perf_counter()
            next_adaptive, adaptive_diag = adaptive_shock_step(
                adaptive, dt, tau, adaptive_left, adaptive_right
            )
            adaptive_step_seconds = perf_counter() - phase

            phase_name = "full_dvm"
            phase = perf_counter()
            next_reference, reference_diag = full_dvm_shock_step(
                reference, dt, tau, left_inflow, right_inflow
            )
            reference_step_seconds = perf_counter() - phase
        except Exception as exc:
            failure_path = output / "stage25a_failure_checkpoint.npz"
            _write_checkpoint(
                failure_path,
                config=config,
                dt=dt,
                completed_steps=step,
                reference=reference,
                macro=macro,
                adaptive=adaptive,
                reference_history=reference_history,
                macro_history=macro_history,
                adaptive_history=adaptive_history,
                active_history=active_history,
                reference_diagnostics=reference_diagnostics,
                macro_diagnostics=macro_diagnostics,
                adaptive_diagnostics=adaptive_diagnostics,
                reference_seconds=reference_seconds,
                macro_seconds=macro_seconds,
                adaptive_seconds=adaptive_seconds,
                failure_phase=phase_name,
                failure_message=f"{type(exc).__name__}: {exc}",
            )
            print(
                json.dumps(
                    {
                        "event": "STAGE25A_FAILURE_CHECKPOINT",
                        "failed_step": step + 1,
                        "phase": phase_name,
                        "checkpoint": str(failure_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            raise

        macro = next_macro
        adaptive = next_adaptive
        reference = next_reference
        macro_seconds += macro_step_seconds
        adaptive_seconds += adaptive_step_seconds
        reference_seconds += reference_step_seconds

        reference_history.append(reference.moments())
        macro_history.append(macro.copy())
        adaptive_history.append(adaptive.moments.copy())
        active_history.append(adaptive.active.copy())
        reference_diagnostics.append(
            {
                "step": step + 1,
                "transport": _transport_summary(reference_diag.transport),
                "minimum_mass": reference_diag.minimum_mass,
                "maximum_projection_residual": reference_diag.maximum_projection_residual,
                "maximum_collision_invariant_drift": (
                    reference_diag.maximum_collision_invariant_drift
                ),
            }
        )
        macro_diagnostics.append(
            {"step": step + 1, "transport": _transport_summary(macro_diag)}
        )
        adaptive_diagnostics.append(
            {
                "step": step + 1,
                **{
                    key: value
                    for key, value in asdict(adaptive_diag).items()
                    if key != "transport"
                },
                "transport": _transport_summary(adaptive_diag.transport),
            }
        )

        completed_steps = step + 1
        if completed_steps % progress_every == 0 or completed_steps == steps:
            print(
                json.dumps(
                    {
                        "event": "STAGE25A_PROGRESS",
                        "step": completed_steps,
                        "steps": steps,
                        "active_fraction": float(np.mean(adaptive.active)),
                        "step_seconds": {
                            "macro": macro_step_seconds,
                            "adaptive": adaptive_step_seconds,
                            "full_dvm": reference_step_seconds,
                        },
                    }
                ),
                flush=True,
            )
        if checkpoint_every and completed_steps % checkpoint_every == 0:
            checkpoint_path = output / "stage25a_restart_checkpoint.npz"
            _write_checkpoint(
                checkpoint_path,
                config=config,
                dt=dt,
                completed_steps=completed_steps,
                reference=reference,
                macro=macro,
                adaptive=adaptive,
                reference_history=reference_history,
                macro_history=macro_history,
                adaptive_history=adaptive_history,
                active_history=active_history,
                reference_diagnostics=reference_diagnostics,
                macro_diagnostics=macro_diagnostics,
                adaptive_diagnostics=adaptive_diagnostics,
                reference_seconds=reference_seconds,
                macro_seconds=macro_seconds,
                adaptive_seconds=adaptive_seconds,
            )
            print(
                json.dumps(
                    {
                        "event": "STAGE25A_CHECKPOINT",
                        "completed_steps": completed_steps,
                        "checkpoint": str(checkpoint_path),
                    }
                ),
                flush=True,
            )

    # The three methods execute serially.  Summing their accumulated timings
    # preserves the original wall-time meaning across checkpoint/resume.
    elapsed = reference_seconds + macro_seconds + adaptive_seconds
    reference_history_array = np.asarray(reference_history)
    macro_history_array = np.asarray(macro_history)
    adaptive_history_array = np.asarray(adaptive_history)
    active_history_array = np.asarray(active_history)

    reference_profiles = shock_profiles(reference_history_array[-1])
    macro_profiles = shock_profiles(macro_history_array[-1])
    adaptive_profiles = shock_profiles(adaptive_history_array[-1])
    profile_errors = {
        name: {
            "macro_vs_dvm_percent": 100.0
            * normalized_profile_error(macro_profiles[name], reference_profiles[name]),
            "adaptive_vs_dvm_percent": 100.0
            * normalized_profile_error(adaptive_profiles[name], reference_profiles[name]),
        }
        for name in reference_profiles
    }
    e4_macro = 100.0 * _fourth_order_error(
        macro_history_array[-1], reference_history_array[-1]
    )
    e4_adaptive = 100.0 * _fourth_order_error(
        adaptive_history_array[-1], reference_history_array[-1]
    )
    maximum_reference_projection = max(
        item["maximum_projection_residual"] for item in reference_diagnostics
    )
    maximum_reference_balance = max(
        max(
            item["transport"]["mass_balance_residual"],
            item["transport"]["momentum_balance_residual"],
            item["transport"]["energy_balance_residual"],
        )
        for item in reference_diagnostics
    )
    maximum_adaptive_balance = max(
        max(
            item["transport"]["mass_balance_residual"],
            item["transport"]["momentum_balance_residual"],
            item["transport"]["energy_balance_residual"],
        )
        for item in adaptive_diagnostics
    )
    maximum_sync_residual = max(
        item["maximum_micro_macro_residual"] for item in adaptive_diagnostics
    )
    blocked_births = int(sum(item["blocked_births"] for item in adaptive_diagnostics))
    development_contracts_pass = bool(
        maximum_reference_projection < 2.0e-9
        and maximum_reference_balance < 2.0e-9
        and maximum_adaptive_balance < 2.0e-8
        and maximum_sync_residual < 2.0e-8
        and blocked_births == 0
        and np.min(reference.masses) > 0.0
        and (
            np.min(adaptive.micro_masses[adaptive.active]) > 0.0
            if np.any(adaptive.active)
            else True
        )
    )
    decision = (
        "SMOKE_PASS_NOT_EVIDENCE" if development_contracts_pass else "SMOKE_FAIL"
    )
    if config["scope"] == "frozen Stage-25A qualification":
        decision = (
            "QUALIFICATION_PASS"
            if development_contracts_pass
            and e4_adaptive < 3.0
            and float(np.mean(active_history_array)) < 0.50
            and reference_seconds / max(adaptive_seconds, 1.0e-30) >= 2.0
            else "QUALIFICATION_HOLD"
        )

    np.savez_compressed(
        output / "stage25a_normal_shock_histories.npz",
        x=xgrid.centers,
        dt=np.asarray(dt),
        reference_moments=reference_history_array,
        macro_moments=macro_history_array,
        adaptive_moments=adaptive_history_array,
        adaptive_active=active_history_array,
    )
    summary = {
        "stage": "25A",
        "case": "normal_shock_ma3",
        "decision": decision,
        "scope": config["scope"],
        "configuration": {
            **config,
            "dt": dt,
            "tau": tau,
            "final_time": steps * dt,
            "velocity_cells": vgrid.size,
        },
        "rankine_hugoniot": asdict(shock),
        "primary_metrics": {
            "E4_macro_vs_dvm_percent": e4_macro,
            "E4_adaptive_vs_dvm_percent": e4_adaptive,
            "mean_active_fraction": float(np.mean(active_history_array)),
            "final_active_fraction": float(np.mean(adaptive.active)),
            "full_dvm_over_adaptive_wall_time": reference_seconds
            / max(adaptive_seconds, 1.0e-30),
        },
        "profile_errors_percent": profile_errors,
        "contracts": {
            "development_contracts_pass": development_contracts_pass,
            "maximum_reference_projection_residual": maximum_reference_projection,
            "maximum_reference_balance_residual": maximum_reference_balance,
            "maximum_adaptive_balance_residual": maximum_adaptive_balance,
            "maximum_micro_macro_residual": maximum_sync_residual,
            "blocked_causal_births": blocked_births,
            "reference_minimum_mass": float(np.min(reference.masses)),
            "adaptive_minimum_active_mass": (
                float(np.min(adaptive.micro_masses[adaptive.active]))
                if np.any(adaptive.active)
                else 0.0
            ),
        },
        "timing_seconds": {
            "full_dvm": reference_seconds,
            "macro": macro_seconds,
            "adaptive": adaptive_seconds,
            "total": elapsed,
        },
        "reference_step_diagnostics": reference_diagnostics,
        "macro_step_diagnostics": macro_diagnostics,
        "adaptive_step_diagnostics": adaptive_diagnostics,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    with (output / "stage25a_normal_shock_summary.json").open("w") as stream:
        json.dump(_jsonable(summary), stream, indent=2)
    report = [
        "# Stage 25A normal-shock run",
        "",
        f"- Decision: **{decision}**",
        f"- Scope: {config['scope']}",
        f"- Mach: {shock.mach:g}",
        f"- Spatial cells: {xgrid.cells}",
        f"- Velocity cells: {vgrid.size}",
        f"- Steps / final time: {steps} / {steps * dt:.6g} tau",
        f"- E4 macro vs DVM: {e4_macro:.6f}%",
        f"- E4 adaptive vs DVM: {e4_adaptive:.6f}%",
        f"- Mean active fraction: {100*np.mean(active_history_array):.3f}%",
        f"- DVM/adaptive wall-time ratio: {reference_seconds/max(adaptive_seconds,1e-30):.3f}x",
        f"- Maximum reference balance residual: {maximum_reference_balance:.3e}",
        f"- Maximum adaptive balance residual: {maximum_adaptive_balance:.3e}",
        f"- Maximum micro/macro sync residual: {maximum_sync_residual:.3e}",
        f"- Blocked causal births: {blocked_births}",
        "",
        "Smoke values are development diagnostics and must not be quoted as the",
        "frozen qualification result.",
    ]
    (output / "STAGE25A_NORMAL_SHOCK_RUN.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = arguments()
    config = configuration(args.mode, args.steps)
    output = args.output
    if output is None:
        output = REPOSITORY_ROOT / "results/riemann35_stage25a" / args.mode
    checkpoint_every = args.checkpoint_every
    if checkpoint_every is None:
        checkpoint_every = 25 if args.mode == "qualification" else 0
    progress_every = args.progress_every
    if progress_every is None:
        progress_every = 1 if args.mode == "smoke" else 5
    summary = run(
        config,
        output,
        checkpoint_every=checkpoint_every,
        resume_checkpoint=args.resume_checkpoint,
        progress_every=progress_every,
    )
    print(json.dumps(_jsonable({
        "decision": summary["decision"],
        "primary_metrics": summary["primary_metrics"],
        "contracts": summary["contracts"],
        "output": str(output),
    }), indent=2))


if __name__ == "__main__":
    main()

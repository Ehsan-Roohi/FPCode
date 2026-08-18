#!/usr/bin/env python3
"""Fast, parallelizable Stage-25A Mach-3 diagnostic.

Each invocation advances exactly one of the macro, adaptive, or full-DVM
models.  A separate ``collect`` action compares their final fields.  The lite
case keeps the qualification velocity grid, closure, sensor thresholds, Mach
number, domain, and CFL while halving the spatial resolution and shortening
the integration.  It is a decision gate for optimization, not publication
evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import zipfile
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


METHODS = ("macro", "adaptive", "full_dvm")
FOURTH_POSITIONS = np.asarray(
    [sum(index) == 4 for index in HYQMOM_35_INDICES], dtype=bool
)


def lite_configuration(steps: int = 600) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be positive")
    return {
        "scope": "Stage-25A qualification-lite; optimization decision only",
        "mach": 3.0,
        "x_lower": -20.0,
        "x_upper": 20.0,
        "spatial_cells": 80,
        "v_lower": (-12.0, -10.0, -10.0),
        "v_upper": (14.0, 10.0, 10.0),
        "velocity_shape": (61, 33, 33),
        "cfl": 0.35,
        "steps": steps,
        "initial_active_half_width": 2,
    }


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


def _balance_maximum(diagnostics) -> float:
    return float(
        max(
            diagnostics.mass_balance_residual,
            diagnostics.momentum_balance_residual,
            diagnostics.energy_balance_residual,
        )
    )


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _checkpoint_payload(method: str, state) -> dict[str, np.ndarray]:
    if method == "macro":
        return {"moments": state}
    if method == "full_dvm":
        return {"masses": state.masses}
    return {
        "moments": state.moments,
        "micro_masses": state.micro_masses,
        "active": state.active,
        "active_steps": state.active_steps,
        "release_counter": state.release_counter,
        "global_step": np.asarray(state.global_step),
        "transition_count": np.asarray(state.transition_count),
        "blocked_births_total": np.asarray(state.blocked_births),
    }


def _write_checkpoint(
    path: Path,
    *,
    method: str,
    config: dict[str, object],
    dt: float,
    completed_steps: int,
    state,
    metrics: dict[str, float | int],
    failure: str | None = None,
) -> None:
    metadata = {
        "format": "stage25a-lite-restart-v1",
        "method": method,
        "configuration": _jsonable(config),
        "dt": dt,
        "completed_steps": completed_steps,
        "metrics": _jsonable(metrics),
        "failure": failure,
    }
    _atomic_savez(
        path,
        metadata=np.asarray(json.dumps(metadata)),
        **_checkpoint_payload(method, state),
    )


def _restore_checkpoint(
    path: Path,
    *,
    method: str,
    config: dict[str, object],
    dt: float,
    xgrid: SpatialGrid1D,
    vgrid: DVMGrid,
):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        if metadata.get("format") != "stage25a-lite-restart-v1":
            raise ValueError("unsupported Stage-25A-lite checkpoint format")
        if metadata.get("method") != method:
            raise ValueError("checkpoint method does not match this task")
        if metadata.get("configuration") != _jsonable(config):
            raise ValueError("checkpoint configuration does not match this run")
        if not np.isclose(float(metadata["dt"]), dt, rtol=0.0, atol=1.0e-15):
            raise ValueError("checkpoint time step does not match this run")
        if method == "macro":
            state = saved["moments"].copy()
        elif method == "full_dvm":
            state = SpatialDVMState(xgrid, vgrid, saved["masses"].copy())
        else:
            state = AdaptiveSpatialState(
                spatial_grid=xgrid,
                velocity_grid=vgrid,
                moments=saved["moments"].copy(),
                micro_masses=saved["micro_masses"].copy(),
                active=saved["active"].copy(),
                active_steps=saved["active_steps"].copy(),
                release_counter=saved["release_counter"].copy(),
                global_step=int(saved["global_step"]),
                transition_count=int(saved["transition_count"]),
                blocked_births=int(saved["blocked_births_total"]),
            )
    return int(metadata["completed_steps"]), state, dict(metadata["metrics"])


def _initial_metrics() -> dict[str, float | int]:
    return {
        "wall_seconds": 0.0,
        "maximum_balance_residual": 0.0,
        "maximum_projection_residual": 0.0,
        "maximum_invariant_drift": 0.0,
        "maximum_micro_macro_residual": 0.0,
        "blocked_births": 0,
        "activations": 0,
        "releases": 0,
        "active_fraction_sum": 0.0,
        "minimum_mass": float("inf"),
    }


def run_method(
    method: str,
    output: Path,
    *,
    steps: int = 600,
    checkpoint_every: int = 50,
    progress_every: int = 10,
    resume_checkpoint: Path | None = None,
) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if checkpoint_every < 0 or progress_every < 1:
        raise ValueError("checkpoint interval must be nonnegative; progress positive")

    output.mkdir(parents=True, exist_ok=True)
    config = lite_configuration(steps)
    shock = normal_shock_rankine_hugoniot(float(config["mach"]))
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
    tau = 1.0

    reference, left_dvm, right_dvm = initialize_normal_shock_dvm(
        xgrid, vgrid, shock
    )
    macro = initialize_normal_shock_moments(xgrid, shock)
    adaptive, adaptive_left, adaptive_right = initialize_adaptive_normal_shock(
        xgrid,
        vgrid,
        shock,
        initial_active_half_width=int(config["initial_active_half_width"]),
    )
    state = {"macro": macro, "adaptive": adaptive, "full_dvm": reference}[method]
    start_step = 0
    metrics = _initial_metrics()

    if resume_checkpoint is not None:
        start_step, state, metrics = _restore_checkpoint(
            resume_checkpoint,
            method=method,
            config=config,
            dt=dt,
            xgrid=xgrid,
            vgrid=vgrid,
        )
        if start_step >= steps:
            raise ValueError("checkpoint already completed this run")
        print(
            json.dumps(
                {
                    "event": "STAGE25A_LITE_RESUME",
                    "method": method,
                    "completed_steps": start_step,
                    "remaining_steps": steps - start_step,
                }
            ),
            flush=True,
        )

    checkpoint_path = output / f"{method}_restart_checkpoint.npz"
    for step in range(start_step, steps):
        started = perf_counter()
        try:
            if method == "macro":
                state, diagnostics = macro_shock_step(
                    state,
                    xgrid,
                    dt,
                    tau,
                    shock.upstream_moments,
                    shock.downstream_moments,
                )
                balance = diagnostics
                minimum_mass = diagnostics.minimum_mass
            elif method == "full_dvm":
                state, diagnostics = full_dvm_shock_step(
                    state, dt, tau, left_dvm, right_dvm
                )
                balance = diagnostics.transport
                minimum_mass = diagnostics.minimum_mass
                metrics["maximum_projection_residual"] = max(
                    float(metrics["maximum_projection_residual"]),
                    diagnostics.maximum_projection_residual,
                )
                metrics["maximum_invariant_drift"] = max(
                    float(metrics["maximum_invariant_drift"]),
                    diagnostics.maximum_collision_invariant_drift,
                )
            else:
                state, diagnostics = adaptive_shock_step(
                    state, dt, tau, adaptive_left, adaptive_right
                )
                balance = diagnostics.transport
                minimum_mass = diagnostics.minimum_micro_mass
                metrics["maximum_micro_macro_residual"] = max(
                    float(metrics["maximum_micro_macro_residual"]),
                    diagnostics.maximum_micro_macro_residual,
                )
                metrics["blocked_births"] += diagnostics.blocked_births
                metrics["activations"] += diagnostics.activations
                metrics["releases"] += diagnostics.releases
                metrics["active_fraction_sum"] += diagnostics.active_fraction
        except Exception as exc:
            _write_checkpoint(
                output / f"{method}_failure_checkpoint.npz",
                method=method,
                config=config,
                dt=dt,
                completed_steps=step,
                state=state,
                metrics=metrics,
                failure=f"{type(exc).__name__}: {exc}",
            )
            raise

        step_seconds = perf_counter() - started
        metrics["wall_seconds"] += step_seconds
        metrics["maximum_balance_residual"] = max(
            float(metrics["maximum_balance_residual"]),
            _balance_maximum(balance),
        )
        if minimum_mass > 0.0:
            metrics["minimum_mass"] = min(
                float(metrics["minimum_mass"]), float(minimum_mass)
            )
        completed_steps = step + 1

        if completed_steps % progress_every == 0 or completed_steps == steps:
            record = {
                "event": "STAGE25A_LITE_PROGRESS",
                "method": method,
                "step": completed_steps,
                "steps": steps,
                "step_seconds": step_seconds,
            }
            if method == "adaptive":
                record["active_fraction"] = float(np.mean(state.active))
            print(json.dumps(record), flush=True)

        if checkpoint_every and completed_steps % checkpoint_every == 0:
            _write_checkpoint(
                checkpoint_path,
                method=method,
                config=config,
                dt=dt,
                completed_steps=completed_steps,
                state=state,
                metrics=metrics,
            )
            print(
                json.dumps(
                    {
                        "event": "STAGE25A_LITE_CHECKPOINT",
                        "method": method,
                        "completed_steps": completed_steps,
                        "checkpoint": str(checkpoint_path),
                    }
                ),
                flush=True,
            )

    if method == "macro":
        final_moments = state
        final_active_fraction = 0.0
    elif method == "full_dvm":
        final_moments = state.moments()
        final_active_fraction = 1.0
    else:
        final_moments = state.moments
        final_active_fraction = float(np.mean(state.active))

    balance_limit = 2.0e-8 if method == "adaptive" else 2.0e-9
    contracts_pass = bool(
        float(metrics["maximum_balance_residual"]) < balance_limit
        and (
            method != "full_dvm"
            or float(metrics["maximum_projection_residual"]) < 2.0e-9
        )
        and (
            method != "adaptive"
            or (
                float(metrics["maximum_micro_macro_residual"]) < 2.0e-8
                and int(metrics["blocked_births"]) == 0
            )
        )
    )
    summary = {
        "stage": "25A-lite",
        "method": method,
        "scope": config["scope"],
        "configuration": {
            **config,
            "dt": dt,
            "tau": tau,
            "final_time": steps * dt,
            "velocity_cells": vgrid.size,
        },
        "completed_steps": steps,
        "contracts_pass": contracts_pass,
        "timing_seconds": float(metrics["wall_seconds"]),
        "metrics": {
            **metrics,
            "mean_active_fraction": (
                float(metrics["active_fraction_sum"]) / steps
                if method == "adaptive"
                else final_active_fraction
            ),
            "final_active_fraction": final_active_fraction,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    _atomic_savez(
        output / f"{method}_final.npz",
        x=xgrid.centers,
        moments=final_moments,
        active=(state.active if method == "adaptive" else np.asarray([], dtype=bool)),
        metadata=np.asarray(json.dumps(_jsonable(summary))),
    )
    (output / f"{method}_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(
        json.dumps(
            {
                "event": "STAGE25A_LITE_METHOD_COMPLETE",
                "method": method,
                "contracts_pass": contracts_pass,
                "wall_seconds": metrics["wall_seconds"],
                "output": str(output),
            }
        ),
        flush=True,
    )
    return summary


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def collect(input_root: Path, bundle: Path) -> dict[str, object]:
    summaries = {
        method: _load_json(input_root / method / f"{method}_summary.json")
        for method in METHODS
    }
    configurations = [summary["configuration"] for summary in summaries.values()]
    if any(item != configurations[0] for item in configurations[1:]):
        raise ValueError("method outputs do not share an identical lite configuration")

    final = {}
    x = None
    for method in METHODS:
        with np.load(
            input_root / method / f"{method}_final.npz", allow_pickle=False
        ) as saved:
            if x is None:
                x = saved["x"].copy()
            elif not np.array_equal(x, saved["x"]):
                raise ValueError("method outputs do not share an identical spatial grid")
            final[method] = saved["moments"].copy()

    reference = final["full_dvm"]
    errors = {}
    profiles = {method: shock_profiles(values) for method, values in final.items()}
    for method in ("macro", "adaptive"):
        errors[method] = {
            "E4_percent": 100.0
            * normalized_profile_error(
                final[method][:, FOURTH_POSITIONS],
                reference[:, FOURTH_POSITIONS],
            ),
            "profiles_percent": {
                name: 100.0
                * normalized_profile_error(values, profiles["full_dvm"][name])
                for name, values in profiles[method].items()
            },
        }

    adaptive_metrics = summaries["adaptive"]["metrics"]
    dvm_seconds = float(summaries["full_dvm"]["timing_seconds"])
    adaptive_seconds = float(summaries["adaptive"]["timing_seconds"])
    speed_ratio = dvm_seconds / max(adaptive_seconds, 1.0e-30)
    contracts_pass = all(bool(item["contracts_pass"]) for item in summaries.values())
    accuracy_pass = bool(errors["adaptive"]["E4_percent"] < 3.0)
    economy_pass = bool(
        float(adaptive_metrics["mean_active_fraction"]) < 0.50
        and speed_ratio >= 2.0
    )
    if not contracts_pass:
        decision = "LITE_FAIL_CONTRACT"
    elif not accuracy_pass:
        decision = "LITE_ACCURACY_HOLD"
    elif not economy_pass:
        decision = "LITE_ACCURACY_PASS_SPEED_HOLD"
    else:
        decision = "LITE_PASS_NOT_PUBLICATION_EVIDENCE"

    combined = {
        "stage": "25A-lite",
        "decision": decision,
        "scope": "optimization decision only; not publication evidence",
        "configuration": configurations[0],
        "method_contracts_pass": {
            method: bool(summary["contracts_pass"])
            for method, summary in summaries.items()
        },
        "errors": errors,
        "timing_seconds": {
            method: float(summary["timing_seconds"])
            for method, summary in summaries.items()
        },
        "full_dvm_over_adaptive_wall_time": speed_ratio,
        "mean_active_fraction": float(adaptive_metrics["mean_active_fraction"]),
        "final_active_fraction": float(adaptive_metrics["final_active_fraction"]),
    }
    (input_root / "STAGE25A_LITE_SUMMARY.json").write_text(
        json.dumps(_jsonable(combined), indent=2) + "\n", encoding="utf-8"
    )

    profile_names = tuple(profiles["full_dvm"])
    with (input_root / "STAGE25A_LITE_PROFILES.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["x"]
            + [f"{method}_{name}" for method in METHODS for name in profile_names]
        )
        for cell, coordinate in enumerate(x):
            writer.writerow(
                [coordinate]
                + [
                    profiles[method][name][cell]
                    for method in METHODS
                    for name in profile_names
                ]
            )

    report = [
        "# Stage 25A qualification-lite",
        "",
        f"- Decision: **{decision}**",
        "- Scope: optimization decision only; not publication evidence",
        f"- Spatial cells / steps: {configurations[0]['spatial_cells']} / {configurations[0]['steps']}",
        f"- Velocity cells: {configurations[0]['velocity_cells']}",
        f"- Adaptive E4 error vs DVM: {errors['adaptive']['E4_percent']:.6f}%",
        f"- Macro E4 error vs DVM: {errors['macro']['E4_percent']:.6f}%",
        f"- Mean adaptive active fraction: {100*float(adaptive_metrics['mean_active_fraction']):.3f}%",
        f"- DVM/adaptive wall-time ratio: {speed_ratio:.3f}x",
        f"- Method contracts: {combined['method_contracts_pass']}",
        "",
        "This shortened, coarser run decides whether closure optimization is",
        "worth pursuing. It does not replace the frozen 160-cell qualification.",
    ]
    (input_root / "STAGE25A_LITE_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary_bundle = bundle.with_name(bundle.name + ".tmp")
    with zipfile.ZipFile(
        temporary_bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(input_root.rglob("*")):
            if path.is_file() and path not in {bundle, temporary_bundle}:
                archive.write(path, path.relative_to(input_root.parent))
    os.replace(temporary_bundle, bundle)
    print(
        json.dumps(
            {
                "event": "STAGE25A_LITE_COLLECT_COMPLETE",
                "decision": decision,
                "bundle": str(bundle),
                "summary": str(input_root / "STAGE25A_LITE_SUMMARY.json"),
            }
        ),
        flush=True,
    )
    return combined


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--method", choices=METHODS, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--steps", type=int, default=600)
    run_parser.add_argument("--checkpoint-every", type=int, default=50)
    run_parser.add_argument("--progress-every", type=int, default=10)
    run_parser.add_argument("--resume-checkpoint", type=Path, default=None)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--input-root", type=Path, required=True)
    collect_parser.add_argument("--bundle", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.action == "run":
        run_method(
            args.method,
            args.output,
            steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            progress_every=args.progress_every,
            resume_checkpoint=args.resume_checkpoint,
        )
    else:
        collect(args.input_root, args.bundle)


if __name__ == "__main__":
    main()

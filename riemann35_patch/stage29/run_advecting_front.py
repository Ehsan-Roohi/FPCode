#!/usr/bin/env python3
"""Run the Stage-29 advecting causal kinetic-front qualification."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fpcode-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    AdaptiveSpatialState,
    DVMGrid,
    DVMState,
    SpatialDVMState,
    SpatialGrid1D,
    adaptive_shock_step,
    full_dvm_shock_step,
    regularized_four_delta_state,
    stage25_hysteresis,
)
from hyqmom_fp.dvm_reference import initialize_diagonal_gaussian_mixture  # noqa: E402
from riemann35_patch.stage28.run_localized_pocket import (  # noqa: E402
    M400,
    _adaptive_tail,
    _causal_activation_audit,
    _dvm_tail,
    _jsonable,
    _maximum_balance,
    _minimum_margin,
    _relative_error,
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
    presets = {
        "smoke": (24, 6, 4),
        "workstation": (48, 24, 8),
    }
    nx, default_steps, sensor_interval = presets[mode]
    values: dict[str, object] = {
        "mode": mode,
        "x_lower": -0.25 * nx,
        "x_upper": 0.25 * nx,
        "spatial_cells": nx,
        "pocket_center": -2.0,
        "pocket_half_width": 1.0,
        "initial_buffer_cells": 1,
        "bulk_velocity_x": 1.0,
        "velocity_lower": (-2.5, -2.5, -1.5),
        "velocity_upper": (3.5, 2.5, 1.5),
        "coarse_velocity_shape": (19, 15, 11),
        "refined_velocity_shape": (21, 17, 13),
        "regularization_fraction": 0.12,
        "cfl": 0.15,
        "tau": 1.0,
        "prandtl": 2.0 / 3.0,
        "steps": default_steps if steps is None else steps,
        "sensor_interval_steps": sensor_interval,
        "macro_equilibrium_tolerance": 1.0e-12,
        "kinetic_front_on": stage25_hysteresis().tail_on,
    }
    if int(values["steps"]) < 1:
        raise ValueError("steps must be positive")
    return values


def _background_components() -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    return ((1.0, np.zeros(3), np.eye(3) / 3.0),)


def _advecting_components(
    bulk_velocity_x: float, regularization_fraction: float
) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    base = regularized_four_delta_state(
        regularization_fraction=regularization_fraction
    )
    shift = np.asarray([bulk_velocity_x, 0.0, 0.0])
    return tuple(
        (weight, center + shift, covariance)
        for weight, center, covariance in base.components
    )


def _initialize_full(
    xgrid: SpatialGrid1D,
    vgrid: DVMGrid,
    config: dict[str, object],
) -> tuple[SpatialDVMState, DVMState, np.ndarray]:
    background, _ = initialize_diagonal_gaussian_mixture(
        vgrid, _background_components(), match_exact_moments=True
    )
    pocket, _ = initialize_diagonal_gaussian_mixture(
        vgrid,
        _advecting_components(
            float(config["bulk_velocity_x"]),
            float(config["regularization_fraction"]),
        ),
        match_exact_moments=True,
    )
    pocket_mask = (
        np.abs(xgrid.centers - float(config["pocket_center"]))
        < float(config["pocket_half_width"])
    )
    masses = np.where(
        pocket_mask[:, None], pocket.masses[None, :], background.masses[None, :]
    )
    return SpatialDVMState(xgrid, vgrid, masses), background, pocket_mask


def _initialize_adaptive(
    full: SpatialDVMState,
    pocket_mask: np.ndarray,
    buffer_cells: int,
) -> tuple[AdaptiveSpatialState, np.ndarray]:
    active = pocket_mask.copy()
    pocket_cells = np.flatnonzero(pocket_mask)
    lower = max(0, int(pocket_cells[0]) - buffer_cells)
    upper = min(full.spatial_grid.cells, int(pocket_cells[-1]) + buffer_cells + 1)
    active[lower:upper] = True
    masses = np.zeros_like(full.masses)
    masses[active] = full.masses[active]
    state = AdaptiveSpatialState(
        spatial_grid=full.spatial_grid,
        velocity_grid=full.velocity_grid,
        moments=full.moments(),
        micro_masses=masses,
        active=active,
        active_steps=np.zeros(full.spatial_grid.cells, dtype=int),
        release_counter=np.zeros(full.spatial_grid.cells, dtype=int),
        global_step=0,
        transition_count=int(np.sum(active)),
        blocked_births=0,
    )
    return state, active.copy()


def _plot(
    output: Path,
    x: np.ndarray,
    dt: float,
    refined_moments: np.ndarray,
    coarse_moments: np.ndarray,
    adaptive_moments: np.ndarray,
    refined_tail: np.ndarray,
    coarse_tail: np.ndarray,
    adaptive_tail: np.ndarray,
    active_history: np.ndarray,
    activation_counts: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(x, refined_moments[-1, :, M400], "k-", label="refined DVM")
    axes[0, 0].plot(x, coarse_moments[-1, :, M400], "C0--", label="coarse DVM")
    axes[0, 0].plot(x, adaptive_moments[-1, :, M400], "C2-.", label="adaptive")
    axes[0, 0].set(title="Final retained M400", xlabel="x", ylabel="M400")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, refined_tail[-1], "k-", label="refined DVM")
    axes[0, 1].plot(x, coarse_tail[-1], "C0--", label="coarse DVM")
    axes[0, 1].plot(x, adaptive_tail[-1], "C2-.", label="adaptive")
    axes[0, 1].set(title="Final predictive M420", xlabel="x", ylabel="M420")

    extent = (x[0], x[-1], active_history.shape[0] * dt, 0.0)
    axes[1, 0].imshow(
        active_history.astype(float),
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )
    axes[1, 0].set(title="Advecting kinetic-memory support", xlabel="x", ylabel="time")

    times = np.arange(active_history.shape[0]) * dt
    axes[1, 1].plot(
        times[1:], np.cumsum(activation_counts), "C4-o", markersize=3
    )
    axes[1, 1].set(
        title="Causal kinetic-front births",
        xlabel="time",
        ylabel="cumulative new cells",
    )
    fig.suptitle("Stage 29 advecting causal kinetic front")
    fig.savefig(output / "stage29_advecting_front.png", dpi=180)
    plt.close(fig)


def run(config: dict[str, object], output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    xgrid = SpatialGrid1D(
        float(config["x_lower"]),
        float(config["x_upper"]),
        int(config["spatial_cells"]),
    )
    coarse_grid = DVMGrid(
        tuple(config["velocity_lower"]),
        tuple(config["velocity_upper"]),
        tuple(config["coarse_velocity_shape"]),
    )
    refined_grid = DVMGrid(
        tuple(config["velocity_lower"]),
        tuple(config["velocity_upper"]),
        tuple(config["refined_velocity_shape"]),
    )
    tau = float(config["tau"])
    prandtl = float(config["prandtl"])
    steps = int(config["steps"])
    maximum_speed = max(
        np.max(np.abs(coarse_grid.centers()[:, 0])),
        np.max(np.abs(refined_grid.centers()[:, 0])),
    )
    dt = float(config["cfl"]) * xgrid.width / maximum_speed

    coarse, coarse_background, pocket_mask = _initialize_full(xgrid, coarse_grid, config)
    refined, refined_background, refined_mask = _initialize_full(
        xgrid, refined_grid, config
    )
    if not np.array_equal(pocket_mask, refined_mask):
        raise RuntimeError("coarse/refined pocket masks differ")
    adaptive, initial_active = _initialize_adaptive(
        coarse, pocket_mask, int(config["initial_buffer_cells"])
    )

    coarse_history = [coarse.moments()]
    refined_history = [refined.moments()]
    adaptive_history = [adaptive.moments.copy()]
    coarse_tail_history = [_dvm_tail(coarse)]
    refined_tail_history = [_dvm_tail(refined)]
    adaptive_tail_history = [_adaptive_tail(adaptive)]
    active_history = [adaptive.active.copy()]
    coarse_diagnostics: list[dict[str, object]] = []
    refined_diagnostics: list[dict[str, object]] = []
    adaptive_diagnostics: list[dict[str, object]] = []
    timing = {"coarse_dvm": 0.0, "refined_dvm": 0.0, "adaptive": 0.0}

    for step in range(1, steps + 1):
        started = time.perf_counter()
        coarse, coarse_diag = full_dvm_shock_step(
            coarse,
            dt,
            tau,
            coarse_background,
            coarse_background,
            prandtl=prandtl,
        )
        timing["coarse_dvm"] += time.perf_counter() - started

        started = time.perf_counter()
        refined, refined_diag = full_dvm_shock_step(
            refined,
            dt,
            tau,
            refined_background,
            refined_background,
            prandtl=prandtl,
        )
        timing["refined_dvm"] += time.perf_counter() - started

        started = time.perf_counter()
        adaptive, adaptive_diag = adaptive_shock_step(
            adaptive,
            dt,
            tau,
            coarse_background,
            coarse_background,
            prandtl=prandtl,
            sensor_interval_steps=int(config["sensor_interval_steps"]),
            macro_equilibrium_tolerance=float(
                config["macro_equilibrium_tolerance"]
            ),
            birth_carrier=coarse_background,
            kinetic_front_on=float(config["kinetic_front_on"]),
        )
        timing["adaptive"] += time.perf_counter() - started

        coarse_history.append(coarse.moments())
        refined_history.append(refined.moments())
        adaptive_history.append(adaptive.moments.copy())
        coarse_tail_history.append(_dvm_tail(coarse))
        refined_tail_history.append(_dvm_tail(refined))
        adaptive_tail_history.append(_adaptive_tail(adaptive))
        active_history.append(adaptive.active.copy())
        coarse_diagnostics.append(asdict(coarse_diag))
        refined_diagnostics.append(asdict(refined_diag))
        adaptive_diagnostics.append(asdict(adaptive_diag))
        print(
            json.dumps(
                _jsonable({
                    "event": "STAGE29_PROGRESS",
                    "step": step,
                    "steps": steps,
                    "active_cells": int(np.sum(adaptive.active)),
                    "activation_cells": adaptive_diag.activation_cells,
                    "activation_reasons": adaptive_diag.activation_reasons,
                })
            ),
            flush=True,
        )

    coarse_array = np.asarray(coarse_history)
    refined_array = np.asarray(refined_history)
    adaptive_array = np.asarray(adaptive_history)
    coarse_tail_array = np.asarray(coarse_tail_history)
    refined_tail_array = np.asarray(refined_tail_history)
    adaptive_tail_array = np.asarray(adaptive_tail_history)
    active_array = np.asarray(active_history)
    activation_counts = np.asarray(
        [int(item["activations"]) for item in adaptive_diagnostics]
    )

    final_errors = {
        "coarse_vs_refined_M400_percent": 100.0 * _relative_error(
            coarse_array[-1, :, M400], refined_array[-1, :, M400]
        ),
        "coarse_vs_refined_M420_percent": 100.0 * _relative_error(
            coarse_tail_array[-1], refined_tail_array[-1]
        ),
        "adaptive_vs_refined_M400_percent": 100.0 * _relative_error(
            adaptive_array[-1, :, M400], refined_array[-1, :, M400]
        ),
        "adaptive_vs_refined_M420_percent": 100.0 * _relative_error(
            adaptive_tail_array[-1], refined_tail_array[-1]
        ),
    }
    space_time_errors = {
        "adaptive_vs_refined_M400_percent": 100.0 * _relative_error(
            adaptive_array[:, :, M400], refined_array[:, :, M400]
        ),
        "adaptive_vs_refined_M420_percent": 100.0 * _relative_error(
            adaptive_tail_array, refined_tail_array
        ),
    }
    causal_pass = _causal_activation_audit(active_history, adaptive_diagnostics)
    births = [
        (cell, source, donor, carrier, fraction, reason, signal)
        for item in adaptive_diagnostics
        for cell, source, donor, carrier, fraction, reason, signal in zip(
            item["activation_cells"],
            item["activation_sources"],
            item["activation_donor_cells"],
            item["activation_carrier_used"],
            item["activation_donor_fractions"],
            item["activation_reasons"],
            item["activation_front_signals"],
        )
    ]
    front_births = [item for item in births if "kinetic_front" in str(item[5])]
    initial_right = int(np.flatnonzero(initial_active)[-1])
    right_front_births = [
        item
        for item in front_births
        if int(item[0]) > initial_right and item[1] == "left_neighbor"
    ]
    front_provenance_pass = bool(
        front_births
        and right_front_births
        and all(bool(item[3]) for item in front_births)
        and all(item[4] is not None and 0.0 <= float(item[4]) <= 1.0 for item in front_births)
        and all(
            item[6] is not None
            and float(item[6]) >= float(config["kinetic_front_on"])
            for item in front_births
        )
    )

    maximum_projection = max(
        max(float(item["maximum_projection_residual"]) for item in coarse_diagnostics),
        max(float(item["maximum_projection_residual"]) for item in refined_diagnostics),
    )
    maximum_collision_drift = max(
        max(float(item["maximum_collision_invariant_drift"]) for item in coarse_diagnostics),
        max(float(item["maximum_collision_invariant_drift"]) for item in refined_diagnostics),
    )
    maximum_sync = max(
        float(item["maximum_micro_macro_residual"]) for item in adaptive_diagnostics
    )
    maximum_balance = max(
        _maximum_balance(coarse_diagnostics),
        _maximum_balance(refined_diagnostics),
        _maximum_balance(adaptive_diagnostics),
    )
    minimum_active_mass = min(
        float(item["minimum_micro_mass"])
        for item in adaptive_diagnostics
        if float(item["active_fraction"]) > 0.0
    )
    minimum_dvm_mass = min(float(np.min(coarse.masses)), float(np.min(refined.masses)))
    minimum_margin = min(
        _minimum_margin(coarse_history),
        _minimum_margin(refined_history),
        _minimum_margin(adaptive_history),
    )
    expensive_sensor_evaluations = int(
        sum(
            int(item["activation_sensor_evaluations"])
            + int(item["release_sensor_evaluations"])
            for item in adaptive_diagnostics
        )
    )
    front_sensor_evaluations = int(
        sum(int(item["front_sensor_evaluations"]) for item in adaptive_diagnostics)
    )
    wall_ratio = timing["adaptive"] / max(timing["coarse_dvm"], 1.0e-30)
    numerical_pass = bool(
        causal_pass
        and front_provenance_pass
        and minimum_dvm_mass > 0.0
        and minimum_active_mass > 0.0
        and minimum_margin >= -2.0e-12
        and maximum_balance < 2.0e-8
        and maximum_projection < 2.0e-9
        and maximum_collision_drift < 2.0e-9
        and maximum_sync < 2.0e-8
        and all(value < 3.0 for value in final_errors.values())
        and all(value < 3.0 for value in space_time_errors.values())
    )
    localization_pass = bool(
        float(np.max(np.mean(active_array, axis=1))) < 0.5
        and float(np.mean(active_array)) < 0.5
    )
    performance_pass = wall_ratio < 1.0
    qualification_pass = numerical_pass and localization_pass and performance_pass
    if config["mode"] == "smoke":
        decision = "SMOKE_PASS" if numerical_pass and localization_pass else "SMOKE_HOLD"
    else:
        decision = "WORKSTATION_PASS" if qualification_pass else "WORKSTATION_HOLD"

    contracts = {
        "synchronous_causal_provenance_pass": causal_pass,
        "nontrivial_kinetic_front_birth_pass": front_provenance_pass,
        "numerical_accuracy_and_invariants_pass": numerical_pass,
        "kinetic_fraction_below_50_percent_pass": localization_pass,
        "measured_speedup_vs_coarse_full_dvm_pass": performance_pass,
        "all_qualification_gates_pass": qualification_pass,
        "minimum_dvm_mass": minimum_dvm_mass,
        "minimum_adaptive_active_mass": minimum_active_mass,
        "minimum_realizability_margin": minimum_margin,
        "maximum_balance_residual": maximum_balance,
        "maximum_projection_residual": maximum_projection,
        "maximum_collision_invariant_drift": maximum_collision_drift,
        "maximum_micro_macro_sync_residual": maximum_sync,
    }
    summary = {
        "stage": "29",
        "case": "advecting_regularized_four_delta_kinetic_front",
        "decision": decision,
        "scope": "numerical qualification, not independent MD/DSMC physical validation",
        "configuration": {**config, "dt": dt, "final_time": steps * dt},
        "reference": "positive guided DVM on coarse and refined velocity grids",
        "primary_metrics": {
            "causal_births": len(births),
            "kinetic_front_births": len(front_births),
            "right_moving_front_births": len(right_front_births),
            "mean_active_fraction_percent": 100.0 * float(np.mean(active_array)),
            "peak_active_fraction_percent": 100.0 * float(
                np.max(np.mean(active_array, axis=1))
            ),
            "final_active_fraction_percent": 100.0 * float(np.mean(active_array[-1])),
            "expensive_sensor_evaluations": expensive_sensor_evaluations,
            "front_sensor_evaluations": front_sensor_evaluations,
            "adaptive_over_coarse_dvm_wall_time": wall_ratio,
            "measured_speedup_factor": 1.0 / max(wall_ratio, 1.0e-30),
            "final_errors": final_errors,
            "space_time_errors": space_time_errors,
        },
        "contracts": contracts,
        "activation_history": [
            {
                "step": step,
                "active_cells_before": np.flatnonzero(active_array[step - 1]),
                "active_cells_after": np.flatnonzero(active_array[step]),
                "activation_cells": item["activation_cells"],
                "activation_sources": item["activation_sources"],
                "activation_donor_cells": item["activation_donor_cells"],
                "activation_carrier_used": item["activation_carrier_used"],
                "activation_donor_fractions": item["activation_donor_fractions"],
                "activation_reasons": item["activation_reasons"],
                "activation_front_signals": item["activation_front_signals"],
            }
            for step, item in enumerate(adaptive_diagnostics, start=1)
        ],
        "timing_seconds": {**timing, "total": float(sum(timing.values()))},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }

    np.savez_compressed(
        output / "stage29_advecting_front_histories.npz",
        x=xgrid.centers,
        dt=np.asarray(dt),
        refined_moments=refined_array,
        coarse_moments=coarse_array,
        adaptive_moments=adaptive_array,
        refined_M420=refined_tail_array,
        coarse_M420=coarse_tail_array,
        adaptive_M420=adaptive_tail_array,
        adaptive_active=active_array,
        activation_counts=activation_counts,
    )
    with (output / "stage29_advecting_front_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_jsonable(summary), stream, indent=2, allow_nan=False)
    _plot(
        output,
        xgrid.centers,
        dt,
        refined_array,
        coarse_array,
        adaptive_array,
        refined_tail_array,
        coarse_tail_array,
        adaptive_tail_array,
        active_array,
        activation_counts,
    )
    report = [
        "# Stage 29 advecting causal kinetic-front result",
        "",
        f"- Decision: **{decision}**",
        f"- Causal births / kinetic-front births: {len(births)} / {len(front_births)}",
        f"- Mean/peak/final kinetic fraction: {100*np.mean(active_array):.3f}% / {100*np.max(np.mean(active_array,axis=1)):.3f}% / {100*np.mean(active_array[-1]):.3f}%",
        f"- Final adaptive M400 error vs refined DVM: {final_errors['adaptive_vs_refined_M400_percent']:.6f}%",
        f"- Final adaptive M420 error vs refined DVM: {final_errors['adaptive_vs_refined_M420_percent']:.6f}%",
        f"- Space-time adaptive M400/M420 errors: {space_time_errors['adaptive_vs_refined_M400_percent']:.6f}% / {space_time_errors['adaptive_vs_refined_M420_percent']:.6f}%",
        f"- Expensive/front sensor evaluations: {expensive_sensor_evaluations} / {front_sensor_evaluations}",
        f"- Adaptive/coarse-DVM wall-time ratio: {wall_ratio:.3f}x",
        f"- Measured speedup: {1/max(wall_ratio,1e-30):.3f}x",
        f"- Maximum finite-volume balance residual: {maximum_balance:.3e}",
        f"- Maximum micro/macro sync residual: {maximum_sync:.3e}",
        "",
        "Every new front cell inherited a positive upwind carrier--donor proposal",
        "from a neighbour active at the start of the step.  No tail was invented",
        "from the 35 retained moments.  Independent MD/DSMC validation remains",
        "necessary before making a physical-fidelity claim.",
    ]
    (output / "STAGE29_ADVECTING_FRONT_RESULT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = arguments()
    config = configuration(args.mode, args.steps)
    output = args.output
    if output is None:
        output = REPOSITORY_ROOT / "results" / "riemann35_stage29" / args.mode
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
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the Stage-28 localized four-delta spatial qualification."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

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
    HYQMOM_35_INDICES,
    SpatialDVMState,
    SpatialGrid1D,
    adaptive_shock_step,
    full_dvm_shock_step,
    realizability_margin_35,
    reconstruct_gaussian_mixture_quadrature,
    regularized_four_delta_state,
)
from hyqmom_fp.dvm_reference import initialize_diagonal_gaussian_mixture  # noqa: E402


POSITION = {index: offset for offset, index in enumerate(HYQMOM_35_INDICES)}
M400 = POSITION[(4, 0, 0)]
M420_INDEX = (4, 2, 0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "workstation", "unity"), default="workstation"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sensor-interval", type=int)
    return parser.parse_args()


def configuration(
    mode: str,
    steps: int | None = None,
    sensor_interval: int | None = None,
) -> dict[str, object]:
    presets = {
        "smoke": (16, 2, 2, 1, (17, 15, 11), (19, 17, 13)),
        "workstation": (24, 9, 9, 1, (17, 15, 11), (19, 17, 13)),
        "unity": (48, 24, 8, 2, (17, 15, 11), (19, 17, 13)),
    }
    nx, default_steps, default_sensor, buffer_cells, coarse, refined = presets[mode]
    values: dict[str, object] = {
        "mode": mode,
        "x_lower": -0.25 * nx,
        "x_upper": 0.25 * nx,
        "spatial_cells": nx,
        "pocket_half_width": 1.0,
        "initial_buffer_cells": buffer_cells,
        "velocity_lower": (-2.5, -2.5, -1.5),
        "velocity_upper": (2.5, 2.5, 1.5),
        "coarse_velocity_shape": coarse,
        "refined_velocity_shape": refined,
        "regularization_fraction": 0.12,
        "cfl": 0.15,
        "tau": 1.0,
        "prandtl": 2.0 / 3.0,
        "steps": default_steps if steps is None else steps,
        "sensor_interval_steps": (
            default_sensor if sensor_interval is None else sensor_interval
        ),
        "macro_equilibrium_tolerance": 1.0e-12,
    }
    if int(values["steps"]) < 1:
        raise ValueError("steps must be positive")
    if int(values["sensor_interval_steps"]) < 1:
        raise ValueError("sensor interval must be positive")
    return values


def _background_components() -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    # Match the four-delta unit energy trace: theta = trace(Cov)/3 = 1/3.
    return ((1.0, np.zeros(3), np.eye(3) / 3.0),)


def _initialize_full(
    xgrid: SpatialGrid1D,
    vgrid: DVMGrid,
    *,
    pocket_half_width: float,
    regularization_fraction: float,
) -> tuple[SpatialDVMState, DVMState, np.ndarray]:
    pocket_data = regularized_four_delta_state(
        regularization_fraction=regularization_fraction
    )
    background, _ = initialize_diagonal_gaussian_mixture(
        vgrid, _background_components(), match_exact_moments=True
    )
    pocket, _ = initialize_diagonal_gaussian_mixture(
        vgrid, pocket_data.components, match_exact_moments=True
    )
    pocket_mask = np.abs(xgrid.centers) < pocket_half_width
    if not np.any(pocket_mask):
        raise ValueError("localized pocket contains no spatial cells")
    masses = np.where(
        pocket_mask[:, None], pocket.masses[None, :], background.masses[None, :]
    )
    return SpatialDVMState(xgrid, vgrid, masses), background, pocket_mask


def _initialize_adaptive(
    full: SpatialDVMState,
    pocket_mask: np.ndarray,
    *,
    buffer_cells: int,
) -> AdaptiveSpatialState:
    nx = full.spatial_grid.cells
    active = np.asarray(pocket_mask, dtype=bool).copy()
    pocket_cells = np.flatnonzero(active)
    lower = max(0, int(pocket_cells[0]) - buffer_cells)
    upper = min(nx, int(pocket_cells[-1]) + buffer_cells + 1)
    active[lower:upper] = True
    masses = np.zeros_like(full.masses)
    masses[active] = full.masses[active]
    return AdaptiveSpatialState(
        spatial_grid=full.spatial_grid,
        velocity_grid=full.velocity_grid,
        moments=full.moments(),
        micro_masses=masses,
        active=active,
        active_steps=np.zeros(nx, dtype=int),
        release_counter=np.zeros(nx, dtype=int),
        global_step=0,
        transition_count=int(np.sum(active)),
        blocked_births=0,
    )


def _dvm_tail(state: SpatialDVMState) -> np.ndarray:
    feature = state.velocity_grid.feature_matrix((M420_INDEX,))[:, 0]
    return state.masses @ feature


def _algebraic_tail(moments: np.ndarray) -> np.ndarray:
    values = []
    for vector in moments:
        quadrature = reconstruct_gaussian_mixture_quadrature(vector)
        nodes = quadrature.nodes
        values.append(
            np.dot(
                quadrature.weights,
                nodes[:, 0] ** M420_INDEX[0]
                * nodes[:, 1] ** M420_INDEX[1],
            )
        )
    return np.asarray(values)


def _adaptive_tail(state: AdaptiveSpatialState) -> np.ndarray:
    result = _algebraic_tail(state.moments)
    if np.any(state.active):
        feature = state.velocity_grid.feature_matrix((M420_INDEX,))[:, 0]
        result[state.active] = state.micro_masses[state.active] @ feature
    return result


def _relative_error(model: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(model) - np.asarray(reference))
        / max(np.linalg.norm(reference), 1.0e-30)
    )


def _minimum_margin(history: Sequence[np.ndarray]) -> float:
    return float(
        min(realizability_margin_35(cell) for field in history for cell in field)
    )


def _causal_activation_audit(
    active_history: Sequence[np.ndarray], diagnostics: Sequence[dict[str, object]]
) -> bool:
    for before, item in zip(active_history[:-1], diagnostics):
        for cell, source, donor in zip(
            item["activation_cells"],
            item["activation_sources"],
            item["activation_donor_cells"],
        ):
            cell = int(cell)
            if before[cell]:
                return False
            if source == "left_inflow":
                if cell != 0 or donor is not None:
                    return False
            elif source == "right_inflow":
                if cell != len(before) - 1 or donor is not None:
                    return False
            elif source in ("left_neighbor", "right_neighbor"):
                if donor is None or abs(cell - int(donor)) != 1 or not before[int(donor)]:
                    return False
            else:
                return False
    return True


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _maximum_balance(items: Sequence[dict[str, object]]) -> float:
    return float(
        max(
            max(
                float((item["transport"] if "transport" in item else item)[key])
                for key in (
                    "mass_balance_residual",
                    "momentum_balance_residual",
                    "energy_balance_residual",
                )
            )
            for item in items
        )
    )


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
    axes[1, 0].set(title="Localized kinetic-memory support", xlabel="x", ylabel="time")

    times = np.arange(active_history.shape[0]) * dt
    axes[1, 1].semilogy(
        times,
        [100.0 * _relative_error(a, r) for a, r in zip(adaptive_tail, refined_tail)],
        "C2-",
        label="adaptive",
    )
    axes[1, 1].semilogy(
        times,
        [100.0 * _relative_error(c, r) for c, r in zip(coarse_tail, refined_tail)],
        "C0--",
        label="coarse DVM",
    )
    axes[1, 1].axhline(3.0, color="0.35", linestyle=":", linewidth=1)
    axes[1, 1].set(title="M420 profile error", xlabel="time", ylabel="relative L2 [%]")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Stage 28 localized nonequilibrium pocket")
    fig.savefig(output / "stage28_localized_pocket.png", dpi=180)
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
    sensor_interval = int(config["sensor_interval_steps"])
    maximum_speed = max(
        np.max(np.abs(coarse_grid.centers()[:, 0])),
        np.max(np.abs(refined_grid.centers()[:, 0])),
    )
    dt = float(config["cfl"]) * xgrid.width / maximum_speed

    coarse, coarse_boundary, pocket_mask = _initialize_full(
        xgrid,
        coarse_grid,
        pocket_half_width=float(config["pocket_half_width"]),
        regularization_fraction=float(config["regularization_fraction"]),
    )
    refined, refined_boundary, refined_pocket_mask = _initialize_full(
        xgrid,
        refined_grid,
        pocket_half_width=float(config["pocket_half_width"]),
        regularization_fraction=float(config["regularization_fraction"]),
    )
    if not np.array_equal(pocket_mask, refined_pocket_mask):
        raise RuntimeError("coarse/refined pocket masks differ")
    adaptive = _initialize_adaptive(
        coarse,
        pocket_mask,
        buffer_cells=int(config["initial_buffer_cells"]),
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
            coarse, dt, tau, coarse_boundary, coarse_boundary, prandtl=prandtl
        )
        timing["coarse_dvm"] += time.perf_counter() - started

        started = time.perf_counter()
        refined, refined_diag = full_dvm_shock_step(
            refined, dt, tau, refined_boundary, refined_boundary, prandtl=prandtl
        )
        timing["refined_dvm"] += time.perf_counter() - started

        started = time.perf_counter()
        adaptive, adaptive_diag = adaptive_shock_step(
            adaptive,
            dt,
            tau,
            coarse_boundary,
            coarse_boundary,
            prandtl=prandtl,
            sensor_interval_steps=sensor_interval,
            macro_equilibrium_tolerance=float(
                config["macro_equilibrium_tolerance"]
            ),
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
                {
                    "event": "STAGE28_PROGRESS",
                    "step": step,
                    "steps": steps,
                    "active_cells": int(np.sum(adaptive.active)),
                    "sensor_evaluated": adaptive_diag.sensor_evaluated,
                    "sensor_evaluations": (
                        adaptive_diag.activation_sensor_evaluations
                        + adaptive_diag.release_sensor_evaluations
                    ),
                    "activations": adaptive_diag.activations,
                    "macro_equilibrium_shortcuts": (
                        adaptive_diag.macro_equilibrium_shortcuts
                    ),
                }
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
    sensor_evaluations = int(
        sum(
            int(item["activation_sensor_evaluations"])
            + int(item["release_sensor_evaluations"])
            for item in adaptive_diagnostics
        )
    )
    equilibrium_shortcuts = int(
        sum(int(item["macro_equilibrium_shortcuts"]) for item in adaptive_diagnostics)
    )
    dense_stage27_evaluations = 2 * xgrid.cells * steps
    wall_ratio = timing["adaptive"] / max(timing["coarse_dvm"], 1.0e-30)
    numerical_pass = bool(
        causal_pass
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
    performance_pass = bool(wall_ratio < 1.0)
    qualification_pass = numerical_pass and localization_pass and performance_pass
    if config["mode"] == "smoke":
        decision = "SMOKE_PASS" if numerical_pass and localization_pass else "SMOKE_HOLD"
    else:
        decision = "QUALIFICATION_PASS" if qualification_pass else "QUALIFICATION_HOLD"

    contracts = {
        "synchronous_causal_provenance_pass": causal_pass,
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
        "stage": "28",
        "case": "localized_regularized_four_delta_pocket",
        "decision": decision,
        "scope": "numerical qualification, not independent MD/DSMC physical validation",
        "configuration": {**config, "dt": dt, "final_time": steps * dt},
        "reference": "positive guided DVM on coarse and refined velocity grids",
        "primary_metrics": {
            "mean_active_fraction_percent": 100.0 * float(np.mean(active_array)),
            "peak_active_fraction_percent": 100.0 * float(
                np.max(np.mean(active_array, axis=1))
            ),
            "final_active_fraction_percent": 100.0 * float(np.mean(active_array[-1])),
            "sensor_evaluations": sensor_evaluations,
            "dense_stage27_style_sensor_evaluations": dense_stage27_evaluations,
            "sensor_evaluation_fraction_percent": (
                100.0 * sensor_evaluations / dense_stage27_evaluations
            ),
            "macro_equilibrium_shortcuts": equilibrium_shortcuts,
            "adaptive_over_coarse_dvm_wall_time": wall_ratio,
            "measured_speedup_factor": 1.0 / max(wall_ratio, 1.0e-30),
            "final_errors": final_errors,
            "space_time_errors": space_time_errors,
        },
        "contracts": contracts,
        "activation_history": [
            {
                "step": step,
                "sensor_evaluated": item["sensor_evaluated"],
                "active_cells_before": np.flatnonzero(active_array[step - 1]),
                "active_cells_after": np.flatnonzero(active_array[step]),
                "activation_cells": item["activation_cells"],
                "activation_sources": item["activation_sources"],
                "activation_donor_cells": item["activation_donor_cells"],
                "blocked_cells": item["blocked_cells"],
            }
            for step, item in enumerate(adaptive_diagnostics, start=1)
        ],
        "timing_seconds": {**timing, "total": float(sum(timing.values()))},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "interpretation": (
            "The initially active pocket and buffer are known positive DVM data. "
            "Any later birth remains synchronous and must inherit a causal donor."
        ),
    }

    np.savez_compressed(
        output / "stage28_localized_pocket_histories.npz",
        x=xgrid.centers,
        dt=np.asarray(dt),
        refined_moments=refined_array,
        coarse_moments=coarse_array,
        adaptive_moments=adaptive_array,
        refined_M420=refined_tail_array,
        coarse_M420=coarse_tail_array,
        adaptive_M420=adaptive_tail_array,
        adaptive_active=active_array,
    )
    with (output / "stage28_localized_pocket_summary.json").open(
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
    )
    report = [
        "# Stage 28 localized kinetic-pocket result",
        "",
        f"- Decision: **{decision}**",
        f"- Mean/peak kinetic fraction: {100*np.mean(active_array):.3f}% / {100*np.max(np.mean(active_array,axis=1)):.3f}%",
        f"- Final adaptive M400 error vs refined DVM: {final_errors['adaptive_vs_refined_M400_percent']:.6f}%",
        f"- Final adaptive M420 error vs refined DVM: {final_errors['adaptive_vs_refined_M420_percent']:.6f}%",
        f"- Space-time adaptive M400/M420 errors: {space_time_errors['adaptive_vs_refined_M400_percent']:.6f}% / {space_time_errors['adaptive_vs_refined_M420_percent']:.6f}%",
        f"- Sensor evaluations: {sensor_evaluations}/{dense_stage27_evaluations} ({100*sensor_evaluations/dense_stage27_evaluations:.3f}%)",
        f"- Exact/near-exact Maxwellian collision shortcuts: {equilibrium_shortcuts}",
        f"- Adaptive/coarse-DVM wall-time ratio: {wall_ratio:.3f}x",
        f"- Measured speedup: {1/max(wall_ratio,1e-30):.3f}x",
        f"- Minimum positive DVM mass: {minimum_dvm_mass:.3e}",
        f"- Maximum finite-volume balance residual: {maximum_balance:.3e}",
        f"- Maximum micro/macro sync residual: {maximum_sync:.3e}",
        "",
        "The initially retained pocket and buffer come from known positive DVM data.",
        "Skipped sensor intervals do not advance release counters or invent births.",
        "This qualifies numerical behavior for the implemented cubic FP operator;",
        "independent MD/DSMC validation is still required for physical fidelity.",
    ]
    (output / "STAGE28_LOCALIZED_POCKET_RESULT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = arguments()
    config = configuration(args.mode, args.steps, args.sensor_interval)
    output = args.output
    if output is None:
        output = REPOSITORY_ROOT / "results" / "riemann35_stage28" / args.mode
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

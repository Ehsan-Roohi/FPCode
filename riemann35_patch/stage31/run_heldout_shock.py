#!/usr/bin/env python3
"""Run the Stage-31 held-out Mach-2 normal-shock qualification."""

from __future__ import annotations

import argparse
import csv
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
    DVMGrid,
    SpatialDVMState,
    SpatialGrid1D,
    adaptive_shock_step,
    full_dvm_shock_step,
    initialize_adaptive_normal_shock,
    initialize_normal_shock_dvm,
    normal_shock_rankine_hugoniot,
    shock_profiles,
    stage25_hysteresis,
)
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


PROFILE_NAMES = (
    "rho",
    "velocity_x",
    "theta",
    "stress_xx",
    "heat_flux_x",
    "M300",
    "M400",
    "M420",
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
        "smoke": (16, 4, (15, 11, 11), (17, 13, 13)),
        "workstation": (48, 48, (19, 15, 15), (23, 17, 17)),
    }
    nx, default_steps, coarse_shape, refined_shape = presets[mode]
    values: dict[str, object] = {
        "mode": mode,
        "scope": "held-out numerical cross-case validation; not MD/DSMC evidence",
        "mach": 2.0,
        "x_lower": -0.25 * nx,
        "x_upper": 0.25 * nx,
        "spatial_cells": nx,
        "velocity_lower": (-8.0, -7.0, -7.0),
        "velocity_upper": (10.0, 7.0, 7.0),
        "coarse_velocity_shape": coarse_shape,
        "refined_velocity_shape": refined_shape,
        "cfl": 0.15,
        "tau": 1.0,
        "prandtl": 2.0 / 3.0,
        "steps": default_steps if steps is None else steps,
        # Preserve the frozen Stage-25A shock-interface initialization.  This
        # is a known t=0 support choice, not a sensor-threshold adjustment.
        "initial_active_half_width": 2,
        # Frozen complete lifecycle inherited from Stage 30.
        "sensor_interval_steps": 8,
        "release_sensor_interval_steps": 4,
        "causal_activation_candidates_only": True,
        "macro_equilibrium_tolerance": 1.0e-12,
        "kinetic_front_on": stage25_hysteresis().tail_on,
        "release_persistence_steps": 4,
        "profile_error_limit_percent": 3.0,
        "comparison_half_width": 3.0,
    }
    if int(values["steps"]) < 1:
        raise ValueError("steps must be positive")
    return values


def _profiles(moments: np.ndarray, tail: np.ndarray) -> dict[str, np.ndarray]:
    result = shock_profiles(moments)
    result["M420"] = np.asarray(tail)
    return result


def _profile_errors(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float]:
    return {
        name: 100.0 * _relative_error(candidate[name][mask], reference[name][mask])
        for name in PROFILE_NAMES
    }


def _plot(
    output: Path,
    x: np.ndarray,
    profiles: dict[str, dict[str, np.ndarray]],
    active_history: np.ndarray,
    dt: float,
) -> None:
    labels = {
        "rho": (r"Density $\rho$", r"$\rho$"),
        "velocity_x": (r"Flow velocity $u_x$", r"$u_x$"),
        "theta": (r"Temperature $\theta$", r"$\theta$"),
        "stress_xx": (r"Normal stress $\sigma_{xx}$", r"$\sigma_{xx}$"),
        "heat_flux_x": (r"Heat flux $q_x$", r"$q_x$"),
        "M400": (r"Retained fourth moment $M_{400}$", r"$M_{400}$"),
        "M420": (r"Predictive sixth moment $M_{420}$", r"$M_{420}$"),
    }
    shown = ("rho", "velocity_x", "theta", "stress_xx", "heat_flux_x", "M400", "M420")
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4), constrained_layout=True)
    for axis, name in zip(axes.flat[:7], shown):
        axis.plot(x, profiles["refined"][name], color="black", lw=2.0, label="refined DVM")
        axis.plot(x, profiles["coarse"][name], color="#277da1", lw=1.5, ls="--", label="coarse DVM")
        axis.plot(x, profiles["adaptive"][name], color="#d1495b", lw=1.5, ls="-.", label="adaptive")
        axis.set_title(labels[name][0])
        axis.set_xlabel(r"$x/\lambda_1$")
        axis.set_ylabel(labels[name][1])
        axis.grid(alpha=0.18, linewidth=0.6)
    axes[0, 0].legend(frameon=False, fontsize=8)
    extent = (x[0], x[-1], active_history.shape[0] * dt, 0.0)
    axes[1, 3].imshow(
        active_history.astype(float),
        aspect="auto",
        interpolation="nearest",
        cmap="magma_r",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )
    axes[1, 3].set(
        title="Causal kinetic support",
        xlabel=r"$x/\lambda_1$",
        ylabel=r"$t/\tau$",
    )
    fig.suptitle("Held-out Mach-2 normal shock — frozen Stage-30 lifecycle", fontsize=14)
    fig.savefig(output / "stage31_heldout_shock_profiles.png", dpi=190)
    plt.close(fig)


def _chatter_events(
    diagnostics: list[dict[str, object]], persistence_steps: int
) -> list[dict[str, int]]:
    activations = {
        (step, int(cell))
        for step, item in enumerate(diagnostics, start=1)
        for cell in item["activation_cells"]
    }
    events: list[dict[str, int]] = []
    for step, item in enumerate(diagnostics, start=1):
        for cell_value in item["release_cells"]:
            cell = int(cell_value)
            for later in range(step + 1, step + persistence_steps + 1):
                if (later, cell) in activations:
                    events.append(
                        {"release_step": step, "reactivation_step": later, "cell": cell}
                    )
                    break
    return events


def run(config: dict[str, object], output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    shock = normal_shock_rankine_hugoniot(float(config["mach"]))
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
    maximum_speed = max(
        float(np.max(np.abs(coarse_grid.centers()[:, 0]))),
        float(np.max(np.abs(refined_grid.centers()[:, 0]))),
    )
    dt = float(config["cfl"]) * xgrid.width / maximum_speed
    tau = float(config["tau"])
    prandtl = float(config["prandtl"])
    steps = int(config["steps"])

    coarse, coarse_left, coarse_right = initialize_normal_shock_dvm(
        xgrid, coarse_grid, shock
    )
    refined, refined_left, refined_right = initialize_normal_shock_dvm(
        xgrid, refined_grid, shock
    )
    adaptive, adaptive_left, adaptive_right = initialize_adaptive_normal_shock(
        xgrid,
        coarse_grid,
        shock,
        initial_active_half_width=int(config["initial_active_half_width"]),
    )
    # This immutable field is known at t=0.  It supplies the correct upstream
    # or downstream positive carrier for the target cell without consulting a
    # future state or reconstructing a tail from its 35 moments.
    cellwise_carriers = SpatialDVMState(xgrid, coarse_grid, coarse.masses.copy())

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
            coarse, dt, tau, coarse_left, coarse_right, prandtl=prandtl
        )
        timing["coarse_dvm"] += time.perf_counter() - started

        started = time.perf_counter()
        refined, refined_diag = full_dvm_shock_step(
            refined, dt, tau, refined_left, refined_right, prandtl=prandtl
        )
        timing["refined_dvm"] += time.perf_counter() - started

        started = time.perf_counter()
        adaptive, adaptive_diag = adaptive_shock_step(
            adaptive,
            dt,
            tau,
            adaptive_left,
            adaptive_right,
            prandtl=prandtl,
            sensor_interval_steps=int(config["sensor_interval_steps"]),
            release_sensor_interval_steps=int(config["release_sensor_interval_steps"]),
            macro_equilibrium_tolerance=float(config["macro_equilibrium_tolerance"]),
            birth_carrier=cellwise_carriers,
            kinetic_front_on=float(config["kinetic_front_on"]),
            causal_activation_candidates_only=bool(
                config["causal_activation_candidates_only"]
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
                _jsonable(
                    {
                        "event": "STAGE31_PROGRESS",
                        "step": step,
                        "steps": steps,
                        "active_cells": int(np.sum(adaptive.active)),
                        "activation_cells": adaptive_diag.activation_cells,
                        "release_cells": adaptive_diag.release_cells,
                    }
                )
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
    final_profiles = {
        "coarse": _profiles(coarse_array[-1], coarse_tail_array[-1]),
        "refined": _profiles(refined_array[-1], refined_tail_array[-1]),
        "adaptive": _profiles(adaptive_array[-1], adaptive_tail_array[-1]),
    }
    full_mask = np.ones(xgrid.cells, dtype=bool)
    core_mask = np.abs(xgrid.centers) <= float(config["comparison_half_width"])
    full_errors = {
        "coarse_vs_refined": _profile_errors(
            final_profiles["coarse"], final_profiles["refined"], full_mask
        ),
        "adaptive_vs_refined": _profile_errors(
            final_profiles["adaptive"], final_profiles["refined"], full_mask
        ),
    }
    core_errors = {
        "coarse_vs_refined": _profile_errors(
            final_profiles["coarse"], final_profiles["refined"], core_mask
        ),
        "adaptive_vs_refined": _profile_errors(
            final_profiles["adaptive"], final_profiles["refined"], core_mask
        ),
    }
    space_time_errors = {
        "adaptive_vs_refined_M400_percent": 100.0
        * _relative_error(adaptive_array[:, :, M400], refined_array[:, :, M400]),
        "adaptive_vs_refined_M420_percent": 100.0
        * _relative_error(adaptive_tail_array, refined_tail_array),
    }

    causal_pass = _causal_activation_audit(active_history, adaptive_diagnostics)
    births = [
        {
            "step": step,
            "cell": int(cell),
            "source": source,
            "donor_cell": donor,
            "carrier_used": bool(carrier),
            "reason": reason,
            "front_signal": signal,
        }
        for step, item in enumerate(adaptive_diagnostics, start=1)
        for cell, source, donor, carrier, reason, signal in zip(
            item["activation_cells"],
            item["activation_sources"],
            item["activation_donor_cells"],
            item["activation_carrier_used"],
            item["activation_reasons"],
            item["activation_front_signals"],
        )
    ]
    front_births = [item for item in births if "kinetic_front" in item["reason"]]
    carrier_provenance_pass = bool(
        front_births
        and all(item["source"] in ("left_neighbor", "right_neighbor") for item in front_births)
        and all(item["carrier_used"] for item in front_births)
        and all(
            item["front_signal"] is not None
            and float(item["front_signal"]) >= float(config["kinetic_front_on"])
            for item in front_births
        )
    )
    chatter = _chatter_events(
        adaptive_diagnostics, int(config["release_persistence_steps"])
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
    minimum_dvm_mass = min(float(np.min(coarse.masses)), float(np.min(refined.masses)))
    minimum_active_mass = min(
        float(item["minimum_micro_mass"])
        for item in adaptive_diagnostics
        if float(item["active_fraction"]) > 0.0
    )
    minimum_margin = min(
        _minimum_margin(coarse_history),
        _minimum_margin(refined_history),
        _minimum_margin(adaptive_history),
    )
    expensive_sensors = int(
        sum(
            int(item["activation_sensor_evaluations"])
            + int(item["release_sensor_evaluations"])
            for item in adaptive_diagnostics
        )
    )
    no_donor_skips = int(
        sum(int(item["activation_sensor_skips_no_donor"]) for item in adaptive_diagnostics)
    )
    wall_ratio = timing["adaptive"] / max(timing["coarse_dvm"], 1.0e-30)
    error_limit = float(config["profile_error_limit_percent"])
    reference_refinement_pass = bool(
        all(value < error_limit for value in full_errors["coarse_vs_refined"].values())
    )
    profile_accuracy_pass = bool(
        all(value < error_limit for value in full_errors["adaptive_vs_refined"].values())
        and all(value < error_limit for value in core_errors["adaptive_vs_refined"].values())
        and all(value < error_limit for value in space_time_errors.values())
    )
    invariants_pass = bool(
        minimum_dvm_mass > 0.0
        and minimum_active_mass > 0.0
        and minimum_margin >= -2.0e-12
        and maximum_balance < 2.0e-8
        and maximum_projection < 2.0e-9
        and maximum_collision_drift < 2.0e-9
        and maximum_sync < 2.0e-8
    )
    lifecycle_pass = bool(causal_pass and carrier_provenance_pass and not chatter)
    localization_pass = bool(
        float(np.mean(active_array)) < 0.50
        and float(np.max(np.mean(active_array, axis=1))) < 0.50
    )
    performance_pass = wall_ratio < 1.0
    all_pass = bool(
        reference_refinement_pass
        and profile_accuracy_pass
        and invariants_pass
        and lifecycle_pass
        and localization_pass
        and performance_pass
    )
    decision = (
        "WORKSTATION_PASS" if config["mode"] == "workstation" and all_pass
        else "WORKSTATION_HOLD" if config["mode"] == "workstation"
        else "SMOKE_PASS" if invariants_pass and lifecycle_pass
        else "SMOKE_HOLD"
    )

    contracts = {
        "held_out_mach_not_stage25a_tuning_case_pass": float(config["mach"]) != 3.0,
        "coarse_refined_profile_agreement_pass": reference_refinement_pass,
        "adaptive_physical_profile_accuracy_pass": profile_accuracy_pass,
        "synchronous_causal_cellwise_carrier_pass": lifecycle_pass,
        "positivity_conservation_realizability_pass": invariants_pass,
        "kinetic_fraction_below_50_percent_pass": localization_pass,
        "measured_speedup_vs_coarse_full_dvm_pass": performance_pass,
        "all_qualification_gates_pass": all_pass,
        "minimum_dvm_mass": minimum_dvm_mass,
        "minimum_adaptive_active_mass": minimum_active_mass,
        "minimum_realizability_margin": minimum_margin,
        "maximum_balance_residual": maximum_balance,
        "maximum_projection_residual": maximum_projection,
        "maximum_collision_invariant_drift": maximum_collision_drift,
        "maximum_micro_macro_sync_residual": maximum_sync,
    }
    metrics = {
        "causal_births": len(births),
        "kinetic_front_births": len(front_births),
        "releases": int(sum(int(item["releases"]) for item in adaptive_diagnostics)),
        "four_step_chatter_events": len(chatter),
        "mean_active_fraction_percent": 100.0 * float(np.mean(active_array)),
        "peak_active_fraction_percent": 100.0 * float(
            np.max(np.mean(active_array, axis=1))
        ),
        "final_active_fraction_percent": 100.0 * float(np.mean(active_array[-1])),
        "expensive_sensor_evaluations": expensive_sensors,
        "activation_sensor_skips_no_donor": no_donor_skips,
        "adaptive_over_coarse_dvm_wall_time": wall_ratio,
        "measured_speedup_factor": 1.0 / max(wall_ratio, 1.0e-30),
        "full_domain_profile_errors_percent": full_errors,
        "shock_core_profile_errors_percent": core_errors,
        "space_time_errors_percent": space_time_errors,
    }
    summary = {
        "stage": "31",
        "case": "held_out_normal_shock_ma2",
        "decision": decision,
        "scope": config["scope"],
        "configuration": {**config, "dt": dt, "final_time": steps * dt},
        "rankine_hugoniot": asdict(shock),
        "reference": "independently evolved positive guided coarse/refined Full-DVM",
        "primary_metrics": metrics,
        "contracts": contracts,
        "birth_history": births,
        "chatter_events": chatter,
        "timing_seconds": {**timing, "total": float(sum(timing.values()))},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    with (output / "stage31_heldout_shock_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_jsonable(summary), stream, indent=2, allow_nan=False)
    np.savez_compressed(
        output / "stage31_heldout_shock_profiles.npz",
        x=xgrid.centers.astype(np.float32),
        dt=np.asarray(dt, dtype=np.float32),
        coarse_final_moments=coarse_array[-1].astype(np.float32),
        refined_final_moments=refined_array[-1].astype(np.float32),
        adaptive_final_moments=adaptive_array[-1].astype(np.float32),
        coarse_final_M420=coarse_tail_array[-1].astype(np.float32),
        refined_final_M420=refined_tail_array[-1].astype(np.float32),
        adaptive_final_M420=adaptive_tail_array[-1].astype(np.float32),
        adaptive_active=active_array,
    )
    with (output / "stage31_heldout_shock_profiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["x"]
            + [f"{method}_{name}" for method in ("refined", "coarse", "adaptive") for name in PROFILE_NAMES]
        )
        for cell, coordinate in enumerate(xgrid.centers):
            writer.writerow(
                [coordinate]
                + [
                    final_profiles[method][name][cell]
                    for method in ("refined", "coarse", "adaptive")
                    for name in PROFILE_NAMES
                ]
            )
    _plot(output, xgrid.centers, final_profiles, active_array, dt)

    adaptive_errors = full_errors["adaptive_vs_refined"]
    report = [
        "# Stage 31 held-out Mach-2 normal-shock result",
        "",
        f"- Decision: **{decision}**",
        f"- Held-out case: Mach {shock.mach:g} (Stage 25A used Mach 3)",
        f"- Steps / final time: {steps} / {steps * dt:.6f} tau",
        f"- Causal births / front births / releases: {len(births)} / {len(front_births)} / {metrics['releases']}",
        f"- Mean/peak/final kinetic fraction: {metrics['mean_active_fraction_percent']:.3f}% / {metrics['peak_active_fraction_percent']:.3f}% / {metrics['final_active_fraction_percent']:.3f}%",
        f"- Full-profile adaptive errors rho/u/T: {adaptive_errors['rho']:.4f}% / {adaptive_errors['velocity_x']:.4f}% / {adaptive_errors['theta']:.4f}%",
        f"- Full-profile adaptive errors stress/heat flux: {adaptive_errors['stress_xx']:.4f}% / {adaptive_errors['heat_flux_x']:.4f}%",
        f"- Full-profile adaptive errors M400/M420: {adaptive_errors['M400']:.4f}% / {adaptive_errors['M420']:.4f}%",
        f"- Space-time adaptive M400/M420 errors: {space_time_errors['adaptive_vs_refined_M400_percent']:.4f}% / {space_time_errors['adaptive_vs_refined_M420_percent']:.4f}%",
        f"- Adaptive/coarse-DVM wall-time ratio: {wall_ratio:.3f}x ({metrics['measured_speedup_factor']:.3f}x speedup)",
        f"- Maximum balance / micro-macro sync residual: {maximum_balance:.3e} / {maximum_sync:.3e}",
        "",
        "This is a held-out numerical cross-case test of the implemented cubic",
        "Fokker-Planck model. It is not independent DSMC or experimental",
        "validation of the collision operator's physical fidelity.",
    ]
    (output / "STAGE31_HELDOUT_SHOCK_RESULT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = arguments()
    config = configuration(args.mode, args.steps)
    output = args.output
    if output is None:
        output = REPOSITORY_ROOT / "results" / "riemann35_stage31" / args.mode
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

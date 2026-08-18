#!/usr/bin/env python3
"""Run the Stage-27 lightweight spatial causal-donor qualification."""

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
    macro_shock_step,
    mixture_of_gaussians_moments_35,
    realizability_margin_35,
    reconstruct_gaussian_mixture_quadrature,
)
from hyqmom_fp.dvm_reference import (  # noqa: E402
    initialize_diagonal_gaussian_mixture,
)


POSITION = {index: offset for offset, index in enumerate(HYQMOM_35_INDICES)}
M400 = POSITION[(4, 0, 0)]
M420_INDEX = (4, 2, 0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "workstation"), default="workstation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    return parser.parse_args()


def configuration(mode: str, steps: int | None = None) -> dict[str, object]:
    values: dict[str, object] = {
        "mode": mode,
        "x_lower": -1.5,
        "x_upper": 1.5,
        "spatial_cells": 9 if mode == "smoke" else 12,
        "velocity_lower": (-3.0, -2.5, -1.5),
        "velocity_upper": (3.0, 2.5, 1.5),
        "coarse_velocity_shape": (15, 13, 9),
        "refined_velocity_shape": (17, 15, 11),
        "cfl": 0.18,
        "tau": 1.0,
        "prandtl": 2.0 / 3.0,
        "steps": 4 if mode == "smoke" else 20,
    }
    if steps is not None:
        if steps < 1:
            raise ValueError("steps must be positive")
        values["steps"] = steps
    return values


def crossing_components() -> tuple[list[tuple], list[tuple]]:
    """Frozen unequal non-equilibrium states on the two sides."""

    left = [
        (0.72, (1.20, 0.55, 0.0), np.diag([0.08, 0.06, 0.05])),
        (0.28, (-0.25, -1.00, 0.0), np.diag([0.05, 0.07, 0.05])),
    ]
    right = [
        (0.62, (-1.10, -0.45, 0.0), np.diag([0.07, 0.06, 0.05])),
        (0.38, (0.35, 0.85, 0.0), np.diag([0.06, 0.07, 0.05])),
    ]
    return left, right


def _initialize_full(
    xgrid: SpatialGrid1D,
    vgrid: DVMGrid,
) -> tuple[SpatialDVMState, DVMState, DVMState]:
    left_components, right_components = crossing_components()
    left, _ = initialize_diagonal_gaussian_mixture(
        vgrid, left_components, match_exact_moments=True
    )
    right, _ = initialize_diagonal_gaussian_mixture(
        vgrid, right_components, match_exact_moments=True
    )
    left_mask = xgrid.centers < 0.0
    masses = np.where(left_mask[:, None], left.masses[None, :], right.masses[None, :])
    return SpatialDVMState(xgrid, vgrid, masses), left, right


def _initialize_adaptive(full: SpatialDVMState) -> AdaptiveSpatialState:
    nx = full.spatial_grid.cells
    active = np.zeros(nx, dtype=bool)
    if nx % 2:
        active[nx // 2] = True
    else:
        active[nx // 2 - 1 : nx // 2 + 1] = True
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
        cells = item["activation_cells"]
        sources = item["activation_sources"]
        donors = item["activation_donor_cells"]
        for cell, source, donor in zip(cells, sources, donors):
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


def _plot(
    output: Path,
    x: np.ndarray,
    dt: float,
    refined_moments: np.ndarray,
    coarse_moments: np.ndarray,
    adaptive_moments: np.ndarray,
    macro_moments: np.ndarray,
    refined_tail: np.ndarray,
    coarse_tail: np.ndarray,
    adaptive_tail: np.ndarray,
    macro_tail: np.ndarray,
    active_history: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(x, refined_moments[-1, :, M400], "k-", label="refined DVM")
    axes[0, 0].plot(x, coarse_moments[-1, :, M400], "C0--", label="coarse DVM")
    axes[0, 0].plot(x, adaptive_moments[-1, :, M400], "C2-.", label="adaptive")
    axes[0, 0].plot(x, macro_moments[-1, :, M400], "C3:", label="macro")
    axes[0, 0].set(title="Final retained M400", xlabel="x", ylabel="M400")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, refined_tail[-1], "k-", label="refined DVM")
    axes[0, 1].plot(x, coarse_tail[-1], "C0--", label="coarse DVM")
    axes[0, 1].plot(x, adaptive_tail[-1], "C2-.", label="adaptive")
    axes[0, 1].plot(x, macro_tail[-1], "C3:", label="macro")
    axes[0, 1].set(title="Final predictive M420", xlabel="x", ylabel="M420")

    extent = (x[0], x[-1], active_history.shape[0] * dt, 0.0)
    axes[1, 0].imshow(
        active_history.astype(float), aspect="auto", interpolation="nearest",
        cmap="Greys", vmin=0.0, vmax=1.0, extent=extent,
    )
    axes[1, 0].set(title="Causal kinetic-memory wavefront", xlabel="x", ylabel="time")

    times = np.arange(active_history.shape[0]) * dt
    axes[1, 1].semilogy(
        times,
        [100.0 * _relative_error(item, ref) for item, ref in zip(adaptive_tail, refined_tail)],
        "C2-",
        label="adaptive",
    )
    axes[1, 1].semilogy(
        times,
        [100.0 * _relative_error(item, ref) for item, ref in zip(macro_tail, refined_tail)],
        "C3--",
        label="macro",
    )
    axes[1, 1].set(title="M420 profile error", xlabel="time", ylabel="relative L2 [%]")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Stage 27 spatial causal-donor audit")
    fig.savefig(output / "stage27_crossing_donor.png", dpi=180)
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

    coarse, coarse_left, coarse_right = _initialize_full(xgrid, coarse_grid)
    refined, refined_left, refined_right = _initialize_full(xgrid, refined_grid)
    adaptive = _initialize_adaptive(coarse)
    macro = coarse.moments()
    left_components, right_components = crossing_components()
    left_moments = mixture_of_gaussians_moments_35(left_components)
    right_moments = mixture_of_gaussians_moments_35(right_components)

    coarse_history = [coarse.moments()]
    refined_history = [refined.moments()]
    adaptive_history = [adaptive.moments.copy()]
    macro_history = [macro.copy()]
    coarse_tail_history = [_dvm_tail(coarse)]
    refined_tail_history = [_dvm_tail(refined)]
    adaptive_tail_history = [_adaptive_tail(adaptive)]
    macro_tail_history = [_algebraic_tail(macro)]
    active_history = [adaptive.active.copy()]
    coarse_diagnostics: list[dict[str, object]] = []
    refined_diagnostics: list[dict[str, object]] = []
    adaptive_diagnostics: list[dict[str, object]] = []
    macro_diagnostics: list[dict[str, object]] = []
    timing = {"coarse_dvm": 0.0, "refined_dvm": 0.0, "adaptive": 0.0, "macro": 0.0}

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
            adaptive, dt, tau, coarse_left, coarse_right, prandtl=prandtl
        )
        timing["adaptive"] += time.perf_counter() - started

        started = time.perf_counter()
        macro, macro_diag = macro_shock_step(
            macro, xgrid, dt, tau, left_moments, right_moments, prandtl=prandtl
        )
        timing["macro"] += time.perf_counter() - started

        coarse_history.append(coarse.moments())
        refined_history.append(refined.moments())
        adaptive_history.append(adaptive.moments.copy())
        macro_history.append(macro.copy())
        coarse_tail_history.append(_dvm_tail(coarse))
        refined_tail_history.append(_dvm_tail(refined))
        adaptive_tail_history.append(_adaptive_tail(adaptive))
        macro_tail_history.append(_algebraic_tail(macro))
        active_history.append(adaptive.active.copy())
        coarse_diagnostics.append(asdict(coarse_diag))
        refined_diagnostics.append(asdict(refined_diag))
        adaptive_diagnostics.append(asdict(adaptive_diag))
        macro_diagnostics.append(asdict(macro_diag))
        print(
            json.dumps(
                {
                    "event": "STAGE27_PROGRESS",
                    "step": step,
                    "steps": steps,
                    "active_cells": int(np.sum(adaptive.active)),
                    "activations": adaptive_diag.activations,
                    "blocked_safe_births": adaptive_diag.blocked_births,
                }
            ),
            flush=True,
        )

    coarse_history_array = np.asarray(coarse_history)
    refined_history_array = np.asarray(refined_history)
    adaptive_history_array = np.asarray(adaptive_history)
    macro_history_array = np.asarray(macro_history)
    coarse_tail_array = np.asarray(coarse_tail_history)
    refined_tail_array = np.asarray(refined_tail_history)
    adaptive_tail_array = np.asarray(adaptive_tail_history)
    macro_tail_array = np.asarray(macro_tail_history)
    active_history_array = np.asarray(active_history)

    fully_active = np.flatnonzero(np.all(active_history_array, axis=1))
    full_activation_step = int(fully_active[0]) if fully_active.size else None
    post_start = full_activation_step if full_activation_step is not None else steps + 1
    final_errors = {
        "coarse_vs_refined_M400_percent": 100.0 * _relative_error(
            coarse_history_array[-1, :, M400], refined_history_array[-1, :, M400]
        ),
        "coarse_vs_refined_M420_percent": 100.0 * _relative_error(
            coarse_tail_array[-1], refined_tail_array[-1]
        ),
        "adaptive_vs_refined_M400_percent": 100.0 * _relative_error(
            adaptive_history_array[-1, :, M400], refined_history_array[-1, :, M400]
        ),
        "adaptive_vs_refined_M420_percent": 100.0 * _relative_error(
            adaptive_tail_array[-1], refined_tail_array[-1]
        ),
        "macro_vs_refined_M400_percent": 100.0 * _relative_error(
            macro_history_array[-1, :, M400], refined_history_array[-1, :, M400]
        ),
        "macro_vs_refined_M420_percent": 100.0 * _relative_error(
            macro_tail_array[-1], refined_tail_array[-1]
        ),
    }
    adaptive_m420_error_by_step = np.asarray(
        [100.0 * _relative_error(value, reference)
         for value, reference in zip(adaptive_tail_array, refined_tail_array)]
    )
    post_activation_errors = {
        "adaptive_vs_refined_M400_percent": (
            100.0
            * _relative_error(
                adaptive_history_array[post_start:, :, M400],
                refined_history_array[post_start:, :, M400],
            )
            if post_start <= steps
            else float("inf")
        ),
        "adaptive_vs_refined_M420_percent": (
            100.0
            * _relative_error(
                adaptive_tail_array[post_start:], refined_tail_array[post_start:]
            )
            if post_start <= steps
            else float("inf")
        ),
    }

    def maximum_balance(items: Sequence[dict[str, object]]) -> float:
        return float(
            max(
                max(
                    float(
                        (item["transport"] if "transport" in item else item)[key]
                    )
                    for key in (
                        "mass_balance_residual",
                        "momentum_balance_residual",
                        "energy_balance_residual",
                    )
                )
                for item in items
            )
        )

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
    maximum_balance_residual = max(
        maximum_balance(coarse_diagnostics),
        maximum_balance(refined_diagnostics),
        maximum_balance(adaptive_diagnostics),
        maximum_balance(macro_diagnostics),
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
        _minimum_margin(macro_history),
    )
    blocked_total = int(sum(int(item["blocked_births"]) for item in adaptive_diagnostics))
    contracts = {
        "synchronous_causal_provenance_pass": causal_pass,
        "full_activation_reached": full_activation_step is not None,
        "full_activation_step": full_activation_step,
        "blocked_safe_birth_attempts_before_donor": blocked_total,
        "final_step_blocked_births": int(adaptive_diagnostics[-1]["blocked_births"]),
        "minimum_dvm_mass": minimum_dvm_mass,
        "minimum_adaptive_active_mass": minimum_active_mass,
        "minimum_realizability_margin": minimum_margin,
        "maximum_balance_residual": maximum_balance_residual,
        "maximum_projection_residual": maximum_projection,
        "maximum_collision_invariant_drift": maximum_collision_drift,
        "maximum_micro_macro_sync_residual": maximum_sync,
    }
    contracts_pass = bool(
        causal_pass
        and full_activation_step is not None
        and int(adaptive_diagnostics[-1]["blocked_births"]) == 0
        and minimum_dvm_mass > 0.0
        and minimum_active_mass > 0.0
        and minimum_margin >= -2.0e-12
        and maximum_balance_residual < 2.0e-8
        and maximum_projection < 2.0e-9
        and maximum_collision_drift < 2.0e-9
        and maximum_sync < 2.0e-8
        and final_errors["coarse_vs_refined_M400_percent"] < 3.0
        and final_errors["coarse_vs_refined_M420_percent"] < 3.0
        and final_errors["adaptive_vs_refined_M400_percent"] < 3.0
        and final_errors["adaptive_vs_refined_M420_percent"] < 3.0
        and post_activation_errors["adaptive_vs_refined_M400_percent"] < 3.0
        and post_activation_errors["adaptive_vs_refined_M420_percent"] < 3.0
    )
    decision = "WORKSTATION_PASS" if contracts_pass else "WORKSTATION_HOLD"

    summary = {
        "stage": "27",
        "case": "two_sided_crossing_populations",
        "decision": decision,
        "scope": "workstation numerical qualification, not MD/DSMC physical validation",
        "configuration": {**config, "dt": dt, "final_time": steps * dt},
        "reference": "positive guided Scharfetter-Gummel DVM on coarse and refined velocity grids",
        "primary_metrics": {
            "initial_adaptive_M420_error_percent": float(adaptive_m420_error_by_step[0]),
            "peak_predonor_adaptive_M420_error_percent": float(
                np.max(adaptive_m420_error_by_step[: max(post_start, 1)])
            ),
            "mean_active_fraction_percent": 100.0 * float(np.mean(active_history_array)),
            "final_active_fraction_percent": 100.0 * float(np.mean(active_history_array[-1])),
            "adaptive_over_coarse_dvm_wall_time": (
                timing["adaptive"] / max(timing["coarse_dvm"], 1.0e-30)
            ),
            "final_errors": final_errors,
            "post_activation_space_time_errors": post_activation_errors,
        },
        "contracts": {**contracts, "all_pass": contracts_pass},
        "activation_history": [
            {
                "step": step,
                "active_cells_before": np.flatnonzero(active_history_array[step - 1]),
                "active_cells_after": np.flatnonzero(active_history_array[step]),
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
            "M420 before donor arrival is intentionally ambiguous; the adaptive path "
            "uses an algebraic value only for diagnostics and does not retain it as memory."
        ),
    }

    np.savez_compressed(
        output / "stage27_crossing_donor_histories.npz",
        x=xgrid.centers,
        dt=np.asarray(dt),
        refined_moments=refined_history_array,
        coarse_moments=coarse_history_array,
        adaptive_moments=adaptive_history_array,
        macro_moments=macro_history_array,
        refined_M420=refined_tail_array,
        coarse_M420=coarse_tail_array,
        adaptive_M420=adaptive_tail_array,
        macro_M420=macro_tail_array,
        adaptive_active=active_history_array,
    )
    with (output / "stage27_crossing_donor_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(summary), stream, indent=2, allow_nan=False)
    _plot(
        output,
        xgrid.centers,
        dt,
        refined_history_array,
        coarse_history_array,
        adaptive_history_array,
        macro_history_array,
        refined_tail_array,
        coarse_tail_array,
        adaptive_tail_array,
        macro_tail_array,
        active_history_array,
    )
    report = [
        "# Stage 27 spatial causal-donor result",
        "",
        f"- Decision: **{decision}**",
        f"- Full causal activation reached at step: {full_activation_step}",
        f"- Safe blocked attempts before donor arrival: {blocked_total}",
        f"- Initial adaptive M420 ambiguity/error: {adaptive_m420_error_by_step[0]:.6f}%",
        f"- Final adaptive M400 error vs refined DVM: {final_errors['adaptive_vs_refined_M400_percent']:.6f}%",
        f"- Final adaptive M420 error vs refined DVM: {final_errors['adaptive_vs_refined_M420_percent']:.6f}%",
        f"- Post-activation space-time M400 error: {post_activation_errors['adaptive_vs_refined_M400_percent']:.6f}%",
        f"- Post-activation space-time M420 error: {post_activation_errors['adaptive_vs_refined_M420_percent']:.6f}%",
        f"- Coarse/refined final M420 difference: {final_errors['coarse_vs_refined_M420_percent']:.6f}%",
        f"- Adaptive/coarse-DVM wall-time ratio: {timing['adaptive']/max(timing['coarse_dvm'],1.0e-30):.3f}x",
        f"- Minimum positive DVM mass: {minimum_dvm_mass:.3e}",
        f"- Maximum finite-volume balance residual: {maximum_balance_residual:.3e}",
        f"- Maximum micro/macro sync residual: {maximum_sync:.3e}",
        f"- Workstation wall time: {sum(timing.values()):.3f} s",
        "",
        "Blocked attempts are expected and safe: an alarm is denied until a causal donor",
        "arrives.  A just-born cell cannot relay memory again during the same step.",
        "This strongly non-equilibrium case becomes 100% kinetic, so the current",
        "unoptimized adaptive path provides accuracy but no wall-time saving.",
        "The result qualifies this implementation numerically; it is not independent",
        "MD/DSMC evidence for physical fidelity.",
    ]
    (output / "STAGE27_CROSSING_DONOR_RESULT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = arguments()
    config = configuration(args.mode, args.steps)
    output = args.output
    if output is None:
        output = REPOSITORY_ROOT / "results" / "riemann35_stage27" / args.mode
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

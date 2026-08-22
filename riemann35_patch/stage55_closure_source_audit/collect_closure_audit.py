#!/usr/bin/env python3
"""Collect, diagnose, plot, and bundle the Stage-55 source audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage55")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    HYQMOM_35_INDICES,
    coefficients_from_moments,
    projected_fp_collision_source,
    reconstruct_gaussian_mixture_quadrature,
    reconstruct_two_population_quadrature,
)
from hyqmom_fp.grad_hyqmom import WeightedNodeTailClosure  # noqa: E402
from riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux import (  # noqa: E402
    central_third_components,
    irreducible_decomposition,
    normalized_component_rmse,
    relative_history_error,
    replicate_spread,
    symmetric_tensor,
)
from riemann35_patch.stage55_closure_source_audit.run_closure_method import (  # noqa: E402
    METHODS,
    TAIL_INDICES,
    central_source_components,
)


REFERENCE = "qmc_reference"
SELECTED = "projected_tail_time_refined"
THIRD_INDICES = tuple(index for index in HYQMOM_35_INDICES if sum(index) == 3)
THIRD_LABELS = tuple(rf"$T_{{{''.join(str(value) for value in index)}}}$" for index in THIRD_INDICES)
METHOD_LABELS = {
    "qmc_reference": "positive Full-FP QMC ±2 SEM",
    "gaussian_hyqmom35": "HyQMOM-35 finite mixture",
    "dynamic_unprojected": "unprojected 35+49 memory",
    "projected_tail_base": "projected 35+49 (base)",
    "projected_tail_node_refined": "projected 35+49 (node-refined)",
    "projected_tail_time_refined": "projected 35+49 (time-refined)",
}
COLORS = {
    "qmc_reference": "#111111",
    "gaussian_hyqmom35": "#c85a17",
    "dynamic_unprojected": "#9b6aa0",
    "projected_tail_base": "#729fcf",
    "projected_tail_node_refined": "#3465a4",
    "projected_tail_time_refined": "#005a8d",
}
STYLES = {
    "qmc_reference": "-",
    "gaussian_hyqmom35": "-.",
    "dynamic_unprojected": ":",
    "projected_tail_base": "--",
    "projected_tail_node_refined": (0, (5, 2)),
    "projected_tail_time_refined": "-",
}
PLOT_METHODS = (
    REFERENCE,
    "gaussian_hyqmom35",
    "projected_tail_node_refined",
    SELECTED,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-spread-gate", type=float, default=0.03)
    parser.add_argument("--refinement-gate", type=float, default=0.03)
    parser.add_argument("--third-gate", type=float, default=0.03)
    parser.add_argument("--tracefree-gate", type=float, default=0.05)
    parser.add_argument("--component-gate", type=float, default=0.03)
    parser.add_argument("--invariant-gate", type=float, default=2.0e-8)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    return parser.parse_args()


def load_method(root: Path, method: str) -> dict[str, object] | None:
    archive_path = root / f"stage55_{method}.npz"
    summary_path = root / f"stage55_{method}_summary.json"
    if not archive_path.is_file() or not summary_path.is_file():
        return None
    archive = np.load(archive_path)
    return {
        "times": np.asarray(archive["times"], dtype=float),
        "histories": np.asarray(archive["histories"], dtype=float),
        "sources": np.asarray(archive["sources"], dtype=float),
        "tails": np.asarray(archive["tails"], dtype=float),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def _raw_tail(closure: WeightedNodeTailClosure, moments: np.ndarray) -> np.ndarray:
    return np.asarray([closure(index, moments) for index in TAIL_INDICES], dtype=float)


def _closure_snapshot(
    moments: np.ndarray,
    *,
    kind: str,
    tau: float,
    prandtl: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if kind == "gaussian":
        quadrature = reconstruct_gaussian_mixture_quadrature(moments)
        diagnostics = {
            "minimum_weight": float(np.min(quadrature.weights)),
            "moment_residual": float(quadrature.relative_moment_residual),
            "negative_mass_fraction": 0.0,
        }
    elif kind == "two_population":
        quadrature = reconstruct_two_population_quadrature(
            moments,
            quadrature_nodes=5,
            minimum_skewness_norm=0.05,
            residual_correction=False,
        )
        diagnostics = {
            "minimum_weight": float(np.min(quadrature.weights)),
            "moment_residual": float(quadrature.base_relative_moment_residual),
            "negative_mass_fraction": float(quadrature.negative_mass_fraction),
        }
    else:  # pragma: no cover - protected by caller
        raise ValueError(f"unknown closure kind {kind}")
    closure = WeightedNodeTailClosure(quadrature.nodes, quadrature.weights, maximum_order=6)
    coefficients = coefficients_from_moments(
        moments,
        tau=tau,
        prandtl=prandtl,
        closure=closure,
    )
    source = projected_fp_collision_source(moments, coefficients, closure=closure)
    return source, _raw_tail(closure, moments), diagnostics


def source_closure_audit(
    reference: dict[str, object], *, tau: float, prandtl: float
) -> dict[str, object]:
    histories = np.asarray(reference["histories"])
    direct_sources = np.asarray(reference["sources"])
    direct_tails = np.asarray(reference["tails"])
    if direct_sources.shape[:2] != histories.shape[:2] or direct_tails.shape[:2] != histories.shape[:2]:
        raise RuntimeError("QMC source/tail snapshots do not align with moment histories")
    output: dict[str, object] = {}
    for kind in ("gaussian", "two_population"):
        closure_sources = np.empty_like(direct_sources)
        closure_tails = np.empty_like(direct_tails)
        minimum_weight = float("inf")
        maximum_moment_residual = 0.0
        maximum_negative_mass_fraction = 0.0
        for replicate in range(histories.shape[0]):
            for sample in range(histories.shape[1]):
                source, tail, diagnostics = _closure_snapshot(
                    histories[replicate, sample],
                    kind=kind,
                    tau=tau,
                    prandtl=prandtl,
                )
                closure_sources[replicate, sample] = source
                closure_tails[replicate, sample] = tail
                minimum_weight = min(minimum_weight, diagnostics["minimum_weight"])
                maximum_moment_residual = max(
                    maximum_moment_residual, diagnostics["moment_residual"]
                )
                maximum_negative_mass_fraction = max(
                    maximum_negative_mass_fraction,
                    diagnostics["negative_mass_fraction"],
                )
        direct_central = np.asarray(
            [
                [central_source_components(moments, source) for moments, source in zip(rep_moments, rep_sources)]
                for rep_moments, rep_sources in zip(histories, direct_sources)
            ]
        )
        closure_central = np.asarray(
            [
                [central_source_components(moments, source) for moments, source in zip(rep_moments, rep_sources)]
                for rep_moments, rep_sources in zip(histories, closure_sources)
            ]
        )
        source_scale = np.maximum(np.linalg.norm(direct_central, axis=-1), 1.0e-14)
        tail_scale = np.maximum(np.linalg.norm(direct_tails, axis=-1), 1.0e-14)
        source_error_time = np.mean(
            np.linalg.norm(closure_central - direct_central, axis=-1) / source_scale,
            axis=0,
        )
        tail_error_time = np.mean(
            np.linalg.norm(closure_tails - direct_tails, axis=-1) / tail_scale,
            axis=0,
        )
        output[kind] = {
            "third_source_relative_l2": float(
                np.linalg.norm(closure_central - direct_central)
                / max(np.linalg.norm(direct_central), 1.0e-14)
            ),
            "tail_M5_M6_relative_l2": float(
                np.linalg.norm(closure_tails - direct_tails)
                / max(np.linalg.norm(direct_tails), 1.0e-14)
            ),
            "third_source_relative_error_by_time": source_error_time,
            "tail_relative_error_by_time": tail_error_time,
            "minimum_weight": minimum_weight,
            "maximum_moment_residual": maximum_moment_residual,
            "maximum_negative_mass_fraction": maximum_negative_mass_fraction,
        }
    return output


def _derived(item: dict[str, object]) -> dict[str, np.ndarray]:
    components = np.asarray(
        [central_third_components(history) for history in item["histories"]]
    )
    heat_flux, carrying, trace_free = irreducible_decomposition(components)
    return {
        "components": components,
        "heat_flux": heat_flux,
        "carrying": carrying,
        "trace_free": trace_free,
        "full_tensor": symmetric_tensor(components),
    }


def _history_errors(
    loaded: dict[str, dict[str, object]], data: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, object]]:
    if REFERENCE not in loaded:
        return {}
    reference = {key: np.mean(value, axis=0) for key, value in data[REFERENCE].items()}
    errors = {}
    for method, item in data.items():
        if method == REFERENCE:
            continue
        if not np.allclose(loaded[method]["times"], loaded[REFERENCE]["times"], rtol=0.0, atol=2.0e-13):
            raise RuntimeError(f"sample times differ for {method}")
        candidate = {key: np.mean(value, axis=0) for key, value in item.items()}
        errors[method] = {
            "heat_flux": relative_history_error(candidate["heat_flux"], reference["heat_flux"]),
            "third_tensor": relative_history_error(candidate["full_tensor"], reference["full_tensor"]),
            "trace_free": relative_history_error(candidate["trace_free"], reference["trace_free"]),
            "component_normalized_rmse": normalized_component_rmse(
                candidate["components"], reference["components"]
            ),
        }
    return errors


def _cumulative_error(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = []
    for end in range(1, candidate.shape[0] + 1):
        values.append(
            np.linalg.norm(candidate[:end] - reference[:end])
            / max(np.linalg.norm(reference[:end]), 1.0e-14)
        )
    return np.asarray(values)


def _plot_components(
    path: Path,
    loaded: dict[str, dict[str, object]],
    data: dict[str, dict[str, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(5, 2, figsize=(12.2, 14.4), sharex=True)
    reference = data[REFERENCE]["components"]
    reference_mean = np.mean(reference, axis=0)
    reference_sem = np.std(reference, axis=0, ddof=1) / np.sqrt(reference.shape[0])
    for position, axis in enumerate(axes.ravel()):
        times = loaded[REFERENCE]["times"]
        axis.fill_between(
            times,
            reference_mean[:, position] - 2.0 * reference_sem[:, position],
            reference_mean[:, position] + 2.0 * reference_sem[:, position],
            color="0.80",
            alpha=0.55,
            linewidth=0.0,
        )
        for method in PLOT_METHODS:
            if method not in loaded:
                continue
            mean = np.mean(data[method]["components"], axis=0)
            axis.plot(
                loaded[method]["times"],
                mean[:, position],
                color=COLORS[method],
                linestyle=STYLES[method],
                linewidth=1.9 if method in (REFERENCE, SELECTED) else 1.55,
                label=METHOD_LABELS[method],
            )
        axis.set_title(THIRD_LABELS[position], fontsize=12, pad=5)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    for axis in axes[-1]:
        axis.set_xlabel(r"Time, $t/\tau$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Central moment")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.986))
    figure.suptitle("Stage 55: all ten third-order moments after source-tail projection", fontsize=16, y=0.999)
    figure.subplots_adjust(top=0.93, bottom=0.055, left=0.085, right=0.985, hspace=0.34, wspace=0.20)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_diagnostics(
    path: Path,
    loaded: dict[str, dict[str, object]],
    data: dict[str, dict[str, np.ndarray]],
    audit: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.4))
    times = loaded[REFERENCE]["times"]
    closure_colors = {"gaussian": "#c85a17", "two_population": "#2a9d68"}
    closure_labels = {
        "gaussian": "finite-mixture M5/M6",
        "two_population": "positive two-population M5/M6",
    }
    for kind in ("gaussian", "two_population"):
        axes[0, 0].plot(
            times,
            100.0 * np.asarray(audit[kind]["third_source_relative_error_by_time"]),
            color=closure_colors[kind],
            linewidth=1.9,
            label=closure_labels[kind],
        )
        axes[0, 1].plot(
            times,
            100.0 * np.asarray(audit[kind]["tail_relative_error_by_time"]),
            color=closure_colors[kind],
            linewidth=1.9,
            label=closure_labels[kind],
        )
    axes[0, 0].set_title("Instantaneous third-order source error")
    axes[0, 0].set_ylabel("Relative error (%)")
    axes[0, 1].set_title("M5/M6 tail reconstruction error")
    axes[0, 1].set_ylabel("Relative error (%)")

    reference_tensor = np.mean(data[REFERENCE]["full_tensor"], axis=0)
    for method in ("gaussian_hyqmom35", "projected_tail_node_refined", SELECTED):
        if method not in data:
            continue
        candidate = np.mean(data[method]["full_tensor"], axis=0)
        axes[1, 0].plot(
            times,
            100.0 * _cumulative_error(candidate, reference_tensor),
            color=COLORS[method],
            linestyle=STYLES[method],
            linewidth=1.9,
            label=METHOD_LABELS[method],
        )
    axes[1, 0].axhline(3.0, color="#8c3b32", linestyle="--", linewidth=1.2, label="3% objective")
    axes[1, 0].set_title("Accumulated full third-tensor error")
    axes[1, 0].set_ylabel("History error (%)")

    for method in (REFERENCE, "gaussian_hyqmom35", SELECTED):
        if method not in data:
            continue
        trace_free = np.mean(data[method]["trace_free"], axis=0)
        norm = np.linalg.norm(trace_free.reshape(trace_free.shape[0], -1), axis=1)
        axes[1, 1].plot(
            times,
            norm,
            color=COLORS[method],
            linestyle=STYLES[method],
            linewidth=1.9,
            label=METHOD_LABELS[method],
        )
    axes[1, 1].set_title("Independent trace-free third-order content")
    axes[1, 1].set_ylabel(r"$\|T^{\mathrm{TF}}\|_F$")

    for axis in axes.ravel():
        axis.set_xlabel(r"Time, $t/\tau$")
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Stage 55: source-local diagnosis and projected-tail qualification", fontsize=16, y=0.99)
    figure.subplots_adjust(top=0.91, bottom=0.09, left=0.09, right=0.985, hspace=0.35, wspace=0.24)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_bundle(root: Path, bundle: Path) -> None:
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.resolve() != bundle.resolve():
                stream.write(path, arcname=path.relative_to(root))


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    args = arguments()
    args.root.mkdir(parents=True, exist_ok=True)
    loaded = {method: item for method in METHODS if (item := load_method(args.root, method)) is not None}
    missing = [method for method in METHODS if method not in loaded]
    data = {method: _derived(item) for method, item in loaded.items()}
    errors = _history_errors(loaded, data)
    audit = (
        source_closure_audit(loaded[REFERENCE], tau=args.tau, prandtl=args.prandtl)
        if REFERENCE in loaded
        else {}
    )

    reference_spread = (
        replicate_spread(data[REFERENCE]["full_tensor"]) if REFERENCE in data else float("inf")
    )
    node_change = float("inf")
    time_change = float("inf")
    if all(method in data for method in ("projected_tail_base", "projected_tail_node_refined")):
        node_change = relative_history_error(
            np.mean(data["projected_tail_base"]["full_tensor"], axis=0),
            np.mean(data["projected_tail_node_refined"]["full_tensor"], axis=0),
        )
    if all(method in data for method in ("projected_tail_node_refined", SELECTED)):
        time_change = relative_history_error(
            np.mean(data["projected_tail_node_refined"]["full_tensor"], axis=0),
            np.mean(data[SELECTED]["full_tensor"], axis=0),
        )
    selected_errors = errors.get(SELECTED, {})
    selected_component = float(
        np.max(selected_errors.get("component_normalized_rmse", [float("inf")]))
    )
    invariant_ok = bool(loaded) and all(
        max(
            float(item["summary"]["invariants"]["maximum_mass_drift"]),
            float(item["summary"]["invariants"]["maximum_momentum_drift"]),
            float(item["summary"]["invariants"]["maximum_energy_trace_drift"]),
        ) < args.invariant_gate
        for item in loaded.values()
    )
    gates = {
        "all_six_runs_completed": not missing,
        "qmc_scramble_spread": reference_spread < args.reference_spread_gate,
        "projected_tail_node_refinement": node_change < args.refinement_gate,
        "projected_tail_time_refinement": time_change < args.refinement_gate,
        "collision_invariants": invariant_ok,
        "selected_H2_realizability": SELECTED in loaded
        and float(loaded[SELECTED]["summary"]["minimum_H2_margin"]) >= -5.0e-13,
        "selected_projection_target_positive": SELECTED in loaded
        and float(loaded[SELECTED]["summary"]["minimum_weight"]) > 0.0,
        "selected_full_third_error": float(selected_errors.get("third_tensor", float("inf"))) < args.third_gate,
        "selected_tracefree_error": float(selected_errors.get("trace_free", float("inf"))) < args.tracefree_gate,
        "selected_component_error": selected_component < args.component_gate,
    }
    qualification_pass = all(gates.values())
    summary = {
        "schema": "riemann35-stage55-closure-source-collection-v1",
        "scientific_scope": (
            "homogeneous source-local audit for the cubic FP closure; the reference is positive Full-FP QMC "
            "for the implemented operator and is not MD/DSMC validation"
        ),
        "important_interpretation": (
            "the selected candidate carries 35 retained moments plus 49 M5/M6 scalars and no velocity microstate; "
            "its projection target is a positive source quadrature, but H2 is the only realizability property "
            "proved for the blended 35+49 state"
        ),
        "missing_methods": missing,
        "gates": gates,
        "qualification_pass": qualification_pass,
        "reference_scramble_spread": reference_spread,
        "projected_tail_node_change": node_change,
        "projected_tail_time_change": time_change,
        "history_relative_l2_vs_qmc": errors,
        "selected_max_component_normalized_rmse": selected_component,
        "source_closure_audit": audit,
        "method_summaries": {method: item["summary"] for method, item in loaded.items()},
    }
    (args.root / "stage55_closure_source_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    rows = []
    for method, item in errors.items():
        rows.append(
            {
                "method": method,
                "heat_flux_history_relative_l2": item["heat_flux"],
                "third_tensor_history_relative_l2": item["third_tensor"],
                "trace_free_history_relative_l2": item["trace_free"],
                "maximum_component_normalized_rmse": float(np.max(item["component_normalized_rmse"])),
            }
        )
    if rows:
        with (args.root / "stage55_history_errors.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if audit:
        with (args.root / "stage55_source_errors.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "closure",
                    "third_source_relative_l2",
                    "tail_M5_M6_relative_l2",
                    "minimum_weight",
                    "maximum_moment_residual",
                    "maximum_negative_mass_fraction",
                ),
            )
            writer.writeheader()
            for kind, item in audit.items():
                writer.writerow({key: item[key] if key != "closure" else kind for key in writer.fieldnames})

    if REFERENCE in loaded:
        _plot_components(args.root / "stage55_third_order_components.png", loaded, data)
        _plot_diagnostics(args.root / "stage55_source_audit.png", loaded, data, audit)

    lines = [
        "# Stage 55: closure-source audit and projected M5/M6 memory",
        "",
        "Stage 54 identified a robust but inaccurate positive finite-mixture history. Stage 55 separates instantaneous M5/M6 source error from finite-time integration error and tests a compressed 35+49-scalar projected-tail state.",
        "",
        f"Qualification objective: **{'PASS' if qualification_pass else 'NOT YET PASSED'}**",
        "",
        "A failed objective is a scientific result, not a failed Slurm job; the collector always returns a complete bundle.",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for gate, passed in gates.items():
        lines.append(f"| {gate.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |")
    if errors:
        lines.extend(
            [
                "",
                "| Method | Heat flux | Full third tensor | Trace-free tensor | Max component |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in ("gaussian_hyqmom35", "dynamic_unprojected", "projected_tail_node_refined", SELECTED):
            if method not in errors:
                continue
            item = errors[method]
            lines.append(
                f"| {METHOD_LABELS[method]} | {item['heat_flux']:.2%} | {item['third_tensor']:.2%} | "
                f"{item['trace_free']:.2%} | {np.max(item['component_normalized_rmse']):.2%} |"
            )
    lines.extend(
        [
            "",
            "The QMC comparison diagnoses the implemented cubic FP closure only. It does not validate the collision model against MD or DSMC.",
        ]
    )
    (args.root / "STAGE55_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_bundle(args.root, args.bundle)
    print(json.dumps(_jsonable(summary), indent=2, allow_nan=True), flush=True)
    print(f"[stage55] bundle={args.bundle} qualification_pass={qualification_pass}", flush=True)


if __name__ == "__main__":
    main()

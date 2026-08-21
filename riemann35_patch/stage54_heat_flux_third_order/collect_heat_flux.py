#!/usr/bin/env python3
"""Collect, gate, plot, report, and bundle the Stage-54 audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage54")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import HYQMOM_35_INDICES  # noqa: E402
from hyqmom_fp.moments import central_moment  # noqa: E402


METHODS = (
    "qmc_base",
    "qmc_node_refined",
    "qmc_time_refined",
    "proposed_hyqmom35",
    "grad_comparator",
    "positive_tail_memory",
)
REFERENCE = "qmc_time_refined"
THIRD_INDICES = tuple(index for index in HYQMOM_35_INDICES if sum(index) == 3)
THIRD_LABELS = tuple(rf"$T_{{{''.join(str(value) for value in index)}}}$" for index in THIRD_INDICES)
METHOD_LABELS = {
    "qmc_base": "QMC base",
    "qmc_node_refined": "QMC node-refined",
    "qmc_time_refined": "positive Full-FP QMC ±2 SEM",
    "proposed_hyqmom35": "proposed HyQMOM-35",
    "grad_comparator": "Grad/GQMOM comparator",
    "positive_tail_memory": "positive tail-memory extension",
}
PLOT_METHODS = (
    "qmc_time_refined",
    "proposed_hyqmom35",
    "positive_tail_memory",
    "grad_comparator",
)
COLORS = {
    "qmc_time_refined": "#111111",
    "proposed_hyqmom35": "#c85a17",
    "positive_tail_memory": "#2468a2",
    "grad_comparator": "#9b6aa0",
}
STYLES = {
    "qmc_time_refined": "-",
    "proposed_hyqmom35": "-.",
    "positive_tail_memory": "--",
    "grad_comparator": ":",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-gate", type=float, default=0.03)
    parser.add_argument("--selected-third-gate", type=float, default=0.03)
    parser.add_argument("--selected-tracefree-gate", type=float, default=0.05)
    parser.add_argument("--selected-component-gate", type=float, default=0.03)
    parser.add_argument("--invariant-gate", type=float, default=2.0e-8)
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def load_method(root: Path, method: str) -> dict[str, object] | None:
    archive_path = root / f"stage54_{method}.npz"
    summary_path = root / f"stage54_{method}_summary.json"
    if not archive_path.is_file() or not summary_path.is_file():
        return None
    archive = np.load(archive_path)
    return {
        "times": np.asarray(archive["times"], dtype=float),
        "histories": np.asarray(archive["histories"], dtype=float),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def central_third_components(history: np.ndarray) -> np.ndarray:
    values = np.empty((history.shape[0], len(THIRD_INDICES)), dtype=float)
    for time_index, moments in enumerate(history):
        values[time_index] = [central_moment(moments, index) for index in THIRD_INDICES]
    return values


def symmetric_tensor(components: np.ndarray) -> np.ndarray:
    """Expand ten unique symmetric components to a full 3x3x3 tensor."""

    tensor = np.zeros((*components.shape[:-1], 3, 3, 3), dtype=float)
    for position, powers in enumerate(THIRD_INDICES):
        entries = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    counts = (int(i == 0) + int(j == 0) + int(k == 0),
                              int(i == 1) + int(j == 1) + int(k == 1),
                              int(i == 2) + int(j == 2) + int(k == 2))
                    if counts == powers:
                        entries.append((i, j, k))
        for i, j, k in entries:
            tensor[..., i, j, k] = components[..., position]
    return tensor


def tensor_components(tensor: np.ndarray) -> np.ndarray:
    values = []
    for powers in THIRD_INDICES:
        indices = tuple(axis for axis, count in enumerate(powers) for _ in range(count))
        values.append(tensor[..., indices[0], indices[1], indices[2]])
    return np.stack(values, axis=-1)


def irreducible_decomposition(components: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return heat flux, trace-carrying tensor, and trace-free tensor.

    For a symmetric rank-three tensor in three dimensions,
    ``T = T_TF + (delta_ij t_k + delta_ik t_j + delta_jk t_i)/5`` and
    ``q=t/2`` under the repository's heat-flux convention.
    """

    tensor = symmetric_tensor(components)
    trace = np.einsum("...ijj->...i", tensor)
    identity = np.eye(3)
    carrying = (
        np.einsum("ij,...k->...ijk", identity, trace)
        + np.einsum("ik,...j->...ijk", identity, trace)
        + np.einsum("jk,...i->...ijk", identity, trace)
    ) / 5.0
    trace_free = tensor - carrying
    return 0.5 * trace, carrying, trace_free


def derived(item: dict[str, object]) -> dict[str, np.ndarray]:
    component_replicates = np.asarray(
        [central_third_components(history) for history in item["histories"]]
    )
    heat_flux, carrying, trace_free = irreducible_decomposition(component_replicates)
    return {
        "components": component_replicates,
        "heat_flux": heat_flux,
        "carrying": carrying,
        "trace_free": trace_free,
        "full_tensor": symmetric_tensor(component_replicates),
    }


def relative_history_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1.0e-14))


def normalized_component_rmse(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    initial_scale = max(float(np.linalg.norm(symmetric_tensor(reference[:1]))), 1.0e-14)
    return np.sqrt(np.mean((candidate - reference) ** 2, axis=0)) / initial_scale


def replicate_spread(replicates: np.ndarray) -> float:
    if replicates.shape[0] < 2:
        return float("inf")
    mean = np.mean(replicates, axis=0)
    spread = np.std(replicates, axis=0, ddof=1)
    return float(np.linalg.norm(spread) / max(np.linalg.norm(mean), 1.0e-14))


def _plot_components(path: Path, loaded: dict[str, dict[str, object]], data: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(5, 2, figsize=(12.0, 14.2), sharex=True)
    reference_item = data[REFERENCE]["components"]
    reference_mean = np.mean(reference_item, axis=0)
    reference_sem = np.std(reference_item, axis=0, ddof=1) / np.sqrt(reference_item.shape[0])
    for position, axis in enumerate(axes.ravel()):
        times = loaded[REFERENCE]["times"]
        axis.fill_between(
            times,
            reference_mean[:, position] - 2.0 * reference_sem[:, position],
            reference_mean[:, position] + 2.0 * reference_sem[:, position],
            color="0.78",
            alpha=0.55,
            linewidth=0.0,
            zorder=1,
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
                linewidth=1.7 if method != REFERENCE else 1.9,
                label=METHOD_LABELS[method],
                zorder=3,
            )
        axis.set_title(THIRD_LABELS[position], fontsize=12, pad=5)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    for axis in axes[-1]:
        axis.set_xlabel(r"Time, $t/\tau$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Central moment")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    figure.suptitle("All ten third-order central moments in the oblique heat-flux case", fontsize=16, y=0.999)
    figure.subplots_adjust(top=0.93, bottom=0.055, left=0.085, right=0.985, hspace=0.34, wspace=0.20)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_summary(path: Path, loaded: dict[str, dict[str, object]], data: dict[str, dict[str, np.ndarray]], errors: dict[str, dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    q_labels = (r"$q_x$", r"$q_y$", r"$q_z$")
    reference_q = np.mean(data[REFERENCE]["heat_flux"], axis=0)
    for component, label in enumerate(q_labels):
        axes[0, 0].plot(loaded[REFERENCE]["times"], reference_q[:, component], linewidth=1.8, label=label)
    axes[0, 0].set_title("Positive-reference heat-flux components")
    axes[0, 0].set_ylabel("Heat flux")
    axes[0, 0].legend(frameon=False, ncol=3)

    for method in PLOT_METHODS:
        if method not in loaded:
            continue
        tf = np.mean(data[method]["trace_free"], axis=0)
        norm = np.linalg.norm(tf.reshape(tf.shape[0], -1), axis=1)
        axes[0, 1].plot(
            loaded[method]["times"], norm,
            color=COLORS[method], linestyle=STYLES[method], linewidth=1.8,
            label=METHOD_LABELS[method],
        )
    axes[0, 1].set_title("Trace-free third-order content")
    axes[0, 1].set_ylabel(r"$\|T^{\mathrm{TF}}\|_F$")
    axes[0, 1].legend(frameon=False, fontsize=8)

    comparison_methods = [method for method in ("proposed_hyqmom35", "positive_tail_memory", "grad_comparator") if method in errors]
    positions = np.arange(len(comparison_methods))
    width = 0.24
    for offset, metric in enumerate(("heat_flux", "third_tensor", "trace_free")):
        axes[1, 0].bar(
            positions + (offset - 1) * width,
            [100.0 * float(errors[method][metric]) for method in comparison_methods],
            width=width,
            label=metric.replace("_", " "),
        )
    axes[1, 0].set_xticks(positions, [METHOD_LABELS[method] for method in comparison_methods], rotation=14, ha="right")
    axes[1, 0].set_ylabel("History error versus fine QMC (%)")
    axes[1, 0].set_title("Accuracy separated into contracted and trace-free parts")
    axes[1, 0].legend(frameon=False, fontsize=8)

    component_errors = np.asarray(errors.get("positive_tail_memory", {}).get("component_normalized_rmse", np.zeros(10)))
    axes[1, 1].bar(np.arange(10), 100.0 * component_errors, color="#2468a2")
    axes[1, 1].set_xticks(np.arange(10), [label.replace("$", "") for label in THIRD_LABELS], rotation=45, ha="right")
    axes[1, 1].set_ylabel("Normalized RMSE (% initial tensor norm)")
    axes[1, 1].set_title("Selected positive extension: componentwise error")

    for axis in axes.ravel():
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        if axis in axes[0, :]:
            axis.set_xlabel(r"Time, $t/\tau$")
    figure.suptitle("Stage 54 heat-flux audit: independent third-order evidence", fontsize=16, y=0.99)
    figure.subplots_adjust(top=0.91, bottom=0.13, left=0.085, right=0.985, hspace=0.38, wspace=0.25)
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
    data = {method: derived(item) for method, item in loaded.items()}
    errors: dict[str, dict[str, object]] = {}

    if REFERENCE in loaded:
        reference = {key: np.mean(value, axis=0) for key, value in data[REFERENCE].items()}
        for method in METHODS:
            if method not in loaded or method == REFERENCE:
                continue
            if not np.allclose(loaded[method]["times"], loaded[REFERENCE]["times"], rtol=0.0, atol=2.0e-13):
                raise RuntimeError(f"sample times differ for {method}")
            candidate = {key: np.mean(value, axis=0) for key, value in data[method].items()}
            errors[method] = {
                "heat_flux": relative_history_error(candidate["heat_flux"], reference["heat_flux"]),
                "third_tensor": relative_history_error(candidate["full_tensor"], reference["full_tensor"]),
                "trace_carrying": relative_history_error(candidate["carrying"], reference["carrying"]),
                "trace_free": relative_history_error(candidate["trace_free"], reference["trace_free"]),
                "component_normalized_rmse": normalized_component_rmse(candidate["components"], reference["components"]),
            }

    reference_spread = (
        replicate_spread(data[REFERENCE]["full_tensor"]) if REFERENCE in data else float("inf")
    )
    qmc_node_change = float("inf")
    qmc_time_change = float("inf")
    if all(method in data for method in ("qmc_base", "qmc_node_refined")):
        qmc_node_change = relative_history_error(
            np.mean(data["qmc_base"]["full_tensor"], axis=0),
            np.mean(data["qmc_node_refined"]["full_tensor"], axis=0),
        )
    if all(method in data for method in ("qmc_node_refined", "qmc_time_refined")):
        qmc_time_change = relative_history_error(
            np.mean(data["qmc_node_refined"]["full_tensor"], axis=0),
            np.mean(data["qmc_time_refined"]["full_tensor"], axis=0),
        )
    initial = next(iter(loaded.values()))["summary"]["initial_state"] if loaded else {}
    initial_components = np.asarray(initial.get("third_components", []), dtype=float)
    initial_tensor_norm = float(np.linalg.norm(symmetric_tensor(initial_components))) if initial_components.size else 0.0
    all_active = bool(initial_components.size) and bool(np.all(np.abs(initial_components) > 5.0e-3 * initial_tensor_norm))
    invariant_ok = bool(loaded) and all(
        max(
            float(item["summary"]["invariants"]["maximum_mass_drift"]),
            float(item["summary"]["invariants"]["maximum_momentum_drift"]),
            float(item["summary"]["invariants"]["maximum_energy_trace_drift"]),
        ) < args.invariant_gate
        for item in loaded.values()
    )
    positive_selected = "positive_tail_memory" in loaded and float(loaded["positive_tail_memory"]["summary"]["minimum_quadrature_weight"]) > 0.0
    realizable_selected = "positive_tail_memory" in loaded and float(loaded["positive_tail_memory"]["summary"]["invariants"]["minimum_H2_margin"]) >= -5.0e-13
    selected_errors = errors.get("positive_tail_memory", {})
    selected_component_max = float(np.max(selected_errors.get("component_normalized_rmse", [float("inf")])))
    gates = {
        "all_six_runs_completed": not missing,
        "initial_mass_momentum_energy_constraints": bool(initial)
        and float(initial["mass_error"]) < 1.0e-12
        and float(initial["momentum_norm"]) < 1.0e-12
        and float(initial["energy_trace_error"]) < 1.0e-12,
        "nonzero_heat_flux": bool(initial) and float(initial["heat_flux_norm"]) > 0.10,
        "all_ten_third_components_active": all_active,
        "qmc_node_convergence": qmc_node_change < args.reference_gate,
        "qmc_time_convergence": qmc_time_change < args.reference_gate,
        "qmc_scramble_spread": reference_spread < args.reference_gate,
        "collision_invariants": invariant_ok,
        "selected_positive_weights": positive_selected,
        "selected_H2_realizability": realizable_selected,
        "selected_full_third_error": float(selected_errors.get("third_tensor", float("inf"))) < args.selected_third_gate,
        "selected_tracefree_error": float(selected_errors.get("trace_free", float("inf"))) < args.selected_tracefree_gate,
        "selected_component_error": selected_component_max < args.selected_component_gate,
    }
    summary = {
        "schema": "riemann35-stage54-heat-flux-collection-v1",
        "scientific_scope": (
            "homogeneous accuracy audit for all ten central third-order moments under nonzero heat flux; "
            "the Full-FP QMC reference tests closure error for the same cubic FP operator, not MD/DSMC model validity"
        ),
        "important_interpretation": (
            "heat flux is the trace contraction of the third-order tensor and its production rate enters the "
            "physical coefficient solve; the seven-dimensional trace-free tensor is the independent closure test"
        ),
        "missing_methods": missing,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "initial_state": initial,
        "reference_scramble_spread": reference_spread,
        "reference_node_change": qmc_node_change,
        "reference_time_change": qmc_time_change,
        "history_relative_l2_vs_time_refined_qmc": errors,
        "selected_max_component_normalized_rmse": selected_component_max,
        "method_summaries": {method: item["summary"] for method, item in loaded.items()},
    }
    (args.root / "stage54_heat_flux_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    rows = []
    for method, method_errors in errors.items():
        row = {
            "method": method,
            "heat_flux_history_relative_l2": method_errors["heat_flux"],
            "third_tensor_history_relative_l2": method_errors["third_tensor"],
            "trace_carrying_history_relative_l2": method_errors["trace_carrying"],
            "trace_free_history_relative_l2": method_errors["trace_free"],
        }
        for position, index in enumerate(THIRD_INDICES):
            row["component_" + "".join(str(value) for value in index) + "_normalized_rmse"] = method_errors["component_normalized_rmse"][position]
        rows.append(row)
    if rows:
        with (args.root / "stage54_heat_flux_errors.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    if REFERENCE in loaded:
        _plot_components(args.root / "stage54_third_order_components.png", loaded, data)
        _plot_summary(args.root / "stage54_heat_flux_summary.png", loaded, data, errors)

    lines = [
        "# Stage 54: third-order moments under heat flux",
        "",
        "The initial positive four-population state is rotated into three dimensions so all ten independent central third-order moments are nonzero. The comparison uses a positive Full-FP QMC hierarchy, the proposed 35-moment finite-mixture closure, the signed Grad/GQMOM comparator, and the positive tail-memory extension.",
        "",
        "Heat flux is a contraction of the third-order tensor and is partly constrained by the physical coefficient solve. The trace-free third-order tensor is therefore the independent closure-accuracy result.",
        "",
        f"Overall qualification gate: **{'PASS' if summary['overall_pass'] else 'FAIL'}**",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for gate, passed in gates.items():
        lines.append(f"| {gate.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |")
    if errors:
        lines.extend([
            "",
            "| Method | Heat-flux error | Full third tensor | Trace-free tensor | Max component error |",
            "|---|---:|---:|---:|---:|",
        ])
        for method in ("proposed_hyqmom35", "positive_tail_memory", "grad_comparator"):
            if method not in errors:
                continue
            item = errors[method]
            lines.append(
                f"| {METHOD_LABELS[method]} | {item['heat_flux']:.2%} | {item['third_tensor']:.2%} | "
                f"{item['trace_free']:.2%} | {np.max(item['component_normalized_rmse']):.2%} |"
            )
    if missing:
        lines.extend(["", "Missing or failed runs: " + ", ".join(missing)])
    lines.extend([
        "",
        "The QMC comparison quantifies closure accuracy for the implemented cubic FP operator. It does not validate that operator against MD or DSMC.",
    ])
    (args.root / "STAGE54_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_bundle(args.root, args.bundle)
    print(json.dumps(_jsonable(summary), indent=2, allow_nan=True), flush=True)
    print(f"[stage54] bundle={args.bundle}", flush=True)
    if not summary["overall_pass"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

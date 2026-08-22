#!/usr/bin/env python3
"""Collect, gate, plot, and bundle the Stage-56 qualification audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage56")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux import (  # noqa: E402
    THIRD_INDICES,
    central_third_components,
    irreducible_decomposition,
    normalized_component_rmse,
    relative_history_error,
    replicate_spread,
    symmetric_tensor,
)
from riemann35_patch.stage56_time_consistent_closure.run_time_consistent_method import (  # noqa: E402
    METHODS,
)


SELECTED = "q5_dt03125"
REFERENCE_NAME = "frozen_stage55_qmc_reference"
LABELS = {
    "q4_dt2500": r"q4, $\Delta t/\tau=2.5\times10^{-3}$",
    "q5_dt2500": r"q5, $\Delta t/\tau=2.5\times10^{-3}$",
    "q5_dt1250": r"q5, $\Delta t/\tau=1.25\times10^{-3}$",
    "q5_dt0625": r"q5, $\Delta t/\tau=6.25\times10^{-4}$",
    "q5_dt03125": r"q5, $\Delta t/\tau=3.125\times10^{-4}$",
    "q6_dt0625": r"q6, $\Delta t/\tau=6.25\times10^{-4}$",
}
COLORS = {
    "q4_dt2500": "#b8860b",
    "q5_dt2500": "#c85a17",
    "q5_dt1250": "#7b4ab5",
    "q5_dt0625": "#2673b8",
    "q5_dt03125": "#006b5f",
    "q6_dt0625": "#4a4a4a",
}
STYLES = {
    "q4_dt2500": ":",
    "q5_dt2500": "-.",
    "q5_dt1250": "--",
    "q5_dt0625": (0, (5, 2)),
    "q5_dt03125": "-",
    "q6_dt0625": (0, (1, 1)),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-spread-gate", type=float, default=0.03)
    parser.add_argument("--refinement-gate", type=float, default=0.01)
    parser.add_argument("--third-gate", type=float, default=0.03)
    parser.add_argument("--tracefree-gate", type=float, default=0.05)
    parser.add_argument("--component-gate", type=float, default=0.03)
    parser.add_argument("--invariant-gate", type=float, default=2.0e-8)
    parser.add_argument("--h3-gate", type=float, default=-1.0e-10)
    parser.add_argument("--limiter-gate", type=float, default=0.999)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_candidate(root: Path, method: str) -> dict[str, object] | None:
    archive_path = root / f"stage56_{method}.npz"
    summary_path = root / f"stage56_{method}_summary.json"
    if not archive_path.is_file() or not summary_path.is_file():
        return None
    with np.load(archive_path) as archive:
        return {
            "times": np.asarray(archive["times"], dtype=float),
            "histories": np.asarray(archive["histories"], dtype=float),
            "tails": np.asarray(archive["tails"], dtype=float),
            "h3_margins": np.asarray(archive["h3_margins"], dtype=float),
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        }


def _load_reference(reference_root: Path) -> dict[str, object]:
    archive_path = reference_root / "stage55_qmc_reference.npz"
    summary_path = reference_root / "stage55_qmc_reference_summary.json"
    if not archive_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Stage-55 reference must contain stage55_qmc_reference.npz and its summary"
        )
    with np.load(archive_path) as archive:
        return {
            "times": np.asarray(archive["times"], dtype=float),
            "histories": np.asarray(archive["histories"], dtype=float),
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "archive_path": archive_path,
            "summary_path": summary_path,
        }


def _derived(histories: np.ndarray) -> dict[str, np.ndarray]:
    components = np.asarray(
        [central_third_components(history) for history in histories]
    )
    heat_flux, carrying, trace_free = irreducible_decomposition(components)
    return {
        "components": components,
        "heat_flux": heat_flux,
        "carrying": carrying,
        "trace_free": trace_free,
        "full_tensor": symmetric_tensor(components),
    }


def _error(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]):
    candidate_mean = {key: np.mean(value, axis=0) for key, value in candidate.items()}
    reference_mean = {key: np.mean(value, axis=0) for key, value in reference.items()}
    return {
        "heat_flux": relative_history_error(
            candidate_mean["heat_flux"], reference_mean["heat_flux"]
        ),
        "third_tensor": relative_history_error(
            candidate_mean["full_tensor"], reference_mean["full_tensor"]
        ),
        "trace_free": relative_history_error(
            candidate_mean["trace_free"], reference_mean["trace_free"]
        ),
        "component_normalized_rmse": normalized_component_rmse(
            candidate_mean["components"], reference_mean["components"]
        ),
    }


def _change(coarse: dict[str, np.ndarray], fine: dict[str, np.ndarray]):
    coarse_mean = {key: np.mean(value, axis=0) for key, value in coarse.items()}
    fine_mean = {key: np.mean(value, axis=0) for key, value in fine.items()}
    return {
        key: relative_history_error(coarse_mean[key], fine_mean[key])
        for key in ("heat_flux", "full_tensor", "trace_free")
    }


def _cumulative_error(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = []
    for end in range(1, candidate.shape[0] + 1):
        values.append(
            np.linalg.norm(candidate[:end] - reference[:end])
            / max(np.linalg.norm(reference[:end]), 1.0e-14)
        )
    return np.asarray(values)


def _plot_summary(
    path: Path,
    reference: dict[str, object],
    loaded: dict[str, dict[str, object]],
    data: dict[str, dict[str, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.5))
    reference_data = data[REFERENCE_NAME]
    reference_times = np.asarray(reference["times"])
    reference_heat = np.mean(reference_data["heat_flux"], axis=0)
    reference_tracefree = np.mean(reference_data["trace_free"], axis=0)
    reference_tensor = np.mean(reference_data["full_tensor"], axis=0)
    reference_heat_norm = np.linalg.norm(reference_heat, axis=-1)
    reference_tracefree_norm = np.sqrt(
        np.sum(reference_tracefree**2, axis=(-3, -2, -1))
    )
    axes[0, 0].plot(reference_times, reference_heat_norm, color="black", linewidth=2.0, label="positive Full-FP QMC")
    axes[0, 1].plot(reference_times, reference_tracefree_norm, color="black", linewidth=2.0, label="positive Full-FP QMC")

    plot_methods = ("q5_dt2500", "q5_dt1250", "q5_dt0625", SELECTED)
    for method in plot_methods:
        if method not in loaded:
            continue
        times = np.asarray(loaded[method]["times"])
        method_data = data[method]
        heat = np.mean(method_data["heat_flux"], axis=0)
        tracefree = np.mean(method_data["trace_free"], axis=0)
        tensor = np.mean(method_data["full_tensor"], axis=0)
        style = STYLES[method]
        axes[0, 0].plot(times, np.linalg.norm(heat, axis=-1), color=COLORS[method], linestyle=style, linewidth=1.7, label=LABELS[method])
        axes[0, 1].plot(
            times,
            np.sqrt(np.sum(tracefree**2, axis=(-3, -2, -1))),
            color=COLORS[method],
            linestyle=style,
            linewidth=1.7,
            label=LABELS[method],
        )
        axes[1, 0].plot(times, 100.0 * _cumulative_error(tensor, reference_tensor), color=COLORS[method], linestyle=style, linewidth=1.7, label=LABELS[method])
        axes[1, 1].plot(times, np.asarray(loaded[method]["h3_margins"]), color=COLORS[method], linestyle=style, linewidth=1.7, label=LABELS[method])

    axes[0, 0].set_title("Heat-flux magnitude")
    axes[0, 0].set_ylabel(r"$\|q\|$")
    axes[0, 1].set_title("Independent trace-free third-order content")
    axes[0, 1].set_ylabel(r"$\|T^{TF}\|_F$")
    axes[1, 0].set_title("Accumulated full third-tensor error")
    axes[1, 0].set_ylabel("History error (%)")
    axes[1, 0].axhline(3.0, color="#8c3b32", linestyle=":", linewidth=1.2)
    axes[1, 1].set_title("Necessary H3 moment-matrix margin")
    axes[1, 1].set_ylabel("Normalized minimum eigenvalue")
    axes[1, 1].axhline(0.0, color="#8c3b32", linestyle=":", linewidth=1.2)
    for axis in axes.ravel():
        axis.set_xlabel(r"Time, $t/\tau$")
        axis.grid(alpha=0.22, linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    figure.suptitle("Stage 56: time-consistent degree-six qualification", fontsize=16, y=0.999)
    figure.subplots_adjust(top=0.86, bottom=0.08, left=0.08, right=0.985, hspace=0.33, wspace=0.22)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_components(
    path: Path,
    reference: dict[str, object],
    loaded: dict[str, dict[str, object]],
    data: dict[str, dict[str, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(5, 2, figsize=(12.2, 14.4), sharex=True)
    reference_components = data[REFERENCE_NAME]["components"]
    reference_mean = np.mean(reference_components, axis=0)
    reference_sem = np.std(reference_components, axis=0, ddof=1) / np.sqrt(reference_components.shape[0])
    labels = [rf"$T_{{{''.join(str(value) for value in index)}}}$" for index in THIRD_INDICES]
    for position, axis in enumerate(axes.ravel()):
        times = np.asarray(reference["times"])
        axis.fill_between(times, reference_mean[:, position] - 2.0 * reference_sem[:, position], reference_mean[:, position] + 2.0 * reference_sem[:, position], color="0.80", alpha=0.55, linewidth=0.0)
        axis.plot(times, reference_mean[:, position], color="black", linewidth=1.9, label="positive Full-FP QMC ±2 SEM")
        for method in ("q5_dt0625", SELECTED):
            if method not in loaded:
                continue
            axis.plot(np.asarray(loaded[method]["times"]), np.mean(data[method]["components"], axis=0)[:, position], color=COLORS[method], linestyle=STYLES[method], linewidth=1.7, label=LABELS[method])
        axis.set_title(labels[position], fontsize=12, pad=5)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    for axis in axes[-1]:
        axis.set_xlabel(r"Time, $t/\tau$")
    for axis in axes[:, 0]:
        axis.set_ylabel("Central moment")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    figure.suptitle("Stage 56: all ten third-order moments", fontsize=16, y=0.999)
    figure.subplots_adjust(top=0.93, bottom=0.055, left=0.085, right=0.985, hspace=0.34, wspace=0.20)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    args = arguments()
    args.root.mkdir(parents=True, exist_ok=True)
    reference = _load_reference(args.reference_root)
    loaded = {
        method: item
        for method in METHODS
        if (item := _load_candidate(args.root, method)) is not None
    }
    missing = [method for method in METHODS if method not in loaded]
    data = {REFERENCE_NAME: _derived(np.asarray(reference["histories"]))}
    for method, item in loaded.items():
        if not np.allclose(item["times"], reference["times"], rtol=0.0, atol=2.0e-13):
            raise RuntimeError(f"sample times differ for {method}")
        data[method] = _derived(np.asarray(item["histories"]))
    errors = {
        method: _error(data[method], data[REFERENCE_NAME])
        for method in loaded
    }

    def change(left: str, right: str):
        if left not in data or right not in data:
            return None
        return _change(data[left], data[right])

    node_coarse = change("q4_dt2500", "q5_dt2500")
    time_middle = change("q5_dt1250", "q5_dt0625")
    time_fine = change("q5_dt0625", "q5_dt03125")
    node_fine = change("q5_dt0625", "q6_dt0625")
    reference_spread = replicate_spread(data[REFERENCE_NAME]["full_tensor"])
    selected_error = errors.get(SELECTED)
    selected_summary = loaded.get(SELECTED, {}).get("summary", {})
    maximum_component = (
        float(np.max(selected_error["component_normalized_rmse"]))
        if selected_error is not None
        else float("inf")
    )
    invariant_max = max(
        (
            max(float(value) for value in item["summary"]["invariants"].values())
            for item in loaded.values()
        ),
        default=float("inf"),
    )
    minimum_h3 = min(
        (float(item["summary"]["minimum_h3_margin"]) for item in loaded.values()),
        default=-float("inf"),
    )
    minimum_weight = min(
        (float(item["summary"]["minimum_projection_weight"]) for item in loaded.values()),
        default=-float("inf"),
    )
    minimum_limiter = min(
        (
            min(
                float(item["summary"]["minimum_nonlinear_limiter"]),
                float(item["summary"]["minimum_projection_limiter"]),
            )
            for item in loaded.values()
        ),
        default=0.0,
    )

    gates = {
        "all_six_runs_completed": not missing,
        "reference_scramble_spread": reference_spread < args.reference_spread_gate,
        "positive_projection_weights": minimum_weight > 0.0,
        "collision_invariants": invariant_max < args.invariant_gate,
        "H3_necessary_condition": minimum_h3 >= args.h3_gate,
        "no_material_H3_limiter": minimum_limiter >= args.limiter_gate,
        "fine_node_convergence": node_fine is not None and max(node_fine["full_tensor"], node_fine["trace_free"]) < args.refinement_gate,
        "fine_time_convergence": time_fine is not None and max(time_fine["full_tensor"], time_fine["trace_free"]) < args.refinement_gate,
        "selected_full_third_error": selected_error is not None and selected_error["third_tensor"] < args.third_gate,
        "selected_tracefree_error": selected_error is not None and selected_error["trace_free"] < args.tracefree_gate,
        "selected_component_error": maximum_component < args.component_gate,
    }
    qualification_pass = all(gates.values())
    reference_copy = args.root / "frozen_stage55_qmc_reference.npz"
    reference_summary_copy = args.root / "frozen_stage55_qmc_reference_summary.json"
    shutil.copy2(reference["archive_path"], reference_copy)
    shutil.copy2(reference["summary_path"], reference_summary_copy)
    provenance = {
        "source_root": str(args.reference_root),
        "source_archive": str(reference["archive_path"]),
        "source_archive_sha256": _sha256(reference["archive_path"]),
        "source_summary_sha256": _sha256(reference["summary_path"]),
    }
    (args.root / "stage56_reference_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema": "riemann35-stage56-time-consistent-qualification-v1",
        "qualification_pass": qualification_pass,
        "gates": gates,
        "missing_methods": missing,
        "reference_scramble_spread": reference_spread,
        "node_change_coarse": node_coarse,
        "time_change_middle": time_middle,
        "time_change_fine": time_fine,
        "node_change_fine": node_fine,
        "history_relative_l2_vs_qmc": errors,
        "selected_max_component_normalized_rmse": maximum_component,
        "minimum_H3_margin_all_methods": minimum_h3,
        "minimum_limiter_all_methods": minimum_limiter,
        "minimum_projection_weight_all_methods": minimum_weight,
        "maximum_invariant_drift_all_methods": invariant_max,
        "selected_summary": selected_summary,
        "reference_provenance": provenance,
        "important_interpretation": "H3 PSD is a necessary but not sufficient degree-six realizability condition; a materially active limiter disqualifies the nominal 84-scalar closure even when the limited trajectory remains finite",
        "stop_rule": "advance to a spatial/shock test only if every gate passes; otherwise stop the 35+49 branch and test a positive compressed cubature",
    }
    (args.root / "stage56_time_consistent_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.root / "stage56_history_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "heat_flux_history_relative_l2", "third_tensor_history_relative_l2", "trace_free_history_relative_l2", "maximum_component_normalized_rmse"))
        for method in METHODS:
            if method not in errors:
                continue
            writer.writerow((method, errors[method]["heat_flux"], errors[method]["third_tensor"], errors[method]["trace_free"], float(np.max(errors[method]["component_normalized_rmse"]))))

    _plot_summary(args.root / "stage56_time_consistent_summary.png", reference, loaded, data)
    _plot_components(args.root / "stage56_third_order_components.png", reference, loaded, data)
    lines = [
        "# Stage 56: time-consistent degree-six qualification",
        "",
        f"Qualification objective: **{'PASS' if qualification_pass else 'NOT PASSED'}**",
        "",
        "This is a scientific stop gate. A failed objective still produces a complete bundle.",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |"
        for name, value in gates.items()
    )
    lines.extend(("", "| Method | Heat flux | Full third tensor | Trace-free tensor | Max component |", "|---|---:|---:|---:|---:|"))
    for method in METHODS:
        if method not in errors:
            continue
        item = errors[method]
        lines.append(f"| {method} | {100.0*item['heat_flux']:.2f}% | {100.0*item['third_tensor']:.2f}% | {100.0*item['trace_free']:.2f}% | {100.0*np.max(item['component_normalized_rmse']):.2f}% |")
    lines.extend(("", "H3 PSD is necessary, not sufficient, for degree-six realizability.", "", "No spatial or shock calculation is authorized unless every Stage-56 gate passes.", ""))
    (args.root / "STAGE56_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.bundle.with_suffix(args.bundle.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(args.root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.root))
    os.replace(temporary, args.bundle)
    (args.bundle.parent / f"{args.bundle.name}.sha256.txt").write_text(
        f"{_sha256(args.bundle)}  {args.bundle.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2), flush=True)
    print(f"[stage56] bundle={args.bundle} qualification_pass={qualification_pass}", flush=True)


if __name__ == "__main__":
    main()

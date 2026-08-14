#!/usr/bin/env python3
"""Collect, gate, plot, and bundle the Stage-26 four-delta audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage26")

import numpy as np


METHODS = ("full_fp_qmc", "stage9_mixture", "grad_gqmom", "adaptive_memory")
METHOD_LABELS = {
    "full_fp_qmc": "Full FP QMC",
    "stage9_mixture": "Stage-9 mixture",
    "grad_gqmom": "Grad/GQMOM",
    "adaptive_memory": "Causal memory",
}
PLOT_METRICS = ("third_order_norm", "M400", "M420", "heat_flux_norm")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--error-gate", type=float, default=0.03)
    parser.add_argument("--reference-spread-gate", type=float, default=0.03)
    return parser.parse_args()


def _load(root: Path, method: str) -> dict[str, object] | None:
    archive_path = root / f"stage26_{method}.npz"
    summary_path = root / f"stage26_{method}_summary.json"
    if not archive_path.is_file() or not summary_path.is_file():
        return None
    archive = np.load(archive_path)
    names = tuple(str(item) for item in archive["metric_names"])
    return {
        "times": archive["times"],
        "histories": archive["histories"],
        "metrics": archive["metrics"],
        "modes": archive["modes"],
        "metric_names": names,
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def _relative_history_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(reference)), 1.0e-14)
    return float(np.linalg.norm(candidate - reference) / scale)


def _replicate_spread(histories: np.ndarray) -> float:
    if histories.shape[0] < 2:
        return float("inf")
    mean = np.mean(histories, axis=0)
    spread = np.std(histories, axis=0, ddof=1)
    return float(np.linalg.norm(spread) / max(np.linalg.norm(mean), 1.0e-14))


def _plot(path: Path, loaded: dict[str, dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    reference = loaded["full_fp_qmc"]
    names = reference["metric_names"]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True)
    colors = {
        "full_fp_qmc": "#000000",
        "stage9_mixture": "#cc3311",
        "grad_gqmom": "#ee7733",
        "adaptive_memory": "#0077bb",
    }
    styles = {
        "full_fp_qmc": "--",
        "stage9_mixture": ":",
        "grad_gqmom": "-.",
        "adaptive_memory": "-",
    }
    for axis, metric in zip(axes.ravel(), PLOT_METRICS):
        position = names.index(metric)
        for method in METHODS:
            if method not in loaded:
                continue
            item = loaded[method]
            mean = np.mean(item["metrics"][:, :, position], axis=0)
            axis.plot(
                item["times"],
                mean,
                color=colors[method],
                linestyle=styles[method],
                linewidth=1.45,
                label=METHOD_LABELS[method],
            )
            if method == "full_fp_qmc" and item["metrics"].shape[0] > 1:
                sem = np.std(item["metrics"][:, :, position], axis=0, ddof=1) / np.sqrt(
                    item["metrics"].shape[0]
                )
                axis.fill_between(
                    item["times"], mean - 2.0 * sem, mean + 2.0 * sem,
                    color="0.75", alpha=0.45, linewidth=0.0,
                )
        axis.set_title(metric.replace("_", " "))
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel(r"$t/\tau$")
    axes[0, 0].legend(fontsize=7.2)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_bundle(root: Path, bundle: Path) -> None:
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stream.write(path, arcname=path.relative_to(root))


def main() -> None:
    args = arguments()
    args.root.mkdir(parents=True, exist_ok=True)
    loaded = {
        method: item
        for method in METHODS
        if (item := _load(args.root, method)) is not None
    }
    missing = [method for method in METHODS if method not in loaded]
    errors: dict[str, dict[str, float]] = {}
    reference_spread: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    shared_metrics = ()
    if "full_fp_qmc" in loaded:
        reference = loaded["full_fp_qmc"]
        shared_metrics = reference["metric_names"]
        reference_mean = np.mean(reference["metrics"], axis=0)
        for position, metric in enumerate(shared_metrics):
            reference_spread[metric] = _replicate_spread(
                reference["metrics"][:, :, position]
            )
        for method in METHODS[1:]:
            if method not in loaded:
                continue
            if loaded[method]["metric_names"] != shared_metrics:
                raise RuntimeError(f"metric ordering differs for {method}")
            candidate_mean = np.mean(loaded[method]["metrics"], axis=0)
            errors[method] = {}
            for position, metric in enumerate(shared_metrics):
                value = _relative_history_error(
                    candidate_mean[:, position], reference_mean[:, position]
                )
                errors[method][metric] = value
                rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "history_relative_l2": value,
                        "reference_scramble_spread": reference_spread[metric],
                    }
                )

    method_summaries = {
        method: item["summary"] for method, item in loaded.items()
    }
    initial = (
        next(iter(method_summaries.values()))["initial_constraints"]
        if method_summaries
        else {}
    )
    all_minimum_margin = min(
        (
            float(summary["minimum_realizability_margin"])
            for summary in method_summaries.values()
        ),
        default=float("-inf"),
    )
    maximum_mass_error = max(
        (float(summary["maximum_mass_error"]) for summary in method_summaries.values()),
        default=float("inf"),
    )
    maximum_momentum = max(
        (float(summary["maximum_momentum_norm"]) for summary in method_summaries.values()),
        default=float("inf"),
    )
    maximum_energy_error = max(
        (
            float(summary["maximum_energy_trace_error"])
            for summary in method_summaries.values()
        ),
        default=float("inf"),
    )
    adaptive_diagnostics = method_summaries.get("adaptive_memory", {}).get(
        "replicate_diagnostics", []
    )
    gates = {
        "all_four_methods_completed": not missing,
        "initial_mass_momentum_energy_constraints": bool(initial)
        and float(initial["mass_error"]) < 1.0e-12
        and float(initial["momentum_norm"]) < 1.0e-12
        and float(initial["energy_trace_error"]) < 1.0e-12,
        "initial_third_order_moments_nonzero": bool(initial)
        and float(initial["central_third_norm"]) > 0.05,
        "positive_realizable_histories": all_minimum_margin >= -5.0e-13,
        "collision_invariants": maximum_mass_error < 2.0e-8
        and maximum_momentum < 2.0e-8
        and maximum_energy_error < 2.0e-8,
        "reference_M400_scramble_spread": reference_spread.get("M400", float("inf"))
        < args.reference_spread_gate,
        "reference_M420_scramble_spread": reference_spread.get("M420", float("inf"))
        < args.reference_spread_gate,
        "adaptive_M400_history_error": errors.get("adaptive_memory", {}).get(
            "M400", float("inf")
        )
        < args.error_gate,
        "adaptive_M420_history_error": errors.get("adaptive_memory", {}).get(
            "M420", float("inf")
        )
        < args.error_gate,
        "no_blocked_causal_activation": bool(adaptive_diagnostics)
        and sum(int(item.get("blocked_activations", 0)) for item in adaptive_diagnostics)
        == 0,
    }
    summary = {
        "schema": "riemann35-stage26-four-delta-collection-v1",
        "scientific_scope": "homogeneous regularized four-delta stress test of the cubic-FP closure; this is not MD/DSMC validation of the physical FP model",
        "missing_methods": missing,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "error_gate": args.error_gate,
        "reference_spread_gate": args.reference_spread_gate,
        "initial_constraints": initial,
        "maximum_invariant_errors": {
            "mass": maximum_mass_error,
            "momentum": maximum_momentum,
            "energy_trace": maximum_energy_error,
        },
        "minimum_realizability_margin": all_minimum_margin,
        "reference_scramble_spread": reference_spread,
        "history_relative_l2_vs_full_fp_qmc": errors,
        "methods": method_summaries,
    }
    summary_path = args.root / "stage26_four_delta_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    with (args.root / "stage26_four_delta_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = (
            "method",
            "metric",
            "history_relative_l2",
            "reference_scramble_spread",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if "full_fp_qmc" in loaded:
        _plot(args.root / "stage26_four_delta_histories.png", loaded)

    lines = [
        "# Stage 26: regularized four-delta nonequilibrium audit",
        "",
        "This homogeneous screening test follows Rodney Fox's proposed four-delta construction: unit mass, zero momentum, unit energy trace, and nonzero third-order moments. The four planar deltas are represented by common narrow 3-D Gaussians so the HyQMOM/GQMOM covariance remains positive definite.",
        "",
        "The reference is a positive independently scrambled Full-FP QMC ensemble. It tests closure accuracy for the same cubic FP operator; it is not a substitute for later MD/DSMC validation of the collision model itself.",
        "",
        f"Overall gate: **{'PASS' if summary['overall_pass'] else 'FAIL'}**",
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
                "| Method | M400 error | M420 error | Third-norm error |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in METHODS[1:]:
            if method not in errors:
                continue
            lines.append(
                f"| {METHOD_LABELS[method]} | {errors[method]['M400']:.2%} | "
                f"{errors[method]['M420']:.2%} | "
                f"{errors[method]['third_order_norm']:.2%} |"
            )
    if missing:
        lines.extend(["", "Missing/failed methods: " + ", ".join(missing)])
    lines.extend(
        [
            "",
            "The adaptive run enforces no-donor persistence: because this homogeneous problem has no inflow or active spatial neighbour, an active microstate is not discarded merely because the sensor becomes temporarily safe.",
        ]
    )
    (args.root / "STAGE26_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    _write_bundle(args.root, args.bundle)
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    print(f"[stage26] bundle={args.bundle}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect, gate, plot, and bundle the Stage-56 time study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage56")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux import (  # noqa: E402
    THIRD_LABELS,
    central_third_components,
    irreducible_decomposition,
    normalized_component_rmse,
    relative_history_error,
    replicate_spread,
    symmetric_tensor,
)
from riemann35_patch.stage56_tail_time_integrator.run_time_method import (  # noqa: E402
    METHODS,
)


REFERENCE = "qmc_reference"
SELECTED = "strang_h3"
STRANG_METHODS = ("strang_h0", "strang_h1", "strang_h2", "strang_h3")
LABELS = {
    "qmc_reference": "positive QMC reference ±2 SEM",
    "legacy_lie_h0": "Stage-55-style Lie, h=0.0025",
    "strang_h0": "exact Strang, h=0.0025",
    "strang_h1": "exact Strang, h=0.00125",
    "strang_h2": "exact Strang, h=0.000625",
    "strang_h3": "exact Strang, h=0.0003125",
}
COLORS = {
    "qmc_reference": "#111111",
    "legacy_lie_h0": "#c85a17",
    "strang_h0": "#8ab6d6",
    "strang_h1": "#4f86b5",
    "strang_h2": "#1f6699",
    "strang_h3": "#004c6d",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-spread-gate", type=float, default=0.03)
    parser.add_argument("--refinement-gate", type=float, default=0.03)
    parser.add_argument("--heat-flux-gate", type=float, default=0.01)
    parser.add_argument("--third-gate", type=float, default=0.03)
    parser.add_argument("--tracefree-gate", type=float, default=0.05)
    parser.add_argument("--component-gate", type=float, default=0.03)
    parser.add_argument("--invariant-gate", type=float, default=2.0e-8)
    return parser.parse_args()


def load_method(root: Path, method: str) -> dict[str, object] | None:
    archive_path = root / f"stage56_{method}.npz"
    summary_path = root / f"stage56_{method}_summary.json"
    if not archive_path.is_file() or not summary_path.is_file():
        return None
    archive = np.load(archive_path)
    return {
        "times": np.asarray(archive["times"], dtype=float),
        "histories": np.asarray(archive["histories"], dtype=float),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def derived(item: dict[str, object]) -> dict[str, np.ndarray]:
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


def history_errors(
    loaded: dict[str, dict[str, object]],
    data: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, object]]:
    if REFERENCE not in loaded:
        return {}
    reference = {key: np.mean(value, axis=0) for key, value in data[REFERENCE].items()}
    output: dict[str, dict[str, object]] = {}
    for method, values in data.items():
        if method == REFERENCE:
            continue
        if not np.allclose(
            loaded[method]["times"], loaded[REFERENCE]["times"], rtol=0.0, atol=2.0e-13
        ):
            raise RuntimeError(f"sample times differ for {method}")
        candidate = {key: np.mean(value, axis=0) for key, value in values.items()}
        output[method] = {
            "heat_flux": relative_history_error(candidate["heat_flux"], reference["heat_flux"]),
            "third_tensor": relative_history_error(candidate["full_tensor"], reference["full_tensor"]),
            "trace_free": relative_history_error(candidate["trace_free"], reference["trace_free"]),
            "component_normalized_rmse": normalized_component_rmse(
                candidate["components"], reference["components"]
            ),
        }
    return output


def refinement_rows(
    loaded: dict[str, dict[str, object]],
    data: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for coarse, fine in zip(STRANG_METHODS[:-1], STRANG_METHODS[1:]):
        if coarse not in data or fine not in data:
            continue
        coarse_mean = {key: np.mean(value, axis=0) for key, value in data[coarse].items()}
        fine_mean = {key: np.mean(value, axis=0) for key, value in data[fine].items()}
        rows.append(
            {
                "coarse": coarse,
                "fine": fine,
                "coarse_dt_over_tau": float(loaded[coarse]["summary"]["controls"]["dt_over_tau"]),
                "fine_dt_over_tau": float(loaded[fine]["summary"]["controls"]["dt_over_tau"]),
                "heat_flux_change": relative_history_error(
                    coarse_mean["heat_flux"], fine_mean["heat_flux"]
                ),
                "third_tensor_change": relative_history_error(
                    coarse_mean["full_tensor"], fine_mean["full_tensor"]
                ),
                "trace_free_change": relative_history_error(
                    coarse_mean["trace_free"], fine_mean["trace_free"]
                ),
            }
        )
    return rows


def _plot_convergence(
    path: Path,
    loaded: dict[str, dict[str, object]],
    errors: dict[str, dict[str, object]],
    refinements: list[dict[str, float | str]],
) -> None:
    import matplotlib.pyplot as plt

    available = [method for method in STRANG_METHODS if method in errors]
    dt = np.asarray(
        [loaded[method]["summary"]["controls"]["dt_over_tau"] for method in available]
    )
    order = np.argsort(dt)
    dt = dt[order]
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    quantities = (
        ("third_tensor", "Full third tensor error", 0.03),
        ("trace_free", "Trace-free third tensor error", 0.05),
        ("heat_flux", "Heat-flux error", 0.01),
    )
    for axis, (key, title, objective) in zip(axes.ravel()[:3], quantities):
        values = np.asarray([errors[method][key] for method in available])[order]
        axis.loglog(dt, values, "o-", color="#005a8d", linewidth=1.8, markersize=5)
        axis.axhline(objective, color="#8c3b32", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel(r"$\Delta t/\tau$")
        axis.set_ylabel("Relative history error")
        axis.grid(alpha=0.25, which="both")
        axis.invert_xaxis()
    axis = axes[1, 1]
    if refinements:
        fine_dt = np.asarray([row["fine_dt_over_tau"] for row in refinements], dtype=float)
        change = np.asarray([row["third_tensor_change"] for row in refinements], dtype=float)
        order = np.argsort(fine_dt)
        axis.loglog(fine_dt[order], change[order], "o-", color="#2a9d68", linewidth=1.8)
    axis.axhline(0.03, color="#8c3b32", linestyle="--", linewidth=1.2)
    axis.set_title("Successive full-tensor change")
    axis.set_xlabel(r"Fine $\Delta t/\tau$")
    axis.set_ylabel("Relative change")
    axis.grid(alpha=0.25, which="both")
    axis.invert_xaxis()
    figure.suptitle("Stage 56: exact-Strang tail time-integration qualification", fontsize=16)
    figure.subplots_adjust(top=0.91, bottom=0.09, left=0.09, right=0.98, hspace=0.34, wspace=0.25)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


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
    methods = (REFERENCE, "legacy_lie_h0", "strang_h0", SELECTED)
    styles = {REFERENCE: "-", "legacy_lie_h0": "-.", "strang_h0": "--", SELECTED: "-"}
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
        for method in methods:
            if method not in data:
                continue
            values = np.mean(data[method]["components"], axis=0)
            axis.plot(
                loaded[method]["times"],
                values[:, position],
                color=COLORS[method],
                linestyle=styles[method],
                linewidth=1.9 if method in (REFERENCE, SELECTED) else 1.5,
                label=LABELS[method],
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
    figure.suptitle("Stage 56: third-order moments after exact tail splitting", fontsize=16, y=0.999)
    figure.subplots_adjust(top=0.93, bottom=0.055, left=0.085, right=0.985, hspace=0.34, wspace=0.20)
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
    errors = history_errors(loaded, data)
    refinements = refinement_rows(loaded, data)
    reference_spread = (
        replicate_spread(data[REFERENCE]["full_tensor"]) if REFERENCE in data else float("inf")
    )
    selected = errors.get(SELECTED, {})
    component_error = float(
        np.max(selected.get("component_normalized_rmse", [float("inf")]))
    )
    changes = [float(row["third_tensor_change"]) for row in refinements]
    finest_change = changes[-1] if len(changes) == 3 else float("inf")
    contracts = len(changes) == 3 and changes[2] < changes[1] < changes[0]
    invariant_ok = bool(loaded) and all(
        max(
            float(item["summary"]["invariants"]["maximum_mass_drift"]),
            float(item["summary"]["invariants"]["maximum_momentum_drift"]),
            float(item["summary"]["invariants"]["maximum_energy_trace_drift"]),
        ) < args.invariant_gate
        for item in loaded.values()
    )
    selected_summary = loaded.get(SELECTED, {}).get("summary", {})
    gates = {
        "all_six_runs_completed": not missing,
        "qmc_scramble_spread": reference_spread < args.reference_spread_gate,
        "strang_refinement_contracts": contracts,
        "finest_time_change": finest_change < args.refinement_gate,
        "collision_invariants": invariant_ok,
        "selected_H2_realizability": bool(selected_summary)
        and float(selected_summary.get("minimum_H2_margin", -1.0)) >= -5.0e-13,
        "selected_projection_positive": bool(selected_summary)
        and float(selected_summary.get("minimum_weight", 0.0)) > 0.0,
        "selected_no_negative_mass": bool(selected_summary)
        and max(
            float(item.get("maximum_negative_mass_fraction", 1.0))
            for item in selected_summary.get("replicate_diagnostics", [{}])
        ) == 0.0,
        "selected_heat_flux_error": float(selected.get("heat_flux", float("inf"))) < args.heat_flux_gate,
        "selected_full_third_error": float(selected.get("third_tensor", float("inf"))) < args.third_gate,
        "selected_tracefree_error": float(selected.get("trace_free", float("inf"))) < args.tracefree_gate,
        "selected_component_error": component_error < args.component_gate,
    }
    qualification_pass = all(gates.values())
    summary = {
        "schema": "riemann35-stage56-tail-time-gate-v1",
        "scientific_scope": (
            "time-integration qualification of the positive 35+49 moment closure for Rodney's "
            "oblique Riemann35 state; the QMC reference is for the implemented collision operator "
            "and is not MD/DSMC validation"
        ),
        "method_change": (
            "the stiff algebraic M5/M6 relaxation is integrated exactly in two symmetric half steps; "
            "the positive target is rebuilt from the current 35 moments and no velocity microstate is retained"
        ),
        "missing_methods": missing,
        "gates": gates,
        "qualification_pass": qualification_pass,
        "reference_scramble_spread": reference_spread,
        "finest_time_change": finest_change,
        "history_relative_l2_vs_qmc": errors,
        "refinement": refinements,
        "selected_max_component_normalized_rmse": component_error,
        "method_summaries": {method: item["summary"] for method, item in loaded.items()},
    }
    summary_path = args.root / "stage56_tail_time_summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    if errors:
        with (args.root / "stage56_history_errors.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = (
                "method",
                "heat_flux_history_relative_l2",
                "third_tensor_history_relative_l2",
                "trace_free_history_relative_l2",
                "maximum_component_normalized_rmse",
            )
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for method, values in errors.items():
                writer.writerow(
                    {
                        "method": method,
                        "heat_flux_history_relative_l2": values["heat_flux"],
                        "third_tensor_history_relative_l2": values["third_tensor"],
                        "trace_free_history_relative_l2": values["trace_free"],
                        "maximum_component_normalized_rmse": float(
                            np.max(values["component_normalized_rmse"])
                        ),
                    }
                )
    if refinements:
        with (args.root / "stage56_refinement.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(refinements[0]))
            writer.writeheader()
            writer.writerows(refinements)
    if REFERENCE in data:
        _plot_convergence(args.root / "stage56_time_convergence.png", loaded, errors, refinements)
        _plot_components(args.root / "stage56_third_order_components.png", loaded, data)
    lines = [
        "# Stage 56: exact projected-tail time-integration gate",
        "",
        f"Qualification objective: **{'PASS' if qualification_pass else 'NOT YET PASSED'}**",
        "",
        "This is the Riemann35 moment-closure study. The QMC calculation is an internal reference evaluator, not a separate FP-PINN result.",
        "",
        "| Gate | Result |",
        "|---|---:|",
        *[f"| {name.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |" for name, value in gates.items()],
        "",
        "| Method | Heat flux | Full third tensor | Trace-free tensor | Max component |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in errors.items():
        lines.append(
            f"| {LABELS[method]} | {100.0 * float(values['heat_flux']):.3f}% | "
            f"{100.0 * float(values['third_tensor']):.3f}% | "
            f"{100.0 * float(values['trace_free']):.3f}% | "
            f"{100.0 * float(np.max(values['component_normalized_rmse'])):.3f}% |"
        )
    lines.extend(("", f"Finest successive full-tensor change: {100.0 * finest_change:.3f}%", ""))
    (args.root / "STAGE56_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    _write_bundle(args.root, args.bundle)
    digest = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    (args.bundle.parent / f"{args.bundle.name}.sha256.txt").write_text(
        f"{digest}  {args.bundle.name}\n", encoding="utf-8"
    )
    print(f"[stage56] bundle={args.bundle} qualification_pass={qualification_pass}", flush=True)


if __name__ == "__main__":
    main()

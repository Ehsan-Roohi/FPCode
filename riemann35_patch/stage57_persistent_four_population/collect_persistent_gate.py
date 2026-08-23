#!/usr/bin/env python3
"""Collect, gate, plot, and bundle the Stage-57 closure study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage57")

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
from riemann35_patch.stage57_persistent_four_population.run_persistent_method import (  # noqa: E402
    METHODS,
)


REFERENCE = "qmc_reference"
CONTROL = "stage56_strang_control"
SELECTED = "persistent4_h3"
PERSISTENT_METHODS = (
    "persistent4_h0",
    "persistent4_h1",
    "persistent4_h2",
    "persistent4_h3",
)
LABELS = {
    REFERENCE: "positive QMC reference ±2 SEM",
    CONTROL: "Stage-56 exact-Strang control",
    "persistent4_h0": "persistent positive four-population, h=0.0025",
    "persistent4_h1": "persistent positive four-population, h=0.00125",
    "persistent4_h2": "persistent positive four-population, h=0.000625",
    "persistent4_h3": "persistent positive four-population, h=0.0003125",
}
COLORS = {
    REFERENCE: "#111111",
    CONTROL: "#c85a17",
    "persistent4_h0": "#8ab6d6",
    "persistent4_h1": "#4f86b5",
    "persistent4_h2": "#1f6699",
    "persistent4_h3": "#006d5b",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-spread-gate", type=float, default=0.03)
    parser.add_argument("--refinement-gate", type=float, default=0.015)
    parser.add_argument("--heat-flux-gate", type=float, default=0.01)
    parser.add_argument("--third-gate", type=float, default=0.03)
    parser.add_argument("--tracefree-gate", type=float, default=0.05)
    parser.add_argument("--component-gate", type=float, default=0.03)
    parser.add_argument("--component-relative-allowance", type=float, default=0.20)
    parser.add_argument("--source-gate", type=float, default=1.0e-9)
    parser.add_argument("--invariant-gate", type=float, default=2.0e-8)
    return parser.parse_args()


def load_method(root: Path, method: str) -> dict[str, object] | None:
    archive_path = root / f"stage57_{method}.npz"
    summary_path = root / f"stage57_{method}_summary.json"
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
    if REFERENCE not in data:
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
            "third_tensor": relative_history_error(
                candidate["full_tensor"], reference["full_tensor"]
            ),
            "trace_free": relative_history_error(
                candidate["trace_free"], reference["trace_free"]
            ),
            "component_normalized_rmse": normalized_component_rmse(
                candidate["components"], reference["components"]
            ),
        }
    return output


def component_qualification(
    reference_replicates: np.ndarray,
    candidate: np.ndarray,
    relative_allowance: float,
) -> list[dict[str, float | str]]:
    reference = np.mean(reference_replicates, axis=0)
    sem = np.std(reference_replicates, axis=0, ddof=1) / np.sqrt(
        reference_replicates.shape[0]
    )
    rows = []
    for position, label in enumerate(THIRD_LABELS):
        error = float(np.linalg.norm(candidate[:, position] - reference[:, position]))
        reference_norm = float(np.linalg.norm(reference[:, position]))
        two_sem = float(2.0 * np.linalg.norm(sem[:, position]))
        allowance = max(relative_allowance * reference_norm, two_sem, 1.0e-14)
        rows.append(
            {
                "component": label.replace("$", ""),
                "relative_history_error": error / max(reference_norm, 1.0e-14),
                "two_sem_relative": two_sem / max(reference_norm, 1.0e-14),
                "qualification_ratio": error / allowance,
            }
        )
    return rows


def refinement_rows(
    loaded: dict[str, dict[str, object]],
    data: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, float | str]]:
    rows = []
    for coarse, fine in zip(PERSISTENT_METHODS[:-1], PERSISTENT_METHODS[1:]):
        if coarse not in data or fine not in data:
            continue
        coarse_mean = {key: np.mean(value, axis=0) for key, value in data[coarse].items()}
        fine_mean = {key: np.mean(value, axis=0) for key, value in data[fine].items()}
        rows.append(
            {
                "coarse": coarse,
                "fine": fine,
                "coarse_dt_over_tau": float(
                    loaded[coarse]["summary"]["controls"]["dt_over_tau"]
                ),
                "fine_dt_over_tau": float(
                    loaded[fine]["summary"]["controls"]["dt_over_tau"]
                ),
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

    available = [method for method in PERSISTENT_METHODS if method in errors]
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
        axis.loglog(dt, values, "o-", color="#006d5b", linewidth=1.8, markersize=5)
        axis.axhline(objective, color="#8c3b32", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel(r"$\Delta t/\tau$")
        axis.set_ylabel("Relative history error")
        axis.grid(alpha=0.25, which="both")
        axis.invert_xaxis()
    axis = axes[1, 1]
    if refinements:
        fine_dt = np.asarray([row["fine_dt_over_tau"] for row in refinements])
        change = np.asarray([row["third_tensor_change"] for row in refinements])
        order = np.argsort(fine_dt)
        axis.loglog(fine_dt[order], change[order], "o-", color="#2a6fbb", linewidth=1.8)
    axis.axhline(0.015, color="#8c3b32", linestyle="--", linewidth=1.2)
    axis.set_title("Successive full-tensor change")
    axis.set_xlabel(r"Fine $\Delta t/\tau$")
    axis.set_ylabel("Relative change")
    axis.grid(alpha=0.25, which="both")
    axis.invert_xaxis()
    figure.suptitle("Stage 57: persistent positive four-population qualification", fontsize=16)
    figure.subplots_adjust(
        top=0.91, bottom=0.09, left=0.09, right=0.98, hspace=0.34, wspace=0.25
    )
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
    methods = (REFERENCE, CONTROL, "persistent4_h0", SELECTED)
    styles = {REFERENCE: "-", CONTROL: "-.", "persistent4_h0": "--", SELECTED: "-"}
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
    figure.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.986)
    )
    figure.suptitle(
        "Stage 57: all third-order moments with persistent positive populations",
        fontsize=16,
        y=0.999,
    )
    figure.subplots_adjust(
        top=0.93, bottom=0.055, left=0.085, right=0.985, hspace=0.34, wspace=0.20
    )
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
    loaded = {
        method: item
        for method in METHODS
        if (item := load_method(args.root, method)) is not None
    }
    missing = [method for method in METHODS if method not in loaded]
    data = {method: derived(item) for method, item in loaded.items()}
    errors = history_errors(loaded, data)
    refinements = refinement_rows(loaded, data)
    reference_spread = (
        replicate_spread(data[REFERENCE]["full_tensor"])
        if REFERENCE in data
        else float("inf")
    )
    selected = errors.get(SELECTED, {})
    component_error = float(
        np.max(selected.get("component_normalized_rmse", [float("inf")]))
    )
    component_rows = []
    if REFERENCE in data and SELECTED in data:
        component_rows = component_qualification(
            data[REFERENCE]["components"],
            np.mean(data[SELECTED]["components"], axis=0),
            args.component_relative_allowance,
        )
    maximum_component_ratio = max(
        [float(row["qualification_ratio"]) for row in component_rows],
        default=float("inf"),
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
    diagnostics = selected_summary.get("replicate_diagnostics", [{}])[0]
    source_ok = max(
        float(diagnostics.get("initial_moment_relative_residual", float("inf"))),
        float(diagnostics.get("initial_tail_relative_error", float("inf"))),
        float(diagnostics.get("initial_third_source_relative_error", float("inf"))),
    ) < args.source_gate
    gates = {
        "all_six_runs_completed": not missing,
        "qmc_scramble_spread": reference_spread < args.reference_spread_gate,
        "persistent_refinement_contracts": contracts,
        "finest_time_change": finest_change < args.refinement_gate,
        "collision_invariants": invariant_ok,
        "initial_source_and_tail_exactness": source_ok,
        "selected_H2_realizability": bool(selected_summary)
        and float(selected_summary.get("minimum_H2_margin", -1.0)) >= -5.0e-13,
        "selected_positive_weights": bool(selected_summary)
        and float(selected_summary.get("minimum_weight", 0.0)) > 0.0,
        "selected_positive_covariances": float(
            diagnostics.get("minimum_covariance_eigenvalue", -1.0)
        ) > 0.0,
        "selected_full_heat_flux_projection": float(
            diagnostics.get("minimum_projection_fraction", 0.0)
        ) == 1.0,
        "selected_projection_residual": float(
            diagnostics.get("maximum_projection_residual", float("inf"))
        ) < 1.0e-8,
        "selected_compact_state": int(
            selected_summary.get("persistent_state_scalars", 10**9)
        ) <= 41,
        "selected_heat_flux_error": float(selected.get("heat_flux", float("inf")))
        < args.heat_flux_gate,
        "selected_full_third_error": float(
            selected.get("third_tensor", float("inf"))
        ) < args.third_gate,
        "selected_tracefree_error": float(
            selected.get("trace_free", float("inf"))
        ) < args.tracefree_gate,
        "selected_component_normalized_error": component_error < args.component_gate,
        "selected_component_sem_aware_error": maximum_component_ratio < 1.0,
    }
    qualification_pass = all(gates.values())
    summary = {
        "schema": "riemann35-stage57-persistent-four-population-gate-v1",
        "scientific_scope": (
            "compact positive persistent-population qualification for Rodney's oblique "
            "Riemann35 moment state; QMC is an internal evaluator for the implemented "
            "collision operator and is not MD/DSMC validation"
        ),
        "method_change": (
            "retain four labelled positive Gaussian populations and impose the analytic "
            "heat-flux relaxation through a positivity-preserving mean/covariance projection; "
            "no velocity microstate or QMC-fitted parameter is retained"
        ),
        "missing_methods": missing,
        "gates": gates,
        "qualification_pass": qualification_pass,
        "reference_scramble_spread": reference_spread,
        "finest_time_change": finest_change,
        "history_relative_l2_vs_qmc": errors,
        "refinement": refinements,
        "component_sem_aware_qualification": component_rows,
        "selected_max_component_normalized_rmse": component_error,
        "selected_max_component_qualification_ratio": maximum_component_ratio,
        "method_summaries": {
            method: item["summary"] for method, item in loaded.items()
        },
    }
    (args.root / "stage57_persistent_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    if errors:
        with (args.root / "stage57_history_errors.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
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
    for filename, rows in (
        ("stage57_refinement.csv", refinements),
        ("stage57_component_qualification.csv", component_rows),
    ):
        if rows:
            with (args.root / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    if REFERENCE in data:
        _plot_convergence(
            args.root / "stage57_time_convergence.png", loaded, errors, refinements
        )
        _plot_components(
            args.root / "stage57_third_order_components.png", loaded, data
        )
    lines = [
        "# Stage 57: persistent positive four-population closure",
        "",
        f"Qualification objective: **{'PASS' if qualification_pass else 'NOT YET PASSED'}**",
        "",
        "This is the Riemann35 moment-closure study. The QMC calculation is an internal reference evaluator, not FP-PINN or MD/DSMC validation.",
        "",
        "| Gate | Result |",
        "|---|---:|",
        *[
            f"| {name.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |"
            for name, value in gates.items()
        ],
        "",
        "| Method | Heat flux | Full third tensor | Trace-free tensor | Max normalized component |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in errors.items():
        lines.append(
            f"| {LABELS[method]} | {100.0 * float(values['heat_flux']):.3f}% | "
            f"{100.0 * float(values['third_tensor']):.3f}% | "
            f"{100.0 * float(values['trace_free']):.3f}% | "
            f"{100.0 * float(np.max(values['component_normalized_rmse'])):.3f}% |"
        )
    lines.extend(
        (
            "",
            f"Finest successive full-tensor change: {100.0 * finest_change:.3f}%",
            f"Worst SEM-aware component qualification ratio: {maximum_component_ratio:.3f}",
            "",
        )
    )
    (args.root / "STAGE57_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    _write_bundle(args.root, args.bundle)
    digest = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    (args.bundle.parent / f"{args.bundle.name}.sha256.txt").write_text(
        f"{digest}  {args.bundle.name}\n", encoding="utf-8"
    )
    print(
        f"[stage57] bundle={args.bundle} qualification_pass={qualification_pass}",
        flush=True,
    )


if __name__ == "__main__":
    main()

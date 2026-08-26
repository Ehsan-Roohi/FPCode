#!/usr/bin/env python3
"""Collect, qualify, visualize, and bundle the Stage-58 blind suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage58")

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
from riemann35_patch.stage57_persistent_four_population.collect_persistent_gate import (  # noqa: E402
    component_qualification,
)
from riemann35_patch.stage58_blind_generalization.blind_cases import (  # noqa: E402
    ANCHOR_CASE,
    BLIND_CASES,
    CASE_NAMES,
    registry_manifest,
)


LABELS = {
    ANCHOR_CASE: "Stage-57 anchor",
    "hot_dense_shifted": "hot/dense, shifted",
    "broad_shifted": "broad, shifted/scaled",
    "alternate_weights": "alternate weights/centers",
    "anisotropic_3d": "anisotropic 3-D",
}
COLORS = {
    ANCHOR_CASE: "#555555",
    "hot_dense_shifted": "#0072B2",
    "broad_shifted": "#E69F00",
    "alternate_weights": "#009E73",
    "anisotropic_3d": "#CC79A7",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-spread-gate", type=float, default=0.02)
    parser.add_argument("--time-change-gate", type=float, default=0.01)
    parser.add_argument("--heat-flux-gate", type=float, default=0.01)
    parser.add_argument("--third-gate", type=float, default=0.03)
    parser.add_argument("--tracefree-gate", type=float, default=0.05)
    parser.add_argument("--component-gate", type=float, default=0.03)
    parser.add_argument("--component-relative-allowance", type=float, default=0.20)
    parser.add_argument("--source-gate", type=float, default=1.0e-9)
    parser.add_argument("--projection-gate", type=float, default=1.0e-8)
    parser.add_argument("--invariant-gate", type=float, default=2.0e-8)
    return parser.parse_args()


def _load_case(root: Path, name: str) -> dict[str, object] | None:
    archive_path = root / f"stage58_{name}.npz"
    summary_path = root / f"stage58_{name}_summary.json"
    if not archive_path.is_file() or not summary_path.is_file():
        return None
    archive = np.load(archive_path)
    return {
        "times": np.asarray(archive["times"], dtype=float),
        "qmc": np.asarray(archive["qmc_histories"], dtype=float),
        "coarse": np.asarray(archive["persistent_coarse_histories"], dtype=float),
        "fine": np.asarray(archive["persistent_fine_histories"], dtype=float),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
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


def _maximum_invariant(summary: dict[str, object]) -> float:
    return max(
        float(summary["maximum_mass_drift"]),
        float(summary["maximum_momentum_drift"]),
        float(summary["maximum_energy_trace_drift"]),
    )


def _case_result(
    name: str,
    loaded: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    qmc = _derived(loaded["qmc"])
    coarse = _derived(loaded["coarse"])
    fine = _derived(loaded["fine"])
    reference = {key: np.mean(value, axis=0) for key, value in qmc.items()}
    candidate = {key: np.mean(value, axis=0) for key, value in fine.items()}
    coarse_candidate = {key: np.mean(value, axis=0) for key, value in coarse.items()}
    errors = {
        "heat_flux": relative_history_error(
            candidate["heat_flux"], reference["heat_flux"]
        ),
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
    maximum_component = float(np.max(errors["component_normalized_rmse"]))
    time_change = relative_history_error(
        candidate["full_tensor"], coarse_candidate["full_tensor"]
    )
    component_rows = component_qualification(
        qmc["components"],
        candidate["components"],
        args.component_relative_allowance,
    )
    maximum_component_ratio = max(
        float(row["qualification_ratio"]) for row in component_rows
    )
    qmc_spread = replicate_spread(qmc["full_tensor"])
    summary = loaded["summary"]
    fine_summary = summary["persistent_fine"]
    fine_diagnostics = fine_summary["diagnostics"]
    initial = summary["initial_audit"]
    registry = registry_manifest()
    fingerprint_ok = (
        summary["case_fingerprint"]
        == registry["case_fingerprints"][name]
        == summary["registry"]["case_fingerprints"][name]
    )
    flags = summary["controls"]
    gates = {
        "fingerprint_frozen": fingerprint_ok,
        "no_qmc_case_design": flags.get("qmc_used_to_define_case") is False,
        "no_closure_refit": flags.get("closure_parameters_refit") is False,
        "initial_constraints": max(
            float(initial["mass_error"]),
            float(initial["bulk_velocity_error"]),
            float(initial["energy_trace_error"]),
        )
        < 1.0e-10,
        "qmc_scramble_spread": qmc_spread < args.reference_spread_gate,
        "time_change": time_change < args.time_change_gate,
        "collision_invariants": max(
            _maximum_invariant(summary["qmc"]["invariants"]),
            _maximum_invariant(summary["persistent_coarse"]["invariants"]),
            _maximum_invariant(summary["persistent_fine"]["invariants"]),
        )
        < args.invariant_gate,
        "initial_source_and_tail_exactness": max(
            float(fine_diagnostics["initial_moment_relative_residual"]),
            float(fine_diagnostics["initial_tail_relative_error"]),
            float(fine_diagnostics["initial_third_source_relative_error"]),
        )
        < args.source_gate,
        "H2_realizability": float(fine_diagnostics["minimum_H2_margin"])
        >= -5.0e-13,
        "positive_weights": float(fine_diagnostics["minimum_weight"]) > 0.0,
        "positive_covariances": float(
            fine_diagnostics["minimum_covariance_eigenvalue"]
        )
        > 0.0,
        "full_heat_flux_projection": float(
            fine_diagnostics["minimum_projection_fraction"]
        )
        >= 1.0 - 2.0e-13,
        "projection_residual": float(
            fine_diagnostics["maximum_projection_residual"]
        )
        < args.projection_gate,
        "compact_state": int(fine_diagnostics["persistent_state_scalars"]) <= 41,
        "heat_flux_error": float(errors["heat_flux"]) < args.heat_flux_gate,
        "full_third_error": float(errors["third_tensor"]) < args.third_gate,
        "tracefree_error": float(errors["trace_free"]) < args.tracefree_gate,
        "component_normalized_error": maximum_component < args.component_gate,
        "component_sem_aware_error": maximum_component_ratio < 1.0,
    }
    return {
        "case": name,
        "role": summary["role"],
        "fingerprint": summary["case_fingerprint"],
        "gates": gates,
        "pass": all(gates.values()),
        "qmc_scramble_spread": qmc_spread,
        "time_change": time_change,
        "errors": errors,
        "maximum_component_normalized_rmse": maximum_component,
        "maximum_component_qualification_ratio": maximum_component_ratio,
        "component_qualification": component_rows,
        "fine_diagnostics": fine_diagnostics,
        "derived": {"qmc": qmc, "coarse": coarse, "fine": fine},
    }


def _plot_error_summary(path: Path, results: dict[str, dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    names = [name for name in CASE_NAMES if name in results]
    positions = np.arange(len(names))
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    panels = (
        ("third_tensor", "Full third-tensor error", 0.03),
        ("trace_free", "Trace-free third-tensor error", 0.05),
        ("heat_flux", "Heat-flux error", 0.01),
    )
    for axis, (key, title, gate) in zip(axes.ravel()[:3], panels):
        values = [float(results[name]["errors"][key]) for name in names]
        axis.plot(positions, values, "o-", color="#006D5B", linewidth=1.8)
        axis.axhline(gate, color="#9E3D32", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.set_ylabel("Relative history error")
        axis.grid(alpha=0.25)
        axis.set_xticks(positions, [LABELS[name] for name in names], rotation=18, ha="right")
    axis = axes[1, 1]
    changes = [float(results[name]["time_change"]) for name in names]
    axis.plot(positions, changes, "o-", color="#2A6FBB", linewidth=1.8)
    axis.axhline(0.01, color="#9E3D32", linestyle="--", linewidth=1.2)
    axis.set_title("Fine/coarse full-tensor change")
    axis.set_ylabel("Relative change")
    axis.grid(alpha=0.25)
    axis.set_xticks(positions, [LABELS[name] for name in names], rotation=18, ha="right")
    figure.suptitle("Stage 58: prospective blind generalization", fontsize=16)
    figure.subplots_adjust(
        top=0.91, bottom=0.17, left=0.085, right=0.985, hspace=0.36, wspace=0.24
    )
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_heat_flux(path: Path, loaded: dict[str, dict[str, object]], results: dict[str, dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(11.8, 10.2), sharex=True)
    for axis, name in zip(axes.ravel(), CASE_NAMES):
        if name not in results:
            axis.set_visible(False)
            continue
        times = loaded[name]["times"]
        qmc = results[name]["derived"]["qmc"]["heat_flux"]
        fine = results[name]["derived"]["fine"]["heat_flux"]
        reference_norm = np.linalg.norm(qmc, axis=-1)
        reference_mean = np.mean(reference_norm, axis=0)
        reference_sem = np.std(reference_norm, axis=0, ddof=1) / np.sqrt(qmc.shape[0])
        candidate_norm = np.linalg.norm(np.mean(fine, axis=0), axis=-1)
        axis.fill_between(
            times,
            reference_mean - 2.0 * reference_sem,
            reference_mean + 2.0 * reference_sem,
            color="0.82",
            linewidth=0.0,
        )
        axis.plot(times, reference_mean, color="#111111", linewidth=1.9, label="QMC ±2 SEM")
        axis.plot(
            times,
            candidate_norm,
            color=COLORS[name],
            linewidth=2.0,
            label="persistent four-population",
        )
        axis.set_title(LABELS[name])
        axis.set_ylabel(r"$|q|$")
        axis.grid(alpha=0.23)
    axes.ravel()[0].legend(frameon=False, fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel(r"Time, $t/\tau$")
    axes.ravel()[-1].set_visible(False)
    figure.suptitle("Stage 58: heat-flux histories without closure refitting", fontsize=16)
    figure.subplots_adjust(
        top=0.92, bottom=0.07, left=0.08, right=0.985, hspace=0.31, wspace=0.22
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
        name: item
        for name in CASE_NAMES
        if (item := _load_case(args.root, name)) is not None
    }
    missing = [name for name in CASE_NAMES if name not in loaded]
    results: dict[str, dict[str, object]] = {}
    for name, item in loaded.items():
        try:
            results[name] = _case_result(name, item, args)
        except Exception as error:
            results[name] = {
                "case": name,
                "pass": False,
                "collector_error": f"{type(error).__name__}: {error}",
                "gates": {},
            }
    anchor_pass = bool(results.get(ANCHOR_CASE, {}).get("pass", False))
    blind_pass = all(bool(results.get(name, {}).get("pass", False)) for name in BLIND_CASES)
    qualification_pass = not missing and anchor_pass and blind_pass
    summary = {
        "schema": "riemann35-stage58-blind-generalization-gate-v1",
        "scientific_scope": (
            "prospective generalization of the compact positive persistent four-population "
            "Riemann35 moment closure; QMC is an internal collision-operator reference, not "
            "MD/DSMC validation"
        ),
        "registry": registry_manifest(),
        "pre_reference_model_only_preflight": registry_manifest()[
            "model_only_preflight_disclosure"
        ],
        "missing_cases": missing,
        "anchor_pass": anchor_pass,
        "all_four_blind_cases_pass": blind_pass,
        "qualification_pass": qualification_pass,
        "thresholds": {
            "reference_spread": args.reference_spread_gate,
            "time_change": args.time_change_gate,
            "heat_flux": args.heat_flux_gate,
            "third_tensor": args.third_gate,
            "trace_free": args.tracefree_gate,
            "component": args.component_gate,
            "projection": args.projection_gate,
        },
        "case_results": {
            name: {
                key: value
                for key, value in result.items()
                if key != "derived"
            }
            for name, result in results.items()
        },
    }
    (args.root / "stage58_generalization_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    with (args.root / "stage58_case_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = (
            "case",
            "role",
            "pass",
            "qmc_scramble_spread",
            "time_change",
            "heat_flux_error",
            "third_tensor_error",
            "trace_free_error",
            "maximum_component_normalized_rmse",
            "maximum_component_qualification_ratio",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in CASE_NAMES:
            result = results.get(name)
            if result is None or "errors" not in result:
                continue
            writer.writerow(
                {
                    "case": name,
                    "role": result["role"],
                    "pass": result["pass"],
                    "qmc_scramble_spread": result["qmc_scramble_spread"],
                    "time_change": result["time_change"],
                    "heat_flux_error": result["errors"]["heat_flux"],
                    "third_tensor_error": result["errors"]["third_tensor"],
                    "trace_free_error": result["errors"]["trace_free"],
                    "maximum_component_normalized_rmse": result[
                        "maximum_component_normalized_rmse"
                    ],
                    "maximum_component_qualification_ratio": result[
                        "maximum_component_qualification_ratio"
                    ],
                }
            )

    with (args.root / "stage58_component_qualification.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = (
            "case",
            "component",
            "relative_history_error",
            "two_sem_relative",
            "qualification_ratio",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in CASE_NAMES:
            for row in results.get(name, {}).get("component_qualification", []):
                writer.writerow({"case": name, **row})

    if results:
        valid = {name: item for name, item in results.items() if "errors" in item}
        if valid:
            _plot_error_summary(args.root / "stage58_generalization_errors.png", valid)
            _plot_heat_flux(args.root / "stage58_heat_flux_histories.png", loaded, valid)

    lines = [
        "# Stage 58: prospective blind generalization",
        "",
        f"Qualification objective: **{'PASS' if qualification_pass else 'FAIL'}**",
        "",
        "The four-population closure and case registry were frozen before QMC evaluation. ",
        "QMC is an internal reference for the implemented collision operator, not MD/DSMC validation.",
        "",
        "| Case | Role | q error | Full third | Trace-free | Time change | QMC spread | Result |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CASE_NAMES:
        result = results.get(name)
        if result is None or "errors" not in result:
            lines.append(f"| {LABELS[name]} | missing | -- | -- | -- | -- | -- | FAIL |")
            continue
        lines.append(
            f"| {LABELS[name]} | {result['role']} | "
            f"{100.0 * result['errors']['heat_flux']:.3f}% | "
            f"{100.0 * result['errors']['third_tensor']:.3f}% | "
            f"{100.0 * result['errors']['trace_free']:.3f}% | "
            f"{100.0 * result['time_change']:.3f}% | "
            f"{100.0 * result['qmc_scramble_spread']:.3f}% | "
            f"{'PASS' if result['pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Stage-57 anchor: **{'PASS' if anchor_pass else 'FAIL'}**",
            f"All four prospective blind cases: **{'PASS' if blind_pass else 'FAIL'}**",
            "",
        ]
    )
    (args.root / "STAGE58_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    _write_bundle(args.root, args.bundle)
    digest = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    checksum = args.bundle.with_name(args.bundle.name + ".sha256.txt")
    checksum.write_text(f"{digest}  {args.bundle.name}\n", encoding="utf-8")
    print(
        f"[stage58] bundle={args.bundle} qualification_pass={qualification_pass}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the aggregate table and figure for the Stage-2 V4 OOD suite."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cubic_operator import OOD_SUITE_CASES, case_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def case_row(run_root: Path, case: str) -> dict[str, Any]:
    case_dir = run_root / case
    status = load_json(case_dir / "case_status.json") or {}
    candidates = [case_dir / "final" / "metrics.json", case_dir / "training" / "metrics.json"]
    metrics_path = next((path for path in candidates if path.is_file()), None)
    metrics = load_json(metrics_path) if metrics_path else None
    spec = case_spec(case)
    row: dict[str, Any] = {
        "case": case,
        "description": spec.description,
        "family": spec.family,
        "nu": spec.nu,
        "initial_qx": spec.heat_flux_qx,
        "variance_x": spec.variances[0],
        "variance_y": spec.variances[1],
        "variance_z": spec.variances[2],
        "state": status.get("state", "MISSING" if metrics is None else "COMPLETED"),
        "exit_code": status.get("exit_code"),
        "gate_passed": bool(metrics and metrics.get("gate_passed", False)),
        "publication_target_passed": bool(
            metrics and metrics.get("publication_target_passed", False)
        ),
        "marginal_relative_l2": None,
        "heat_flux_active_axis_relative_l2": None,
        "stress_history_relative_l2": None,
        "max_mass_error": None,
        "max_momentum_norm": None,
        "max_energy_error": None,
    }
    if metrics:
        for key in (
            "marginal_relative_l2",
            "heat_flux_active_axis_relative_l2",
            "stress_history_relative_l2",
            "max_mass_error",
            "max_momentum_norm",
            "max_energy_error",
        ):
            value = metrics.get(key)
            row[key] = float(value) if value is not None else None
        if spec.heat_flux_qx == 0.0:
            row["heat_flux_active_axis_relative_l2"] = None
        if np.allclose(spec.variances, (1.0, 1.0, 1.0)):
            row["stress_history_relative_l2"] = None
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_or_nan(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [np.nan if row[key] is None else float(row[key]) for row in rows],
        dtype=np.float64,
    )


def make_figure(output: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["case"].replace("ood_", "") for row in rows]
    x = np.arange(len(rows))
    q_error = finite_or_nan(rows, "heat_flux_active_axis_relative_l2")
    stress_error = finite_or_nan(rows, "stress_history_relative_l2")
    marginal = finite_or_nan(rows, "marginal_relative_l2")
    conservation_components = np.vstack(
        [
            finite_or_nan(rows, "max_mass_error"),
            finite_or_nan(rows, "max_momentum_norm"),
            finite_or_nan(rows, "max_energy_error"),
        ]
    )
    conservation = np.asarray(
        [
            np.nan if np.all(np.isnan(column)) else np.nanmax(column)
            for column in conservation_components.T
        ]
    )

    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    axes[0, 0].bar(x - 0.18, q_error, width=0.36, label=r"heat flux $Q_x$")
    axes[0, 0].bar(x + 0.18, stress_error, width=0.36, label="stress")
    axes[0, 0].axhline(0.05, color="black", ls="--", lw=1, label="5% target")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(title="Moment-history relative error", ylabel="relative L2")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].bar(x, marginal, color="#4c78a8")
    axes[0, 1].axhline(0.15, color="black", ls="--", lw=1)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Distribution marginal error", ylabel="relative L2")

    axes[1, 0].bar(x, conservation, color="#f58518")
    axes[1, 0].axhline(0.04, color="black", ls="--", lw=1)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="Worst conservation error", ylabel="absolute error")

    colors = ["#54a24b" if row["gate_passed"] else "#e45756" for row in rows]
    axes[1, 1].bar(x, [1.0] * len(rows), color=colors)
    axes[1, 1].set(ylim=(0, 1.15), yticks=[], title="Independent validation gate")
    axes[1, 1].text(0.02, 0.92, "green = pass; red = fail/missing", transform=axes[1, 1].transAxes)

    for axis in axes.flat:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    fig.suptitle("Stage-2 V4: homogeneous cubic Fokker–Planck OOD suite", fontsize=13)
    fig.savefig(output / "stage2_v4_ood_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "stage2_v4_ood_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [case_row(run_root, case) for case in OOD_SUITE_CASES]
    write_csv(output / "stage2_v4_ood_summary.csv", rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_cases": list(OOD_SUITE_CASES),
        "completed_cases": sum(row["state"] == "COMPLETED" for row in rows),
        "gate_passed_cases": sum(bool(row["gate_passed"]) for row in rows),
        "all_cases_completed": all(row["state"] == "COMPLETED" for row in rows),
        "all_gates_passed": all(bool(row["gate_passed"]) for row in rows),
        "particle_data_used_in_training": False,
        "rows": rows,
    }
    (output / "stage2_v4_ood_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    make_figure(output, rows)
    print("OOD_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

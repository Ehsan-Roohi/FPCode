#!/usr/bin/env python3
"""Collect independent Stage-53 epsilon tasks without discarding partial results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .run_boundary_realizability import METHOD_LABELS, METHODS, METHOD_STYLES, configuration


def collect(result_root: Path, mode: str = "unity") -> dict[str, object]:
    expected = configuration(mode).epsilons
    records: list[dict[str, object]] = []
    boundary: list[dict[str, float]] = []
    tasks: list[dict[str, object]] = []

    for index, expected_epsilon in enumerate(expected):
        task_root = result_root / f"epsilon_{index}"
        summary_path = task_root / "stage53_summary.json"
        if not summary_path.is_file():
            tasks.append(
                {
                    "epsilon_index": index,
                    "expected_epsilon": expected_epsilon,
                    "status": "MISSING",
                    "summary": str(summary_path.relative_to(result_root)),
                }
            )
            continue
        try:
            summary = json.loads(summary_path.read_text())
            actual = float(summary["boundary_family"]["epsilon"][0])
            if not np.isclose(actual, expected_epsilon, rtol=0.0, atol=1.0e-15):
                raise ValueError(
                    f"expected epsilon {expected_epsilon:g}, found {actual:g}"
                )
            records.extend(summary["records"])
            boundary.append(
                {
                    "epsilon": actual,
                    "initial_H2_margin": float(
                        summary["boundary_family"]["initial_H2_margin"][0]
                    ),
                    "initial_H4_margin": float(
                        summary["boundary_family"]["initial_H4_margin"][0]
                    ),
                }
            )
            tasks.append(
                {
                    "epsilon_index": index,
                    "expected_epsilon": expected_epsilon,
                    "status": summary["overall_status"],
                    "summary": str(summary_path.relative_to(result_root)),
                }
            )
        except Exception as error:
            tasks.append(
                {
                    "epsilon_index": index,
                    "expected_epsilon": expected_epsilon,
                    "status": "INVALID",
                    "summary": str(summary_path.relative_to(result_root)),
                    "message": f"{type(error).__name__}: {error}",
                }
            )

    complete = len(records) == len(expected) * len(METHODS)
    overall = (
        "INCOMPLETE"
        if not complete
        else "PASS"
        if all(record["status"] == "PASS" for record in records)
        else "HOLD"
    )
    monotone_h2 = (
        complete
        and len(boundary) == len(expected)
        and all(
            boundary[offset + 1]["initial_H2_margin"]
            < boundary[offset]["initial_H2_margin"]
            for offset in range(len(boundary) - 1)
        )
    )
    aggregate: dict[str, object] = {
        "stage": 53,
        "execution": "independent epsilon job array",
        "overall_status": overall,
        "complete": complete,
        "expected_epsilons": list(expected),
        "H2_strictly_approaches_zero": monotone_h2 if complete else None,
        "tasks": tasks,
        "boundary_family": boundary,
        "records": records,
    }
    (result_root / "stage53_array_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    _write_report(result_root, aggregate)
    if records:
        _write_overview(result_root, records)
    return aggregate


def _write_report(result_root: Path, aggregate: dict[str, object]) -> None:
    lines = [
        "# Stage 53 array results",
        "",
        f"Overall predeclared status: **{aggregate['overall_status']}**.",
        "",
        "Each boundary parameter was run and saved as an independent Slurm array task.",
        "",
        "| epsilon | arm | min H2 | min H4 | conservation | dt refinement | status |",
        "|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for record in aggregate["records"]:
        lines.append(
            f"| {record['epsilon']:.3g} | {record['method_label']} | "
            f"{record['minimum_H2_margin']:.3e} | "
            f"{record['minimum_H4_margin']:.3e} | "
            f"{record['maximum_conservation_error']:.3e} | "
            f"{record['coarse_fine_relative_field_L2']:.3e} | "
            f"{record['status']} |"
        )
    lines.extend(["", "## Task accounting", ""])
    for task in aggregate["tasks"]:
        lines.append(
            f"- index {task['epsilon_index']}, epsilon={task['expected_epsilon']:g}: "
            f"{task['status']}"
        )
    lines.extend(
        [
            "",
            "No smoothing, curve fitting, moment replacement, or gate relaxation was used.",
        ]
    )
    (result_root / "STAGE53_ARRAY_RESULTS.md").write_text("\n".join(lines) + "\n")


def _write_overview(result_root: Path, records: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    metrics = (
        ("minimum_H2_margin", "minimum normalized H2 eigenvalue", False),
        ("minimum_H4_margin", "minimum normalized H4 eigenvalue", False),
        ("maximum_conservation_error", "maximum conservation error", True),
        ("coarse_fine_relative_field_L2", "coarse/fine relative field L2", True),
    )
    for method in METHODS:
        method_records = sorted(
            (item for item in records if item["method"] == method),
            key=lambda item: float(item["epsilon"]),
            reverse=True,
        )
        if not method_records:
            continue
        color, linestyle, marker = METHOD_STYLES[method]
        for axis, (key, label, logarithmic) in zip(axes.ravel(), metrics):
            values = [float(item[key]) for item in method_records]
            if logarithmic:
                values = [max(value, 1.0e-18) for value in values]
            axis.plot(
                [float(item["epsilon"]) for item in method_records],
                values,
                color=color,
                ls=linestyle,
                marker=marker,
                lw=2,
                ms=6,
                label=METHOD_LABELS[method],
            )
            axis.set_ylabel(label)
            if logarithmic:
                axis.set_yscale("log")
    for axis in axes.ravel():
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel(r"boundary parameter $\epsilon \to 0^+$")
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(result_root / "stage53_array_overview.png", dpi=220, bbox_inches="tight")
    fig.savefig(result_root / "stage53_array_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "unity"), default="unity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect(args.result_root, args.mode)
    print(
        f"[stage53-collect] status={result['overall_status']} root={args.result_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()

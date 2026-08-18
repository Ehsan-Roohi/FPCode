#!/usr/bin/env python3
"""Calibrate a predictive kinetic-activation sensor on the Stage-10 ensemble.

The sensor is deliberately computable from the retained 35 moments and the
two already-available algebraic reconstructions.  The generating Gaussian
mixture is used only offline to label whether *both* algebraic closures miss
the instantaneous fourth-order cubic-FP source by more than the requested
accuracy gate.

This is an in-sample synthetic calibration, not a claim that the generating
mixture tail is nature's unique answer.  Stage 17 supplies the complementary
identifiability result: several positive tails can share the same 35 moments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage19")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    WeightedNodeTailClosure,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
    realizability_margin_35,
    reconstruct_gaussian_mixture_quadrature,
    reconstruct_grad_hyqmom_quadrature,
)
from hyqmom_fp.moments import HYQMOM_35_INDICES, central_moment  # noqa: E402
from riemann35_patch.stage10.run_general_realizability_audit import (  # noqa: E402
    ExactMixtureTailClosure,
    TAIL_INDICES,
    deterministic_states,
    random_state,
    source_from_closure,
    tail_vector,
)


FOURTH_POSITIONS = np.asarray(
    [sum(index) == 4 for index in HYQMOM_35_INDICES], dtype=bool
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-states", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--accuracy-gate", type=float, default=0.03)
    parser.add_argument("--target-recall", type=float, default=0.95)
    return parser.parse_args()


def _fourth_source_error(
    approximation: np.ndarray, exact: np.ndarray, physical_scale: float
) -> float:
    difference = approximation[FOURTH_POSITIONS] - exact[FOURTH_POSITIONS]
    denominator = max(
        float(np.linalg.norm(exact[FOURTH_POSITIONS])),
        1.0e-8 * physical_scale,
    )
    return float(np.linalg.norm(difference) / denominator)


def _symmetric_relative_difference(
    left: np.ndarray, right: np.ndarray, floor: float
) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(0.5 * (np.linalg.norm(left) + np.linalg.norm(right)), floor)
    )


def _non_gaussian_invariants(moments: np.ndarray) -> tuple[float, float]:
    """Return rotation-invariant standardized third and fourth cumulant norms."""

    state = macroscopic_state(moments)
    rho = state.rho
    theta = state.theta
    third = np.zeros((3, 3, 3))
    fourth_cumulant = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                powers = tuple(
                    int(i == axis) + int(j == axis) + int(k == axis)
                    for axis in range(3)
                )
                third[i, j, k] = central_moment(moments, powers) / rho
                for ell in range(3):
                    powers4 = tuple(
                        int(i == axis)
                        + int(j == axis)
                        + int(k == axis)
                        + int(ell == axis)
                        for axis in range(3)
                    )
                    raw4 = central_moment(moments, powers4) / rho
                    gaussian4 = (
                        state.covariance[i, j] * state.covariance[k, ell]
                        + state.covariance[i, k] * state.covariance[j, ell]
                        + state.covariance[i, ell] * state.covariance[j, k]
                    )
                    fourth_cumulant[i, j, k, ell] = raw4 - gaussian4
    return (
        float(np.linalg.norm(third) / theta**1.5),
        float(np.linalg.norm(fourth_cumulant) / theta**2),
    )


def audit_state(task) -> dict[str, object]:
    state, accuracy_gate = task
    moments = mixture_of_gaussians_moments_35(state.components)
    macro = macroscopic_state(moments)
    physical_source_scale = macro.rho * macro.theta**2
    exact_closure = ExactMixtureTailClosure(state.components)
    exact_source, exact_coefficients = source_from_closure(moments, exact_closure)
    exact_tail = tail_vector(exact_closure, moments)

    method_data: dict[str, dict[str, object]] = {}
    closures = {}
    for name, builder in (
        ("stage9", reconstruct_gaussian_mixture_quadrature),
        ("grad", reconstruct_grad_hyqmom_quadrature),
    ):
        try:
            quadrature = builder(moments)
            closure = WeightedNodeTailClosure(quadrature.nodes, quadrature.weights)
            source, coefficients = source_from_closure(moments, closure)
            tail = tail_vector(closure, moments)
            method_data[name] = {
                "status": "PASS",
                "fourth_source_relative_error": _fourth_source_error(
                    source, exact_source, physical_source_scale
                ),
                "full_source_relative_error": float(
                    np.linalg.norm(source - exact_source)
                    / max(np.linalg.norm(exact_source), 1.0e-8 * physical_source_scale)
                ),
                "tail_relative_error": float(
                    np.linalg.norm(tail - exact_tail)
                    / max(np.linalg.norm(exact_tail), 1.0e-12)
                ),
                "beta": float(coefficients.beta),
                "gamma_norm": float(np.linalg.norm(coefficients.gamma)),
            }
            closures[name] = (source, tail)
        except Exception as error:
            method_data[name] = {
                "status": "FAILED",
                "message": f"{type(error).__name__}: {error}",
                "fourth_source_relative_error": float("inf"),
                "full_source_relative_error": float("inf"),
                "tail_relative_error": float("inf"),
            }

    reconstruction_failure = len(closures) != 2
    if reconstruction_failure:
        source_disagreement = float("inf")
        tail_disagreement = float("inf")
    else:
        source_disagreement = _symmetric_relative_difference(
            closures["stage9"][0][FOURTH_POSITIONS],
            closures["grad"][0][FOURTH_POSITIONS],
            1.0e-8 * physical_source_scale,
        )
        tail_scales = np.asarray(
            [macro.rho * macro.theta ** (sum(index) / 2.0) for index in TAIL_INDICES]
        )
        stage9_tail = closures["stage9"][1] / tail_scales
        grad_tail = closures["grad"][1] / tail_scales
        tail_disagreement = _symmetric_relative_difference(
            stage9_tail, grad_tail, 1.0e-8
        )

    third_norm, fourth_cumulant_norm = _non_gaussian_invariants(moments)
    best_error = min(
        float(method_data["stage9"]["fourth_source_relative_error"]),
        float(method_data["grad"]["fourth_source_relative_error"]),
    )
    return {
        "name": state.name,
        "family": state.family,
        "components": len(state.components),
        "realizability_margin": float(realizability_margin_35(moments)),
        "exact_beta": float(exact_coefficients.beta),
        "exact_gamma_norm": float(np.linalg.norm(exact_coefficients.gamma)),
        "standardized_third_cumulant_norm": third_norm,
        "standardized_fourth_cumulant_norm": fourth_cumulant_norm,
        "fourth_source_disagreement": source_disagreement,
        "tail_disagreement": tail_disagreement,
        "reconstruction_failure": reconstruction_failure,
        "stage9_fourth_source_error": method_data["stage9"][
            "fourth_source_relative_error"
        ],
        "grad_fourth_source_error": method_data["grad"][
            "fourth_source_relative_error"
        ],
        "best_algebraic_fourth_source_error": best_error,
        "both_algebraic_closures_unsafe": bool(best_error > accuracy_gate),
        "stage9_status": method_data["stage9"]["status"],
        "grad_status": method_data["grad"]["status"],
    }


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    true_positive = int(np.sum(labels & predictions))
    false_positive = int(np.sum(~labels & predictions))
    true_negative = int(np.sum(~labels & ~predictions))
    false_negative = int(np.sum(labels & ~predictions))
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "recall": true_positive / max(true_positive + false_negative, 1),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "false_positive_rate": false_positive / max(false_positive + true_negative, 1),
        "active_fraction": float(np.mean(predictions)),
    }


def threshold_candidates(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    return np.unique(
        np.concatenate(
            ([0.0], np.quantile(finite, np.linspace(0.0, 1.0, 121)), [np.inf])
        )
    )


def calibrate_rules(
    rows: list[dict[str, object]], target_recall: float
) -> dict[str, object]:
    labels = np.asarray(
        [row["both_algebraic_closures_unsafe"] for row in rows], dtype=bool
    )
    failure = np.asarray([row["reconstruction_failure"] for row in rows], dtype=bool)
    disagreement = np.asarray(
        [row["fourth_source_disagreement"] for row in rows], dtype=float
    )
    tail_disagreement = np.asarray(
        [row["tail_disagreement"] for row in rows], dtype=float
    )

    def select_best(candidates):
        feasible = []
        for specification, predictions in candidates:
            statistics = confusion(labels, predictions)
            if statistics["recall"] >= target_recall:
                key = (
                    statistics["active_fraction"],
                    statistics["false_positive_rate"],
                    -statistics["precision"],
                )
                feasible.append((key, specification, statistics, predictions))
        if not feasible:
            raise RuntimeError("no sensor rule reached the requested recall")
        _, specification, statistics, predictions = min(feasible, key=lambda item: item[0])
        return specification, statistics, predictions

    source_candidates = []
    for threshold in threshold_candidates(disagreement):
        predictions = failure | (disagreement >= threshold)
        source_candidates.append(
            ({"source_disagreement_threshold": float(threshold)}, predictions)
        )
    source_spec, source_statistics, source_predictions = select_best(source_candidates)

    combined_candidates = []
    for source_threshold in threshold_candidates(disagreement):
        for tail_threshold in threshold_candidates(tail_disagreement):
            predictions = (
                failure
                | (disagreement >= source_threshold)
                | (tail_disagreement >= tail_threshold)
            )
            combined_candidates.append(
                (
                    {
                        "source_disagreement_threshold": float(source_threshold),
                        "tail_disagreement_threshold": float(tail_threshold),
                    },
                    predictions,
                )
            )
    combined_spec, combined_statistics, combined_predictions = select_best(
        combined_candidates
    )

    for row, source_prediction, combined_prediction in zip(
        rows, source_predictions, combined_predictions
    ):
        row["source_sensor_active"] = bool(source_prediction)
        row["combined_sensor_active"] = bool(combined_prediction)

    return {
        "unsafe_state_count": int(np.sum(labels)),
        "safe_state_count": int(np.sum(~labels)),
        "source_disagreement_rule": {
            **source_spec,
            **source_statistics,
        },
        "source_or_tail_disagreement_rule": {
            **combined_spec,
            **combined_statistics,
        },
        "calibration_scope": "in-sample on Gaussian-mixture-generated Stage-10 states; thresholds require confirmation on held-out kinetic/DVM states",
    }


def family_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    for family in sorted({str(row["family"]) for row in rows}):
        local = [row for row in rows if row["family"] == family]
        labels = np.asarray(
            [row["both_algebraic_closures_unsafe"] for row in local], dtype=bool
        )
        result[family] = {
            "states": len(local),
            "unsafe": int(np.sum(labels)),
            "source_sensor_active": int(
                np.sum([row["source_sensor_active"] for row in local])
            ),
            "combined_sensor_active": int(
                np.sum([row["combined_sensor_active"] for row in local])
            ),
            "median_best_algebraic_error": float(
                np.median([row["best_algebraic_fourth_source_error"] for row in local])
            ),
        }
    return result


def make_plot(path: Path, rows: list[dict[str, object]], threshold: float) -> None:
    import matplotlib.pyplot as plt

    errors = np.asarray(
        [row["best_algebraic_fourth_source_error"] for row in rows], dtype=float
    )
    disagreement = np.asarray(
        [row["fourth_source_disagreement"] for row in rows], dtype=float
    )
    finite = np.isfinite(disagreement) & np.isfinite(errors)
    plot_disagreement = np.maximum(disagreement, 1.0e-6)
    plot_errors = np.maximum(errors, 1.0e-6)
    unsafe = np.asarray(
        [row["both_algebraic_closures_unsafe"] for row in rows], dtype=bool
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    axis.scatter(
        plot_disagreement[finite & ~unsafe],
        plot_errors[finite & ~unsafe],
        s=18,
        facecolors="none",
        edgecolors="#0077bb",
        linewidths=0.8,
        label="At least one algebraic closure within 3%",
    )
    axis.scatter(
        plot_disagreement[finite & unsafe],
        plot_errors[finite & unsafe],
        s=20,
        color="#cc3311",
        alpha=0.78,
        label="Both algebraic closures above 3%",
    )
    axis.axvline(threshold, color="black", linestyle="--", linewidth=1.0)
    axis.axhline(0.03, color="0.35", linestyle=":", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Stage-9 / Grad fourth-source disagreement")
    axis.set_ylabel("Oracle best algebraic fourth-source error")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7.6)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = arguments()
    states = deterministic_states() + [
        random_state(index, args.seed) for index in range(args.random_states)
    ]
    tasks = [(state, args.accuracy_gate) for state in states]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(audit_state, tasks))
    calibration = calibrate_rules(rows, args.target_recall)
    families = family_summary(rows)
    summary = {
        "schema": "riemann35-stage19-predictive-sensor-calibration-v1",
        "state_count": len(rows),
        "accuracy_label": "unsafe only when both Stage-9 and Grad exceed the fourth-order source error gate",
        "accuracy_gate": args.accuracy_gate,
        "sensor_inputs_available_online": [
            "Stage-9/Grad fourth-order source disagreement",
            "Stage-9/Grad scaled M5/M6 tail disagreement",
            "reconstruction failure",
        ],
        "offline_label_only": "exact tail of the generating Gaussian mixture",
        "calibration": calibration,
        "families": families,
        "stage17_relation": "Stage 17 proves non-identifiability and supplies a compact-support source-span anchor. This Stage-19 calibration measures whether closure disagreement is a useful online surrogate; it does not turn the generating mixture into a unique truth.",
        "microstate_initialization_rule": "A newly active microstate must be inherited from an active neighbor or an incoming kinetic boundary, or initialized while a physically known decomposition is still available. It must not be synthesized from the same 35 moments after tail information has already been lost.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stage19_sensor_calibration.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    with (args.output / "stage19_sensor_states.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_rule = calibration["source_disagreement_rule"]
    make_plot(
        args.output / "stage19_sensor_calibration.png",
        rows,
        source_rule["source_disagreement_threshold"],
    )
    lines = [
        "# Stage 19: predictive kinetic-activation sensor calibration",
        "",
        "The online sensor compares the already-available Stage-9 and Grad/GQMOM cubic-FP sources. The offline label uses the exact tail of each generating Gaussian mixture and marks a state unsafe only when both algebraic closures exceed the 3% fourth-order source gate. The generating mixture is a controlled audit truth, not a claim of unique identifiability.",
        "",
        f"The ensemble contains {len(rows)} states, of which {calibration['unsafe_state_count']} are unsafe for both algebraic closures.",
        "",
        "| Rule | Threshold(s) | Recall | Precision | False-positive rate | Active fraction |",
        "|---|---|---:|---:|---:|---:|",
        f"| fourth-source disagreement | d >= {source_rule['source_disagreement_threshold']:.5g} | {source_rule['recall']:.1%} | {source_rule['precision']:.1%} | {source_rule['false_positive_rate']:.1%} | {source_rule['active_fraction']:.1%} |",
    ]
    combined = calibration["source_or_tail_disagreement_rule"]
    lines.append(
        f"| source OR tail disagreement | d >= {combined['source_disagreement_threshold']:.5g} or t >= {combined['tail_disagreement_threshold']:.5g} | {combined['recall']:.1%} | {combined['precision']:.1%} | {combined['false_positive_rate']:.1%} | {combined['active_fraction']:.1%} |"
    )
    lines.extend(
        [
            "",
            "These are in-sample thresholds on the 292-state synthetic Gaussian-mixture ensemble. They are suitable for selecting the sensor form, but not yet publication-level operating thresholds. Held-out DVM/kinetic states and a spatial false-positive/cost study remain required.",
            "",
            "Microstate activation is causal: the kinetic state must be inherited from an active neighbor or kinetic inflow, or initialized while a known physical population decomposition is still available. Reconstructing it from the same 35 moments after the alarm would reproduce the non-identifiability exposed by Stage 17.",
        ]
    )
    (args.output / "STAGE19_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

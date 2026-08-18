#!/usr/bin/env python3
"""Stage 20: audit causal positive tail memory with sensor hysteresis."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage20")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    ActivationHysteresis,
    adaptive_tail_memory_fp_step,
    initialize_adaptive_tail_memory,
    kinetic_activation_sensor,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
    positive_microstate_from_components,
    realizability_margin_35,
)
from riemann35_patch.stage10.run_general_realizability_audit import (  # noqa: E402
    deterministic_states,
)
from riemann35_patch.stage11.run_particle_validation import (  # noqa: E402
    POSITION,
    SELECTED_CASES,
    error_summary,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage11", type=Path, required=True)
    parser.add_argument("--stage14", type=Path, required=True)
    parser.add_argument(
        "--stage19-states",
        type=Path,
        default=REPOSITORY_ROOT
        / "results/riemann35_stage19/stage19_sensor_states.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--sensor-every", type=int, default=10)
    parser.add_argument("--points-per-component", type=int, default=1024)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_810)
    parser.add_argument("--source-on", type=float, default=0.10124)
    parser.add_argument("--source-off", type=float, default=0.05062)
    parser.add_argument("--tail-on", type=float, default=0.41005)
    parser.add_argument("--tail-off", type=float, default=0.205025)
    parser.add_argument("--skew-on", type=float, default=1.0e-3)
    parser.add_argument("--skew-off", type=float, default=5.0e-4)
    parser.add_argument("--release-hold", type=int, default=8)
    parser.add_argument("--minimum-active-steps", type=int, default=20)
    return parser.parse_args()


def _policy(args: argparse.Namespace) -> ActivationHysteresis:
    return ActivationHysteresis(
        source_on=args.source_on,
        source_off=args.source_off,
        tail_on=args.tail_on,
        tail_off=args.tail_off,
        skew_on=args.skew_on,
        skew_off=args.skew_off,
        release_hold_steps=args.release_hold,
        minimum_active_steps=args.minimum_active_steps,
    )


def run_case(task: tuple) -> dict[str, object]:
    (
        name,
        components,
        replicate,
        initially_active,
        policy,
        points_per_component,
        dt,
        steps,
        sample_every,
        sensor_every,
        seed,
    ) = task
    target = mixture_of_gaussians_moments_35(components)
    projection_residual = 0.0
    minimum_probability = 1.0
    candidate = None
    if initially_active:
        candidate, target, projection = positive_microstate_from_components(
            components,
            points_per_component=points_per_component,
            seed=seed,
            provenance=f"known-initial-mixture:{name}:replicate-{replicate}",
        )
        projection_residual = projection.relative_moment_residual
        minimum_probability = projection.minimum_probability
    adaptive = initialize_adaptive_tail_memory(
        target,
        candidate_microstate=candidate,
        hysteresis=policy,
        noise_seed=seed + 1_000_003,
    )
    initial_macro = macroscopic_state(target)
    history = [target.copy()]
    mode_samples = [int(adaptive.mode == "micro")]
    sensor_samples = [
        (
            adaptive.sensor_reading.fourth_source_disagreement,
            adaptive.sensor_reading.tail_disagreement,
        )
    ]
    active_steps = 0
    blocked_steps = 0
    sensor_evaluations = 1
    minimum_margin = float(realizability_margin_35(target))
    maximum_mass_drift = 0.0
    maximum_momentum_drift = 0.0
    maximum_temperature_drift = 0.0
    transitions: list[dict[str, object]] = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        adaptive, diagnostics = adaptive_tail_memory_fp_step(
            adaptive,
            dt,
            1.0,
            hysteresis=policy,
            sensor_interval_steps=sensor_every,
        )
        active_steps += int(diagnostics.used_micro_step)
        blocked_steps += int(diagnostics.activation_blocked)
        sensor_evaluations += int(diagnostics.sensor_evaluated)
        minimum_margin = min(minimum_margin, diagnostics.realizability_margin)
        if diagnostics.transition not in ("macro->macro", "micro->micro"):
            transitions.append(
                {
                    "step": step,
                    "time_over_tau": step * dt,
                    "transition": diagnostics.transition,
                }
            )
        macro = macroscopic_state(adaptive.moments)
        maximum_mass_drift = max(
            maximum_mass_drift, abs(macro.rho - initial_macro.rho)
        )
        maximum_momentum_drift = max(
            maximum_momentum_drift,
            float(np.linalg.norm(macro.velocity - initial_macro.velocity)),
        )
        maximum_temperature_drift = max(
            maximum_temperature_drift,
            abs(macro.theta - initial_macro.theta),
        )
        if step % sample_every == 0 or step == steps:
            history.append(adaptive.moments.copy())
            mode_samples.append(int(adaptive.mode == "micro"))
            sensor_samples.append(
                (
                    adaptive.sensor_reading.fourth_source_disagreement,
                    adaptive.sensor_reading.tail_disagreement,
                )
            )
    micro_to_macro = sum(
        item["transition"] == "micro->macro" for item in transitions
    )
    macro_to_micro = sum(
        item["transition"] == "macro->micro" for item in transitions
    )
    return {
        "case": name,
        "replicate": replicate,
        "initially_active": initially_active,
        "history": np.asarray(history),
        "mode_samples": np.asarray(mode_samples, dtype=int),
        "sensor_samples": np.asarray(sensor_samples, dtype=float),
        "active_steps": active_steps,
        "active_fraction": active_steps / steps,
        "blocked_steps": blocked_steps,
        "sensor_evaluations": sensor_evaluations,
        "transitions": transitions,
        "micro_to_macro_transitions": micro_to_macro,
        "macro_to_micro_transitions": macro_to_micro,
        "chatter_events": max(micro_to_macro + macro_to_micro - 1, 0),
        "final_mode": adaptive.mode,
        "tail_ambiguous_final": adaptive.tail_ambiguous,
        "projection_relative_residual": projection_residual,
        "minimum_probability": minimum_probability,
        "minimum_realizability_margin": minimum_margin,
        "maximum_mass_drift": maximum_mass_drift,
        "maximum_momentum_drift": maximum_momentum_drift,
        "maximum_temperature_drift": maximum_temperature_drift,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _scramble_spread(histories: np.ndarray, position: int) -> float:
    if histories.shape[0] < 2:
        return 0.0
    values = histories[:, :, position]
    return float(
        np.linalg.norm(np.std(values, axis=0, ddof=1))
        / max(np.linalg.norm(np.mean(values, axis=0)), 1.0e-14)
    )


def retrospective_stage19_metrics(
    path: Path, policy: ActivationHysteresis
) -> dict[str, object]:
    """Quantify the tradeoff introduced by the target-family skew gate."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    labels = np.asarray(
        [row["both_algebraic_closures_unsafe"].lower() == "true" for row in rows]
    )
    failures = np.asarray(
        [row["reconstruction_failure"].lower() == "true" for row in rows]
    )
    source = np.asarray(
        [float(row["fourth_source_disagreement"]) for row in rows]
    )
    tail = np.asarray([float(row["tail_disagreement"]) for row in rows])
    skew = np.asarray(
        [float(row["standardized_third_cumulant_norm"]) for row in rows]
    )

    def statistics(predictions: np.ndarray) -> dict[str, object]:
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
            "false_positive_rate": false_positive
            / max(false_positive + true_negative, 1),
            "active_fraction": float(np.mean(predictions)),
        }

    original = failures | (source >= policy.source_on) | (tail >= policy.tail_on)
    gated = (
        failures
        | ((source >= policy.source_on) & (skew >= policy.skew_on))
        | (tail >= policy.tail_on)
    )
    return {
        "state_count": len(rows),
        "original_source_or_tail_rule": statistics(original),
        "source_rule_with_skew_gate": statistics(gated),
        "interpretation": "retrospective disclosure only; Stage-19 synthetic states were used to select the original thresholds",
    }


def run_release_control(
    components,
    policy: ActivationHysteresis,
    *,
    dt: float,
    sensor_every: int,
    points_per_component: int,
    seed: int,
) -> dict[str, object]:
    """Exercise micro-to-macro release from a causal but already-safe state."""

    microstate, target, projection = positive_microstate_from_components(
        components,
        points_per_component=points_per_component,
        seed=seed,
        provenance="stage20-safe-causal-release-control",
    )
    adaptive = initialize_adaptive_tail_memory(
        target,
        candidate_microstate=microstate,
        hysteresis=policy,
        force_causal_birth=True,
        noise_seed=seed + 1_000_003,
    )
    maximum_steps = max(
        policy.minimum_active_steps,
        policy.release_hold_steps * sensor_every,
    ) + 2 * sensor_every
    transitions = []
    release_step = None
    for step in range(1, maximum_steps + 1):
        adaptive, diagnostics = adaptive_tail_memory_fp_step(
            adaptive,
            dt,
            1.0,
            hysteresis=policy,
            sensor_interval_steps=sensor_every,
        )
        if diagnostics.transition not in ("micro->micro", "macro->macro"):
            transitions.append(diagnostics.transition)
        if diagnostics.transition == "micro->macro":
            release_step = step
            break
    return {
        "case": "stage9_correlated_safe_causal_birth",
        "initial_mode": "micro",
        "final_mode": adaptive.mode,
        "release_step": release_step,
        "release_time_over_tau": None if release_step is None else release_step * dt,
        "transitions": transitions,
        "projection_relative_residual": projection.relative_moment_residual,
        "minimum_probability": projection.minimum_probability,
        "realizability_margin_at_release": float(
            realizability_margin_35(adaptive.moments)
        ),
        "passed": bool(
            adaptive.mode == "macro"
            and transitions == ["micro->macro"]
            and adaptive.microstate is None
        ),
    }


def _plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["case"]).replace("_", " ") for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].bar(
        positions - width / 2,
        [100.0 * float(row["stage9_M400_error_vs_particle"]) for row in rows],
        width,
        color="0.65",
        label="Stage 9",
    )
    axes[0].bar(
        positions + width / 2,
        [100.0 * float(row["adaptive_M400_error_vs_particle"]) for row in rows],
        width,
        color="#0077bb",
        label="adaptive tail memory",
    )
    axes[0].axhline(3.0, color="#cc3311", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel(r"$M_{400}$ history error vs particles (\%)")
    axes[0].set_xticks(positions, labels, rotation=32, ha="right")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=7.5)
    axes[1].bar(
        positions,
        [100.0 * float(row["micro_active_fraction"]) for row in rows],
        color="#117864",
    )
    axes[1].set_ylabel("micro collision steps (%)")
    axes[1].set_xticks(positions, labels, rotation=32, ha="right")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = arguments()
    if args.dt <= 0.0 or args.final_time <= 0.0:
        raise ValueError("time controls must be positive")
    steps = int(round(args.final_time / args.dt))
    if steps % args.sample_every:
        raise ValueError("sample_every must divide the requested step count")
    policy = _policy(args)
    states = {state.name: state for state in deterministic_states()}
    initial_readings = {}
    tasks = []
    for case in SELECTED_CASES:
        state = states[case]
        moments = mixture_of_gaussians_moments_35(state.components)
        reading = kinetic_activation_sensor(moments)
        active = policy.requests_activation(reading)
        initial_readings[case] = reading
        repeats = args.replicates if active else 1
        for replicate in range(repeats):
            tasks.append(
                (
                    case,
                    state.components,
                    replicate,
                    active,
                    policy,
                    args.points_per_component,
                    args.dt,
                    steps,
                    args.sample_every,
                    args.sensor_every,
                    args.seed + 15_485_863 * replicate,
                )
            )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(run_case, tasks))

    release_control = run_release_control(
        states["stage9_correlated"].components,
        policy,
        dt=args.dt,
        sensor_every=args.sensor_every,
        points_per_component=args.points_per_component,
        seed=args.seed + 91_427_661,
    )
    stage19_retrospective = retrospective_stage19_metrics(
        args.stage19_states, policy
    )

    particle_archive = np.load(
        args.stage11 / "stage11_particle_seed_histories.npz"
    )
    closure_archive = np.load(
        args.stage11 / "stage11_closure_histories_base_and_half_dt.npz"
    )
    qmc_archive = np.load(
        args.stage14 / "stage14_qmc_scramble_histories.npz"
    )
    rows: list[dict[str, object]] = []
    histories_for_archive = {}
    case_details = {}
    for case in SELECTED_CASES:
        state = states[case]
        local = [result for result in results if result["case"] == case]
        replicate_histories = np.asarray([result["history"] for result in local])
        adaptive_mean = np.mean(replicate_histories, axis=0)
        sample_count = adaptive_mean.shape[0]
        initial = mixture_of_gaussians_moments_35(state.components)
        particle_seeds = particle_archive[f"{case}_aligned_moments"][
            :, :sample_count
        ]
        particle_mean = np.mean(particle_seeds, axis=0)
        particle_sem = np.std(particle_seeds, axis=0, ddof=1) / np.sqrt(
            particle_seeds.shape[0]
        )
        stage9 = closure_archive[f"{case}_stage9_finite_map_base"][:sample_count]
        adaptive_error = error_summary(
            adaptive_mean, particle_mean, particle_sem, initial
        )
        stage9_error = error_summary(stage9, particle_mean, particle_sem, initial)
        adaptive_m400 = adaptive_error["physical_observables"]["M400"][
            "history_relative_l2"
        ]
        stage9_m400 = stage9_error["physical_observables"]["M400"][
            "history_relative_l2"
        ]
        qmc_m400 = None
        if f"{case}_mean" in qmc_archive.files:
            qmc_mean = qmc_archive[f"{case}_mean"][:sample_count]
            qmc_sem = qmc_archive[f"{case}_sem"][:sample_count]
            qmc_m400 = error_summary(
                adaptive_mean, qmc_mean, qmc_sem, initial
            )["physical_observables"]["M400"]["history_relative_l2"]
        spread = _scramble_spread(replicate_histories, POSITION[(4, 0, 0)])
        reading = initial_readings[case]
        row = {
            "case": case,
            "initial_source_disagreement": reading.fourth_source_disagreement,
            "initial_tail_disagreement": reading.tail_disagreement,
            "initial_standardized_skewness_norm": reading.standardized_skewness_norm,
            "initial_sensor_active": policy.requests_activation(reading),
            "replicates": len(local),
            "micro_active_fraction": float(
                np.mean([result["active_fraction"] for result in local])
            ),
            "blocked_activation_steps": int(
                sum(result["blocked_steps"] for result in local)
            ),
            "chatter_events": int(sum(result["chatter_events"] for result in local)),
            "transition_count": int(
                sum(len(result["transitions"]) for result in local)
            ),
            "adaptive_M400_error_vs_particle": float(adaptive_m400),
            "stage9_M400_error_vs_particle": float(stage9_m400),
            "adaptive_M400_error_vs_QMC": (
                None if qmc_m400 is None else float(qmc_m400)
            ),
            "M400_scramble_relative_spread": spread,
            "adaptive_all35_error_vs_particle": float(
                adaptive_error["all_35_dimensionless_history_relative_l2"]
            ),
            "stage9_all35_error_vs_particle": float(
                stage9_error["all_35_dimensionless_history_relative_l2"]
            ),
            "minimum_realizability_margin": float(
                min(result["minimum_realizability_margin"] for result in local)
            ),
            "maximum_projection_relative_residual": float(
                max(result["projection_relative_residual"] for result in local)
            ),
            "minimum_micro_probability": float(
                min(result["minimum_probability"] for result in local)
            ),
            "maximum_mass_drift": float(
                max(result["maximum_mass_drift"] for result in local)
            ),
            "maximum_momentum_drift": float(
                max(result["maximum_momentum_drift"] for result in local)
            ),
            "maximum_temperature_drift": float(
                max(result["maximum_temperature_drift"] for result in local)
            ),
            "elapsed_seconds_sum": float(
                sum(result["elapsed_seconds"] for result in local)
            ),
        }
        rows.append(row)
        case_details[case] = {
            "summary": row,
            "replicate_lifecycle": [
                {
                    key: value
                    for key, value in result.items()
                    if key
                    not in ("history", "mode_samples", "sensor_samples")
                }
                for result in local
            ],
        }
        histories_for_archive[f"{case}_adaptive_replicates"] = replicate_histories
        histories_for_archive[f"{case}_adaptive_mean"] = adaptive_mean
        histories_for_archive[f"{case}_mode_samples"] = np.asarray(
            [result["mode_samples"] for result in local]
        )
        histories_for_archive[f"{case}_sensor_samples"] = np.asarray(
            [result["sensor_samples"] for result in local]
        )

    rare_beam = next(row for row in rows if row["case"] == "rare_beam_ma20")
    rare_beam_envelope = max(
        float(rare_beam["adaptive_M400_error_vs_particle"]),
        float(rare_beam["adaptive_M400_error_vs_QMC"]),
        float(rare_beam["M400_scramble_relative_spread"]),
    )
    gates = {
        "rare_beam_3pct_reference_envelope": rare_beam_envelope < 0.03,
        "no_blocked_causal_activations": sum(
            int(row["blocked_activation_steps"]) for row in rows
        )
        == 0,
        "no_hysteresis_chatter": sum(int(row["chatter_events"]) for row in rows)
        == 0,
        "positive_realizable_histories": min(
            float(row["minimum_realizability_margin"]) for row in rows
        )
        >= -5.0e-13,
        "conservative_projection": max(
            float(row["maximum_projection_relative_residual"]) for row in rows
        )
        < 2.0e-8,
        "hysteretic_micro_to_macro_release_control": release_control["passed"],
    }
    summary = {
        "schema": "riemann35-stage20-hysteretic-tail-memory-v1",
        "method": "causal positive QMC microstate plus Stage-9 macro collision map",
        "controls": {
            "dt_over_tau": args.dt,
            "final_time_over_tau": args.final_time,
            "sensor_interval_steps": args.sensor_every,
            "points_per_known_component": args.points_per_component,
            "micro_replicates": args.replicates,
            "hysteresis": policy.__dict__,
            "birth_rule": "known initial physical mixture only; later activation requires an externally supplied inherited/inflow microstate",
            "macro_to_micro_projection": "positive entropy reweighting on the causal support, matching all 35 moments",
            "micro_to_macro_projection": "direct positive weighted moments through degree four",
        },
        "gates": gates,
        "overall_pass": all(gates.values()),
        "rare_beam_reference_envelope": rare_beam_envelope,
        "equal_case_average_micro_active_fraction": float(
            np.mean([row["micro_active_fraction"] for row in rows])
        ),
        "release_control": release_control,
        "stage19_retrospective": stage19_retrospective,
        "cases": case_details,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stage20_hysteresis_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    with (args.output / "stage20_case_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output / "stage20_hysteresis_histories.npz",
        **histories_for_archive,
    )
    _plot(args.output / "stage20_accuracy_and_activation.png", rows)

    release_time_text = (
        "--"
        if release_control["release_time_over_tau"] is None
        else f"{float(release_control['release_time_over_tau']):.3f}"
    )
    original_rule = stage19_retrospective["original_source_or_tail_rule"]
    gated_rule = stage19_retrospective["source_rule_with_skew_gate"]
    lines = [
        "# Stage 20: causal positive tail memory with hysteresis",
        "",
        "The Stage-19 disagreement sensor is now part of the solver rather than an offline script. A positive microstate may be born only from a known initial decomposition or a supplied causal donor. Entropy reweighting matches all 35 transported moments on that support; failure to match is explicit. Dropping an active microstate is the direct positive moment projection.",
        "",
        f"Sensor cadence is every {args.sensor_every} collision steps. The on/off source thresholds are {policy.source_on:.5g}/{policy.source_off:.5g}; the tail thresholds are {policy.tail_on:.5g}/{policy.tail_off:.5g}. A source alarm additionally requires standardized skewness above {policy.skew_on:.1e}; this excludes the symmetric counter-stream false alarm found in the lifecycle audit. Release requires {policy.release_hold_steps} consecutive safe sensor evaluations and at least {policy.minimum_active_steps} active collision steps.",
        "",
        "| Case | Sensor at t=0 | Micro steps | Stage-9 M400 error | Adaptive M400 error | QMC error | Scramble spread | Blocked | Chatter |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        qmc_text = (
            "--"
            if row["adaptive_M400_error_vs_QMC"] is None
            else f"{float(row['adaptive_M400_error_vs_QMC']):.2%}"
        )
        lines.append(
            f"| {row['case']} | {'ON' if row['initial_sensor_active'] else 'off'} | "
            f"{float(row['micro_active_fraction']):.1%} | "
            f"{float(row['stage9_M400_error_vs_particle']):.2%} | "
            f"{float(row['adaptive_M400_error_vs_particle']):.2%} | {qmc_text} | "
            f"{float(row['M400_scramble_relative_spread']):.2%} | "
            f"{row['blocked_activation_steps']} | {row['chatter_events']} |"
        )
    lines.extend(
        [
            "",
            f"The rare-beam reference-envelope maximum is {rare_beam_envelope:.2%}; the 3% gate is {'PASS' if gates['rare_beam_3pct_reference_envelope'] else 'FAIL'}. The equal-case average micro active fraction is {summary['equal_case_average_micro_active_fraction']:.1%}.",
            "",
            f"A separate safe-causal-birth control exercised the off path and released micro to macro at step {release_control['release_step']} (t/tau={release_time_text}) with exactly one transition.",
            "",
            f"Retrospectively applying the skew gate to the 292 Stage-19 synthetic states changes recall from {original_rule['recall']:.1%} to {gated_rule['recall']:.1%}, precision from {original_rule['precision']:.1%} to {gated_rule['precision']:.1%}, false-positive rate from {original_rule['false_positive_rate']:.1%} to {gated_rule['false_positive_rate']:.1%}, and active fraction from {original_rule['active_fraction']:.1%} to {gated_rule['active_fraction']:.1%}. This is a disclosed target-family tradeoff, not a new universal calibration.",
            "",
            "This is a homogeneous lifecycle gate. It validates causal birth, positive bidirectional projection, and the activation/release state machine. It does not yet validate spatial donor selection, kinetic transport across cell faces, or an independent DVM reference.",
        ]
    )
    (args.output / "STAGE20_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

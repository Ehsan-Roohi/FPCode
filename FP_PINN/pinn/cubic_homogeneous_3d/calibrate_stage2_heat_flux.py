#!/usr/bin/env python3
"""Physics-moment projection for the Stage-2 heat-flux PINN.

The canonical 9x9 closure fixes the homogeneous third-moment equation exactly:

    dQ/dt = -(4/3) nu Q.

Earlier training imposed this consequence only through a noisy Monte Carlo
projection of the pointwise PDE residual.  This program projects a portable
Stage-2 checkpoint onto the exact operator moment law using deterministic
whole-space Gauss-Hermite integration.  Particle data are opened only after
optimization, for the independent final validation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import tensorflow as tf

from cubic_operator import (
    ALL_CASE_NAMES,
    case_default_nu,
    case_has_heat_flux,
    case_is_axisymmetric_heat_flux,
    initial_heat_flux_qx,
)
from train_stage2 import (
    Config,
    DensityModel,
    deterministic_moment_tensors,
    evaluate,
    exact_heat_flux_target,
    gauss_hermite_velocity_quadrature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=ALL_CASE_NAMES, default="heat_flux")
    parser.add_argument("--resume-weights", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=2.0)
    parser.add_argument("--train-order", type=int, default=16)
    parser.add_argument("--heldout-order", type=int, default=20)
    parser.add_argument("--time-points", type=int, default=21)
    parser.add_argument("--check-every", type=int, default=25)
    parser.add_argument("--heat-flux-weight", type=float, default=100.0)
    parser.add_argument("--mass-weight", type=float, default=50.0)
    parser.add_argument("--momentum-weight", type=float, default=30.0)
    parser.add_argument("--energy-weight", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--correction-cap", type=float, default=12.0)
    parser.add_argument("--bridge-rate", type=float, default=1.0)
    parser.add_argument("--nu", type=float)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--evaluation-samples", type=int, default=131_072)
    parser.add_argument("--evaluation-quadrature-order", type=int, default=48)
    parser.add_argument("--marginal-quadrature-order", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--strict-gate", action="store_true")
    args = parser.parse_args()
    if not case_has_heat_flux(args.case):
        parser.error("--case must select a heat-flux or coupled benchmark")
    if args.nu is None:
        args.nu = case_default_nu(args.case)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.learning_rate <= 0.0 or args.gradient_clip_norm <= 0.0:
        parser.error("learning rate and gradient clip norm must be positive")
    if args.time_points < 3:
        parser.error("--time-points must be at least three")
    if args.check_every < 1:
        parser.error("--check-every must be positive")
    if (
        args.train_order < 4
        or args.heldout_order < 4
        or args.evaluation_quadrature_order < 4
    ):
        parser.error("quadrature orders must be at least four")
    return args


def build_model_config(args: argparse.Namespace, output: Path) -> Config:
    return Config(
        case=args.case,
        output_dir=str(output),
        reference=str(Path(args.reference).resolve()),
        width=args.width,
        depth=args.depth,
        correction_cap=args.correction_cap,
        bridge_rate=args.bridge_rate,
        nu=args.nu,
        tmax=args.tmax,
        axisymmetric_heat_flux=case_is_axisymmetric_heat_flux(args.case),
        antithetic_heat_flux_quadrature=True,
        evaluation_samples=args.evaluation_samples,
        evaluation_quadrature_order=args.evaluation_quadrature_order,
        marginal_quadrature_order=args.marginal_quadrature_order,
        seed=args.seed,
        evaluate_only=True,
    )


def physics_losses(
    model: DensityModel,
    times: tf.Tensor,
    velocity: tf.Tensor,
    log_weights: tf.Tensor,
    config: Config,
    args: argparse.Namespace,
) -> dict[str, tf.Tensor]:
    moments = deterministic_moment_tensors(model, times, velocity, log_weights)
    q0 = max(abs(initial_heat_flux_qx(config.case)), 1.0e-12)
    target = tf.cast(
        exact_heat_flux_target(times, nu=config.nu, initial_qx=q0), tf.float64
    )
    heat_flux = tf.reduce_mean(tf.square((moments["q"] - target) / q0))
    mass = tf.reduce_mean(tf.square(moments["mass"] - 1.0))
    momentum = tf.reduce_mean(tf.square(moments["mean"]))
    energy = tf.reduce_mean(tf.square(moments["dm2"] - 3.0))
    total = (
        args.heat_flux_weight * heat_flux
        + args.mass_weight * mass
        + args.momentum_weight * momentum
        + args.energy_weight * energy
    )
    return {
        "total": total,
        "heat_flux": heat_flux,
        "mass": mass,
        "momentum": momentum,
        "energy": energy,
    }


def physics_metrics(
    model: DensityModel,
    times: tf.Tensor,
    velocity: tf.Tensor,
    log_weights: tf.Tensor,
    config: Config,
) -> dict[str, float | bool]:
    moments = deterministic_moment_tensors(model, times, velocity, log_weights)
    target = exact_heat_flux_target(
        times, nu=config.nu, initial_qx=initial_heat_flux_qx(config.case)
    ).numpy().astype(np.float64)
    q = moments["q"].numpy()
    mass = moments["mass"].numpy()
    mean = moments["mean"].numpy()
    dm2 = moments["dm2"].numpy()
    active_error = float(
        np.linalg.norm(q[:, 0] - target[:, 0])
        / max(np.linalg.norm(target[:, 0]), 1.0e-15)
    )
    full_error = float(np.linalg.norm(q - target) / max(np.linalg.norm(target), 1.0e-15))
    max_mass = float(np.max(np.abs(mass - 1.0)))
    max_momentum = float(np.max(np.linalg.norm(mean, axis=1)))
    max_energy = float(np.max(np.abs(dm2 - 3.0)))
    admissible = max_mass < 0.01 and max_momentum < 0.01 and max_energy < 0.01
    score = (
        active_error
        + 2.0 * max_mass
        + max_momentum
        + 2.0 * max_energy
    )
    return {
        "physics_qx_relative_l2": active_error,
        "physics_q_relative_l2": full_error,
        "max_mass_error": max_mass,
        "max_momentum_norm": max_momentum,
        "max_energy_error": max_energy,
        "physics_admissible": admissible,
        "selection_score": score,
    }


def save_history(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "step", "elapsed_seconds", "training_total", "training_heat_flux",
        "training_mass", "training_momentum", "training_energy", "gradient_norm",
        "physics_qx_relative_l2", "physics_q_relative_l2", "max_mass_error",
        "max_momentum_norm", "max_energy_error", "physics_admissible",
        "selection_score",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_reproducibility_snapshot(output: Path) -> None:
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    here = Path(__file__).resolve().parent
    names = [
        "calibrate_stage2_heat_flux.py",
        "audit_stage2_residual.py",
        "train_stage2.py",
        "cubic_operator.py",
        "evaluate_stage2_checkpoints.py",
        "package_stage2.py",
        "HEAT_FLUX_V2.md",
        "README.md",
    ]
    for name in names:
        shutil.copy2(here / name, snapshot / name)
    shutil.copy2(
        here / "slurm" / "run_stage2_heatflux_v3.sbatch",
        snapshot / "run_stage2_heatflux_v3.sbatch",
    )
    shutil.copytree(here / "tests", snapshot / "tests", dirs_exist_ok=True)


def main() -> None:
    args = parse_args()
    tf.keras.backend.set_floatx("float32")
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints_h5"
    checkpoints.mkdir(exist_ok=True)
    resume = Path(args.resume_weights).resolve()
    reference = Path(args.reference).resolve()
    if not resume.is_file() or not reference.is_file():
        raise SystemExit("resume weights or independent reference file is missing")
    copied_resume = output / "resume_input.weights.h5"
    if resume.resolve() != copied_resume.resolve():
        shutil.copy2(resume, copied_resume)
    reference_dir = output / "reference_particle"
    reference_dir.mkdir(exist_ok=True)
    copied_reference = reference_dir / "reference.npz"
    if reference.resolve() != copied_reference.resolve():
        shutil.copy2(reference, copied_reference)

    config = build_model_config(args, output)
    config.reference = str(copied_reference)
    run_config = {
        "method": "deterministic_exact_operator_moment_projection",
        "particle_data_used_in_optimization": False,
        "operator_identity": "dQ/dt=-(4/3)*nu*Q",
        "projection": vars(args),
        "network": asdict(config),
    }
    (output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    model = DensityModel(config)
    model.log_density(tf.zeros((1, 1)), tf.zeros((1, 3)))
    model.load_weights(copied_resume)
    train_velocity, train_log_weights = gauss_hermite_velocity_quadrature(
        args.train_order
    )
    heldout_velocity, heldout_log_weights = gauss_hermite_velocity_quadrature(
        args.heldout_order
    )
    train_times = tf.constant(
        np.linspace(0.0, args.tmax, args.time_points)[:, None], tf.float32
    )
    midpoints = (np.arange(args.time_points - 1) + 0.5) / (args.time_points - 1)
    heldout_times = tf.constant(
        np.concatenate([args.tmax * midpoints, [args.tmax]])[:, None], tf.float32
    )

    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    @tf.function(reduce_retracing=True)
    def projection_step() -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            losses = physics_losses(
                model, train_times, train_velocity, train_log_weights, config, args
            )
        gradients = tape.gradient(losses["total"], model.trainable_variables)
        gradients = [
            tf.where(tf.math.is_finite(value), value, tf.zeros_like(value))
            for value in gradients
        ]
        gradients, norm = tf.clip_by_global_norm(gradients, args.gradient_clip_norm)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return {**losses, "gradient_norm": norm}

    rows: list[dict[str, Any]] = []
    baseline_physics = physics_metrics(
        model, heldout_times, heldout_velocity, heldout_log_weights, config
    )
    best_key = (
        not bool(baseline_physics["physics_admissible"]),
        float(baseline_physics["selection_score"]),
    )
    best_step = 0
    best_weights = output / "stage2_best.weights.h5"
    model.save_weights(best_weights)
    started = time.perf_counter()
    last_losses: dict[str, float] = {}

    for step in range(1, args.steps + 1):
        raw = projection_step()
        last_losses = {name: float(value.numpy()) for name, value in raw.items()}
        if step == 1 or step % args.check_every == 0 or step == args.steps:
            heldout = physics_metrics(
                model, heldout_times, heldout_velocity, heldout_log_weights, config
            )
            row = {
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                **{f"training_{key}": value for key, value in last_losses.items()
                   if key != "gradient_norm"},
                "gradient_norm": last_losses["gradient_norm"],
                **heldout,
            }
            rows.append(row)
            checkpoint = checkpoints / f"projection-step-{step:06d}.weights.h5"
            model.save_weights(checkpoint)
            key = (
                not bool(heldout["physics_admissible"]),
                float(heldout["selection_score"]),
            )
            if key < best_key:
                best_key = key
                best_step = step
                model.save_weights(best_weights)
            print(
                "PROJECTION "
                + json.dumps(
                    {
                        "step": step,
                        "qx_l2": heldout["physics_qx_relative_l2"],
                        "mass": heldout["max_mass_error"],
                        "energy": heldout["max_energy_error"],
                        "score": heldout["selection_score"],
                        "best_step": best_step,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_weights = output / "stage2_final.weights.h5"
    model.save_weights(final_weights)
    save_history(output / "projection_history.csv", rows)

    baseline_dir = output / "baseline_evaluation"
    baseline_dir.mkdir(exist_ok=True)
    baseline_model = DensityModel(config)
    baseline_model.log_density(tf.zeros((1, 1)), tf.zeros((1, 3)))
    baseline_model.load_weights(copied_resume)
    baseline_metrics = evaluate(baseline_model, config, baseline_dir)
    (baseline_dir / "metrics.json").write_text(
        json.dumps(baseline_metrics, indent=2, sort_keys=True) + "\n"
    )

    best_model = DensityModel(config)
    best_model.log_density(tf.zeros((1, 1)), tf.zeros((1, 3)))
    best_model.load_weights(best_weights)
    best_metrics = evaluate(best_model, config, output)
    validation = np.load(output / "validation.npz")
    reference_data = np.load(copied_reference)
    times = reference_data["time"]
    exact = initial_heat_flux_qx(config.case) * np.exp(
        -(4.0 / 3.0) * config.nu * times
    )
    particle_qx = reference_data["q"][:, 0]
    model_qx = validation["model_q"][:, 0]
    best_metrics.update(
        {
            "method": "deterministic_exact_operator_moment_projection",
            "particle_data_used_in_optimization": False,
            "best_projection_step": best_step,
            "baseline_heat_flux_active_axis_relative_l2": baseline_metrics[
                "heat_flux_active_axis_relative_l2"
            ],
            "particle_vs_exact_operator_qx_relative_l2": float(
                np.linalg.norm(particle_qx - exact) / np.linalg.norm(particle_qx)
            ),
            "model_vs_exact_operator_qx_relative_l2": float(
                np.linalg.norm(model_qx - exact) / np.linalg.norm(exact)
            ),
            "publication_target_qx_l2": 0.05,
        }
    )
    best_metrics["publication_target_passed"] = bool(
        best_metrics["heat_flux_active_axis_relative_l2"] < 0.05
        and best_metrics["gate_passed"]
    )
    (output / "metrics.json").write_text(
        json.dumps(best_metrics, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "baseline_particle_validation": baseline_metrics,
        "best_particle_validation": best_metrics,
        "baseline_heldout_physics": baseline_physics,
        "best_projection_step": best_step,
        "selection_uses_particle_data": False,
        "training_seconds": time.perf_counter() - started,
    }
    (output / "projection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "execution": "local_cpu_workstation",
                "tensorflow": tf.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    copy_reproducibility_snapshot(output)
    print("FINAL_METRICS " + json.dumps(best_metrics, sort_keys=True), flush=True)
    print(f"Artifacts: {output}", flush=True)
    if args.strict_gate and not best_metrics["publication_target_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

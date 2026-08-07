#!/usr/bin/env python3
"""Stable Stage-1C PINN for the 1-D OU Fokker--Planck verification problem.

The analytic transient solution is never used in the training loss.  The
network instead uses the known symmetric initial density, the stationary OU
Maxwellian, the FP residual, zero boundary flux, and moment equations.  Model
weights are persisted in Keras H5 files and every save is immediately reloaded
into a fresh model and compared on a deterministic probe grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import time
from dataclasses import asdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf

from train_stage1 import Config, Stage1OUPINN, current_learning_rate, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40000)
    parser.add_argument("--n-interior", type=int, default=8192)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--n-moment-times", type=int, default=40)
    parser.add_argument("--n-velocity-quad", type=int, default=513)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--lr-decay-steps", type=int, default=10000)
    parser.add_argument("--lr-decay-rate", type=float, default=0.3)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--checkpoint-every", type=int, default=2500)
    parser.add_argument("--strong-weight", type=float, default=1.0)
    parser.add_argument("--relative-weight", type=float, default=0.10)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--mass-weight", type=float, default=50.0)
    parser.add_argument("--first-moment-weight", type=float, default=0.0)
    parser.add_argument("--second-moment-weight", type=float, default=10.0)
    parser.add_argument("--focus-center", type=float, default=0.30)
    parser.add_argument("--focus-width", type=float, default=0.18)
    parser.add_argument("--time-focus-amplitude", type=float, default=1.5)
    parser.add_argument("--equilibrium-floor", type=float, default=0.15)
    parser.add_argument("--correction-cap", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", default="outputs/stage1c-ou")
    parser.add_argument(
        "--resume-weights",
        default=None,
        help="portable .weights.h5 file to continue from",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="completed epoch corresponding to --resume-weights",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    return Config(
        epochs=args.epochs,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        n_moment_times=args.n_moment_times,
        n_velocity_quad=args.n_velocity_quad,
        width=args.width,
        depth=args.depth,
        learning_rate=args.learning_rate,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_rate=args.lr_decay_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        checkpoint_every=args.checkpoint_every,
        vmax=6.0,
        tmax=1.0,
        strong_weight=args.strong_weight,
        relative_weight=args.relative_weight,
        boundary_weight=args.boundary_weight,
        mass_weight=args.mass_weight,
        first_moment_weight=args.first_moment_weight,
        second_moment_weight=args.second_moment_weight,
        seed=args.seed,
        print_every=args.print_every,
        dtype=args.dtype,
        output_dir=args.output_dir,
    )


class Stage1CPINN(Stage1OUPINN):
    """Even, positive PINN around a physics-based equilibrium bridge."""

    def __init__(self, config: Config, args: argparse.Namespace) -> None:
        super().__init__(config)
        self.focus_center = args.focus_center
        self.focus_width = args.focus_width
        self.time_focus_amplitude = args.time_focus_amplitude
        self.equilibrium_floor = args.equilibrium_floor
        self.correction_cap = args.correction_cap

        schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            config.learning_rate,
            decay_steps=config.lr_decay_steps,
            decay_rate=config.lr_decay_rate,
            staircase=True,
        )
        self.optimizer = tf.keras.optimizers.Adam(schedule)

        # More moment points near both t=0 and t=tmax, without using the exact
        # transient density.
        theta = tf.linspace(
            tf.cast(0.0, self.dtype),
            tf.cast(np.pi, self.dtype),
            config.n_moment_times,
        )
        self.t_moment = tf.reshape(
            0.5 * tf.cast(config.tmax, self.dtype) * (1.0 - tf.cos(theta)),
            (-1, 1),
        )

    def equilibrium_density(self, v: tf.Tensor) -> tf.Tensor:
        two_pi = tf.cast(2.0 * np.pi, self.dtype)
        return tf.exp(-0.5 * tf.square(v)) / tf.sqrt(two_pi)

    def bridge_density(self, t: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        """Mass-one bridge from f0 to the stationary Maxwellian.

        The mixing rate is the known OU second-moment relaxation rate.  This is
        not the analytic transient solution; it is only a stable positive base
        state that the PINN corrects through the FP residual.
        """
        alpha = -tf.math.expm1(-2.0 * t)
        initial = tf.exp(self.log_initial_density(v))
        equilibrium = self.equilibrium_density(v)
        return (1.0 - alpha) * initial + alpha * equilibrium

    def log_density(self, t: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        cfg = self.config
        t_scaled = 2.0 * t / tf.cast(cfg.tmax, self.dtype) - 1.0
        v2 = tf.square(v / tf.cast(cfg.vmax, self.dtype))
        features = tf.concat((t_scaled, v2, tf.square(v2)), axis=1)

        # Features depend only on v^2, so symmetry is exact and M1 vanishes by
        # construction for this symmetric OU verification problem.
        raw = self.model(features)
        cap = tf.cast(self.correction_cap, self.dtype)
        correction = (
            t / tf.cast(cfg.tmax, self.dtype)
        ) * cap * tf.tanh(raw / cap)
        tiny = tf.cast(1.0e-30 if self.dtype == tf.float32 else 1.0e-300, self.dtype)
        return tf.math.log(tf.maximum(self.bridge_density(t, v), tiny)) + correction

    def sample_times(self, count: int) -> tf.Tensor:
        dtype = self.dtype
        n_uniform = count // 2
        n_focus = count // 4
        n_early = count - n_uniform - n_focus
        uniform = tf.random.uniform(
            (n_uniform, 1), 0.0, self.config.tmax, dtype=dtype
        )
        focused = tf.clip_by_value(
            tf.random.normal(
                (n_focus, 1),
                mean=tf.cast(self.focus_center, dtype),
                stddev=tf.cast(self.focus_width, dtype),
                dtype=dtype,
            ),
            tf.cast(0.0, dtype),
            tf.cast(self.config.tmax, dtype),
        )
        early_uniform = tf.random.uniform((n_early, 1), 0.0, 1.0, dtype=dtype)
        early = tf.square(early_uniform) * tf.cast(self.config.tmax, dtype)
        return tf.random.shuffle(tf.concat((uniform, focused, early), axis=0))

    def sample_positive_velocities(self, count: int) -> tf.Tensor:
        dtype = self.dtype
        n_uniform = count // 4
        n_center = count // 4
        n_thermal = count // 4
        n_peaks = count - n_uniform - n_center - n_thermal
        uniform = tf.random.uniform(
            (n_uniform, 1), 0.0, self.config.vmax, dtype=dtype
        )
        center = tf.abs(tf.random.normal((n_center, 1), stddev=0.55, dtype=dtype))
        thermal = tf.abs(
            tf.random.normal((n_thermal, 1), stddev=1.5, dtype=dtype)
        )
        peaks = tf.abs(
            tf.random.normal(
                (n_peaks, 1), mean=tf.cast(1.6, dtype), stddev=0.70, dtype=dtype
            )
        )
        velocity = tf.concat((uniform, center, thermal, peaks), axis=0)
        return tf.random.shuffle(
            tf.clip_by_value(
                velocity,
                tf.cast(0.0, dtype),
                tf.cast(self.config.vmax, dtype),
            )
        )

    def sample_training_points(self) -> tuple[tf.Tensor, ...]:
        cfg = self.config
        if cfg.n_interior % 2:
            raise ValueError("--n-interior must be even for paired +/-v sampling")

        half = cfg.n_interior // 2
        t_half = self.sample_times(half)
        v_half = self.sample_positive_velocities(half)
        t_interior = tf.concat((t_half, t_half), axis=0)
        v_interior = tf.concat((v_half, -v_half), axis=0)
        permutation = tf.random.shuffle(tf.range(cfg.n_interior))
        t_interior = tf.gather(t_interior, permutation)
        v_interior = tf.gather(v_interior, permutation)

        t_boundary_half = self.sample_times(cfg.n_boundary)
        t_boundary = tf.concat((t_boundary_half, t_boundary_half), axis=0)
        v_boundary = tf.concat(
            (
                tf.fill((cfg.n_boundary, 1), tf.cast(-cfg.vmax, self.dtype)),
                tf.fill((cfg.n_boundary, 1), tf.cast(cfg.vmax, self.dtype)),
            ),
            axis=0,
        )
        return t_interior, v_interior, t_boundary, v_boundary

    @tf.function(reduce_retracing=True)
    def train_step(
        self,
        t_interior: tf.Tensor,
        v_interior: tf.Tensor,
        t_boundary: tf.Tensor,
        v_boundary: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        cfg = self.config
        with tf.GradientTape() as parameter_tape:
            strong, relative = self.residuals(t_interior, v_interior)
            prediction = tf.stop_gradient(self.density(t_interior, v_interior))
            equilibrium = self.equilibrium_density(v_interior)
            support = tf.maximum(
                prediction,
                tf.cast(self.equilibrium_floor, self.dtype) * equilibrium,
            )
            support_weight = tf.clip_by_value(
                support / (tf.reduce_mean(support) + 1.0e-12), 0.10, 8.0
            )
            time_distance = (
                t_interior - tf.cast(self.focus_center, self.dtype)
            ) / tf.cast(self.focus_width, self.dtype)
            time_weight = 1.0 + tf.cast(
                self.time_focus_amplitude, self.dtype
            ) * tf.exp(-0.5 * tf.square(time_distance))

            flux = self.boundary_flux(t_boundary, v_boundary)
            strong_loss = tf.reduce_mean(time_weight * tf.square(strong))
            relative_loss = tf.reduce_mean(
                time_weight * support_weight * tf.square(relative)
            )
            boundary_loss = tf.reduce_mean(tf.square(flux))
            mass_loss, first_loss, second_loss = self.conservation_losses()
            total = (
                cfg.strong_weight * strong_loss
                + cfg.relative_weight * relative_loss
                + cfg.boundary_weight * boundary_loss
                + cfg.mass_weight * mass_loss
                + cfg.first_moment_weight * first_loss
                + cfg.second_moment_weight * second_loss
            )

        gradients = parameter_tape.gradient(total, self.model.trainable_variables)
        gradients, gradient_norm = tf.clip_by_global_norm(
            gradients, cfg.gradient_clip_norm
        )
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return (
            total,
            strong_loss,
            relative_loss,
            boundary_loss,
            mass_loss,
            first_loss,
            second_loss,
            gradient_norm,
        )


def model_probe(solver: Stage1CPINN) -> np.ndarray:
    times = np.linspace(0.0, solver.config.tmax, 9, dtype=np.float64)
    velocity = np.linspace(-solver.config.vmax, solver.config.vmax, 129)
    tt, vv = np.meshgrid(times, velocity, indexing="ij")
    values = solver.density(
        tf.convert_to_tensor(tt.reshape(-1, 1), dtype=solver.dtype),
        tf.convert_to_tensor(vv.reshape(-1, 1), dtype=solver.dtype),
    )
    return np.asarray(values.numpy(), dtype=np.float64).reshape(tt.shape)


def save_and_verify_weights(
    solver: Stage1CPINN,
    args: argparse.Namespace,
    path: pathlib.Path,
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    before = model_probe(solver)
    solver.model.save_weights(str(path))
    verifier = Stage1CPINN(solver.config, args)
    verifier.model.load_weights(str(path))
    after = model_probe(verifier)
    difference = float(np.max(np.abs(before - after)))
    if difference > 1.0e-7:
        raise RuntimeError(
            f"Portable weight round-trip failed for {path}: Linf={difference:.3e}"
        )
    return difference


def load_and_verify_weights(
    solver: Stage1CPINN,
    args: argparse.Namespace,
    path: pathlib.Path,
) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Missing resume weights: {path}")
    solver.model.load_weights(str(path))
    expected = model_probe(solver)
    verifier = Stage1CPINN(solver.config, args)
    verifier.model.load_weights(str(path))
    actual = model_probe(verifier)
    difference = float(np.max(np.abs(expected - actual)))
    if difference > 1.0e-7:
        raise RuntimeError(
            f"Resume weight verification failed for {path}: Linf={difference:.3e}"
        )
    return difference


def main() -> None:
    args = parse_args()
    if args.start_epoch < 0 or args.start_epoch >= args.epochs:
        raise SystemExit("--start-epoch must be in [0, epochs)")
    if bool(args.resume_weights) != bool(args.start_epoch):
        raise SystemExit("Provide both --resume-weights and a positive --start-epoch")

    config = make_config(args)
    output_dir = pathlib.Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints_h5"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    print(f"TensorFlow: {tf.__version__}", flush=True)
    print(f"Visible GPUs: {tf.config.list_physical_devices('GPU')}", flush=True)
    print(json.dumps({**asdict(config), **vars(args)}, indent=2), flush=True)

    solver = Stage1CPINN(config, args)
    if args.resume_weights:
        resume_path = pathlib.Path(args.resume_weights).expanduser().resolve()
        resume_linf = load_and_verify_weights(solver, args, resume_path)
        print(
            f"Resumed portable weights from {resume_path}; "
            f"reload Linf={resume_linf:.3e}",
            flush=True,
        )

    history_path = output_dir / "loss_history.csv"
    append = args.start_epoch > 0 and history_path.exists()
    history = history_path.open(
        "a" if append else "w", newline="", encoding="utf-8"
    )
    writer = csv.writer(history)
    if not append:
        writer.writerow(
            (
                "epoch",
                "total",
                "strong_pde",
                "relative_pde",
                "boundary_flux",
                "mass",
                "first_moment_ode",
                "second_moment_ode",
                "gradient_norm",
                "learning_rate",
            )
        )

    start = time.perf_counter()
    try:
        for epoch in range(args.start_epoch + 1, config.epochs + 1):
            losses = solver.train_step(*solver.sample_training_points())
            values = tuple(float(value.numpy()) for value in losses)
            learning_rate = current_learning_rate(solver.optimizer)
            writer.writerow((epoch, *values, learning_rate))

            if not np.all(np.isfinite(values)):
                emergency = checkpoint_dir / f"nonfinite-{epoch:06d}.weights.h5"
                save_and_verify_weights(solver, args, emergency)
                history.flush()
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {values}")

            if epoch % config.checkpoint_every == 0:
                saved = checkpoint_dir / f"epoch-{epoch:06d}.weights.h5"
                reload_linf = save_and_verify_weights(solver, args, saved)
                with (output_dir / "latest_portable_checkpoint.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(
                        {
                            "epoch": epoch,
                            "weights": str(saved),
                            "reload_linf": reload_linf,
                        },
                        handle,
                        indent=2,
                    )
                history.flush()
                print(
                    f"Portable checkpoint: {saved}; reload Linf={reload_linf:.3e}",
                    flush=True,
                )

            if epoch == args.start_epoch + 1 or epoch % config.print_every == 0:
                total, strong, relative, boundary, mass, first, second, grad = values
                print(
                    f"stage1c_epoch={epoch:6d} total={total:.3e} "
                    f"strong={strong:.3e} relative={relative:.3e} "
                    f"flux={boundary:.3e} mass={mass:.3e} "
                    f"m1ode={first:.3e} m2ode={second:.3e} "
                    f"grad={grad:.3e} lr={learning_rate:.3e} "
                    f"elapsed={time.perf_counter()-start:.1f}s",
                    flush=True,
                )
                history.flush()
    finally:
        history.close()

    final_weights = output_dir / "stage1c_final.weights.h5"
    final_reload_linf = save_and_verify_weights(solver, args, final_weights)
    live_metrics = evaluate(solver, output_dir)

    reload_solver = Stage1CPINN(config, args)
    reload_solver.model.load_weights(str(final_weights))
    reload_dir = output_dir / "final_reload_audit"
    reload_dir.mkdir(parents=True, exist_ok=True)
    reload_metrics = evaluate(reload_solver, reload_dir)
    metric_names = (
        "relative_l2",
        "final_time_relative_l2",
        "max_mass_error",
        "max_first_moment",
        "max_second_moment_error",
        "minimum_density",
        "initial_condition_linf",
    )
    metric_differences = {
        name: abs(float(live_metrics[name]) - float(reload_metrics[name]))
        for name in metric_names
    }
    reload_gate = final_reload_linf <= 1.0e-7 and max(metric_differences.values()) <= 1.0e-10

    live_metrics["method"] = (
        "even_v2_features+equilibrium_bridge+paired_collocation+portable_h5"
    )
    live_metrics["training_seconds"] = float(time.perf_counter() - start)
    live_metrics["portable_weights"] = str(final_weights)
    live_metrics["portable_reload_linf"] = final_reload_linf
    live_metrics["portable_reload_metric_differences"] = metric_differences
    live_metrics["portable_reload_gate"] = reload_gate
    live_metrics["gate_passed"] = bool(live_metrics["gate_passed"] and reload_gate)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(live_metrics, handle, indent=2)
    with (output_dir / "config_stage1c.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({**asdict(config), **vars(args)}, handle, indent=2)

    old_plot = output_dir / "stage1_validation.png"
    if old_plot.exists():
        old_plot.replace(output_dir / "stage1c_validation.png")
    gate = "PASS" if live_metrics["gate_passed"] else "FAIL"
    print("FINAL_METRICS " + json.dumps(live_metrics, sort_keys=True), flush=True)
    print(f"STAGE1C_GATE {gate}", flush=True)
    print(f"Artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage-1 verification PINN for the 1-D Ornstein--Uhlenbeck FP equation.

The analytic OU solution is used only after training for an independent gate.
Training uses the FP residual, positivity and exact initial condition, unit mass,
and the OU first- and second-moment evolution equations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf

from reference import exact_density, exact_second_moment, trapezoidal_integral


@dataclass(frozen=True)
class Config:
    epochs: int
    n_interior: int
    n_boundary: int
    n_moment_times: int
    n_velocity_quad: int
    width: int
    depth: int
    learning_rate: float
    lr_decay_steps: int
    lr_decay_rate: float
    gradient_clip_norm: float
    vmax: float
    tmax: float
    strong_weight: float
    relative_weight: float
    boundary_weight: float
    mass_weight: float
    first_moment_weight: float
    second_moment_weight: float
    seed: int
    print_every: int
    dtype: str
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--n-interior", type=int, default=8192)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--n-moment-times", type=int, default=32)
    parser.add_argument("--n-velocity-quad", type=int, default=513)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--lr-decay-steps", type=int, default=7500)
    parser.add_argument("--lr-decay-rate", type=float, default=0.3)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--vmax", type=float, default=6.0)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--strong-weight", type=float, default=1.0)
    parser.add_argument("--relative-weight", type=float, default=0.05)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--mass-weight", type=float, default=50.0)
    parser.add_argument("--first-moment-weight", type=float, default=5.0)
    parser.add_argument("--second-moment-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", default="outputs/stage1-ou")
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
        vmax=args.vmax,
        tmax=args.tmax,
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


class CorrectionNetwork(tf.keras.Model):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.hidden = [
            tf.keras.layers.Dense(
                width,
                activation="tanh",
                kernel_initializer="glorot_normal",
            )
            for _ in range(depth)
        ]
        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer="zeros")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        z = inputs
        for layer in self.hidden:
            z = layer(z)
        return self.output_layer(z)


class Stage1OUPINN:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.dtype = tf.float64 if config.dtype == "float64" else tf.float32
        tf.keras.backend.set_floatx(config.dtype)
        self.model = CorrectionNetwork(config.width, config.depth)
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            config.learning_rate,
            decay_steps=config.lr_decay_steps,
            decay_rate=config.lr_decay_rate,
            staircase=True,
        )
        self.optimizer = tf.keras.optimizers.Adam(schedule)
        self.v_quad = tf.linspace(
            tf.cast(-config.vmax, self.dtype),
            tf.cast(config.vmax, self.dtype),
            config.n_velocity_quad,
        )
        self.t_moment = tf.reshape(
            tf.linspace(
                tf.cast(0.0, self.dtype),
                tf.cast(config.tmax, self.dtype),
                config.n_moment_times,
            ),
            (-1, 1),
        )
        _ = self.model(tf.zeros((1, 3), dtype=self.dtype))

    def log_initial_density(self, v: tf.Tensor) -> tf.Tensor:
        # Stable log of 0.5*N(-2,0.25) + 0.5*N(2,0.25).
        mu = tf.cast(2.0, self.dtype)
        variance = tf.cast(0.25, self.dtype)
        x = v * mu / variance
        abs_x = tf.abs(x)
        log_cosh = abs_x + tf.math.softplus(-2.0 * abs_x) - tf.math.log(
            tf.cast(2.0, self.dtype)
        )
        normalizer = 0.5 * tf.math.log(
            tf.cast(2.0 * np.pi, self.dtype) * variance
        )
        return -(tf.square(v) + tf.square(mu)) / (2.0 * variance) - normalizer + log_cosh

    def log_density(self, t: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        cfg = self.config
        t_scaled = 2.0 * t / tf.cast(cfg.tmax, self.dtype) - 1.0
        v_scaled = v / tf.cast(cfg.vmax, self.dtype)
        features = tf.concat((t_scaled, v_scaled, tf.square(v_scaled)), axis=1)
        raw = self.model(features)
        cap = tf.cast(24.0, self.dtype)
        correction = (t / tf.cast(cfg.tmax, self.dtype)) * cap * tf.tanh(raw / cap)
        return self.log_initial_density(v) + correction

    def density(self, t: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        return tf.exp(self.log_density(t, v))

    def residuals(self, t: tf.Tensor, v: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # With h=log(f), division of f_t=(vf)_v+f_vv by f gives
        # h_t - 1 - v*h_v - h_vv - h_v^2 = 0.
        with tf.GradientTape(persistent=True) as outer:
            outer.watch((t, v))
            with tf.GradientTape(persistent=True) as inner:
                inner.watch((t, v))
                h = self.log_density(t, v)
            h_t = inner.gradient(h, t)
            h_v = inner.gradient(h, v)
            del inner
        h_vv = outer.gradient(h_v, v)
        del outer
        relative = h_t - 1.0 - v * h_v - h_vv - tf.square(h_v)
        strong = tf.exp(h) * relative
        return strong, relative

    def boundary_flux(self, t: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(v)
            h = self.log_density(t, v)
        h_v = tape.gradient(h, v)
        return tf.exp(h) * (v + h_v)

    def quadrature_moments(
        self, times: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        nt = tf.shape(times)[0]
        nv = self.config.n_velocity_quad
        t = tf.repeat(times, repeats=nv, axis=0)
        v = tf.tile(tf.reshape(self.v_quad, (-1, 1)), (nt, 1))
        f = tf.reshape(self.density(t, v), (nt, nv))
        v_rows = tf.tile(tf.reshape(self.v_quad, (1, -1)), (nt, 1))
        dv = tf.cast(2.0 * self.config.vmax / (nv - 1), self.dtype)

        def integrate(values: tf.Tensor) -> tf.Tensor:
            return dv * (
                0.5 * values[:, 0]
                + tf.reduce_sum(values[:, 1:-1], axis=1)
                + 0.5 * values[:, -1]
            )

        mass = integrate(f)
        first = integrate(f * v_rows)
        second = integrate(f * tf.square(v_rows))
        return mass, first, second

    def conservation_losses(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.t_moment)
            mass, first, second = self.quadrature_moments(self.t_moment)
        first_t = tape.gradient(first, self.t_moment)
        second_t = tape.gradient(second, self.t_moment)
        del tape
        first_t = tf.reshape(first_t, (-1,))
        second_t = tf.reshape(second_t, (-1,))
        # OU moment equations: M1_t=-M1 and M2_t=2(1-M2).
        mass_loss = tf.reduce_mean(tf.square(mass - 1.0))
        first_loss = tf.reduce_mean(tf.square(first_t + first))
        second_loss = tf.reduce_mean(tf.square(second_t + 2.0 * (second - 1.0)))
        return mass_loss, first_loss, second_loss

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
            f_support = tf.stop_gradient(self.density(t_interior, v_interior))
            support_weight = tf.clip_by_value(
                f_support / (tf.reduce_mean(f_support) + 1.0e-12), 0.05, 10.0
            )
            flux = self.boundary_flux(t_boundary, v_boundary)
            strong_loss = tf.reduce_mean(tf.square(strong))
            relative_loss = tf.reduce_mean(support_weight * tf.square(relative))
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

    def sample_training_points(self) -> tuple[tf.Tensor, ...]:
        cfg = self.config
        dtype = self.dtype
        t_interior = tf.random.uniform(
            (cfg.n_interior, 1), 0.0, cfg.tmax, dtype=dtype
        )
        n_uniform = cfg.n_interior // 2
        n_focused = cfg.n_interior - n_uniform
        uniform = tf.random.uniform(
            (n_uniform, 1), -cfg.vmax, cfg.vmax, dtype=dtype
        )
        focused = tf.clip_by_value(
            tf.random.normal((n_focused, 1), stddev=2.0, dtype=dtype),
            tf.cast(-cfg.vmax, dtype),
            tf.cast(cfg.vmax, dtype),
        )
        v_interior = tf.concat((uniform, focused), axis=0)
        permutation = tf.random.shuffle(tf.range(cfg.n_interior))
        v_interior = tf.gather(v_interior, permutation)

        t_boundary = tf.random.uniform(
            (2 * cfg.n_boundary, 1), 0.0, cfg.tmax, dtype=dtype
        )
        v_boundary = tf.concat(
            (
                tf.fill((cfg.n_boundary, 1), tf.cast(-cfg.vmax, dtype)),
                tf.fill((cfg.n_boundary, 1), tf.cast(cfg.vmax, dtype)),
            ),
            axis=0,
        )
        return t_interior, v_interior, t_boundary, v_boundary


def evaluate(solver: Stage1OUPINN, output_dir: pathlib.Path) -> dict[str, object]:
    cfg = solver.config
    times = np.linspace(0.0, cfg.tmax, 41, dtype=np.float64)
    velocity = np.linspace(-cfg.vmax, cfg.vmax, 801, dtype=np.float64)
    tt, vv = np.meshgrid(times, velocity, indexing="ij")
    prediction = solver.density(
        tf.convert_to_tensor(tt.reshape(-1, 1), dtype=solver.dtype),
        tf.convert_to_tensor(vv.reshape(-1, 1), dtype=solver.dtype),
    ).numpy().reshape(tt.shape)
    exact = exact_density(tt, vv)

    error = prediction - exact
    relative_by_time = np.sqrt(
        np.sum(error**2, axis=1) / np.maximum(np.sum(exact**2, axis=1), 1.0e-300)
    )
    mass = trapezoidal_integral(prediction, velocity, axis=1)
    first = trapezoidal_integral(prediction * velocity[None, :], velocity, axis=1)
    second = trapezoidal_integral(
        prediction * velocity[None, :] ** 2, velocity, axis=1
    )
    exact_second = exact_second_moment(times)

    metrics: dict[str, object] = {
        "relative_l2": float(np.sqrt(np.sum(error**2) / np.sum(exact**2))),
        "final_time_relative_l2": float(relative_by_time[-1]),
        "max_mass_error": float(np.max(np.abs(mass - 1.0))),
        "max_first_moment": float(np.max(np.abs(first))),
        "max_second_moment_error": float(np.max(np.abs(second - exact_second))),
        "minimum_density": float(np.min(prediction)),
        "initial_condition_linf": float(np.max(np.abs(prediction[0] - exact[0]))),
    }

    thresholds = {
        "relative_l2": 5.0e-2,
        "final_time_relative_l2": 5.0e-2,
        "max_mass_error": 1.0e-2,
        "max_first_moment": 1.0e-2,
        "max_second_moment_error": 3.0e-2,
        "initial_condition_linf": 1.0e-6,
    }
    checks = {key: float(metrics[key]) <= limit for key, limit in thresholds.items()}
    checks["nonnegative_density"] = float(metrics["minimum_density"]) >= 0.0
    metrics["gate_thresholds"] = thresholds
    metrics["gate_checks"] = checks
    metrics["gate_passed"] = bool(all(checks.values()))

    with (output_dir / "metrics_by_time.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "time",
                "relative_l2",
                "mass",
                "mass_error",
                "first_moment",
                "second_moment",
                "exact_second_moment",
                "second_moment_error",
                "minimum_density",
            )
        )
        for index, current_time in enumerate(times):
            writer.writerow(
                (
                    current_time,
                    relative_by_time[index],
                    mass[index],
                    mass[index] - 1.0,
                    first[index],
                    second[index],
                    exact_second[index],
                    second[index] - exact_second[index],
                    np.min(prediction[index]),
                )
            )

    np.savez_compressed(
        output_dir / "solution_grid.npz",
        time=times,
        velocity=velocity,
        pinn=prediction,
        exact=exact,
        error=error,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
        for index in (0, 4, 10, 20, 40):
            axes[0, 0].plot(velocity, prediction[index], label=f"PINN t={times[index]:.2f}")
            axes[0, 0].plot(velocity, exact[index], "--", linewidth=1.1)
        axes[0, 0].set_xlabel("v")
        axes[0, 0].set_ylabel("f")
        axes[0, 0].set_title("Solid: PINN; dashed: exact")
        axes[0, 0].legend(fontsize=7, ncol=2)

        image = axes[0, 1].pcolormesh(velocity, times, np.abs(error), shading="auto")
        axes[0, 1].set_xlabel("v")
        axes[0, 1].set_ylabel("t")
        axes[0, 1].set_title("Absolute error")
        figure.colorbar(image, ax=axes[0, 1])

        axes[1, 0].semilogy(times, relative_by_time + 1.0e-16, label="relative L2")
        axes[1, 0].semilogy(times, np.abs(mass - 1.0) + 1.0e-16, label="mass error")
        axes[1, 0].semilogy(
            times, np.abs(second - exact_second) + 1.0e-16, label="M2 error"
        )
        axes[1, 0].set_xlabel("t")
        axes[1, 0].set_ylabel("error")
        axes[1, 0].legend()

        axes[1, 1].plot(times, second, label="PINN M2")
        axes[1, 1].plot(times, exact_second, "--", label="exact M2")
        axes[1, 1].plot(times, mass, label="mass")
        axes[1, 1].set_xlabel("t")
        axes[1, 1].set_ylabel("moment")
        axes[1, 1].legend()
        axes[1, 1].set_title("Conservation diagnostics")

        figure.tight_layout()
        figure.savefig(output_dir / "stage1_validation.png", dpi=200)
        plt.close(figure)
    except Exception as exc:
        print(f"Plot skipped: {exc}", flush=True)

    return metrics


def current_learning_rate(optimizer: tf.keras.optimizers.Optimizer) -> float:
    value = optimizer.learning_rate
    if callable(value):
        value = value(optimizer.iterations)
    return float(tf.keras.backend.get_value(value))


def main() -> None:
    args = parse_args()
    config = make_config(args)
    output_dir = pathlib.Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    print(f"TensorFlow: {tf.__version__}", flush=True)
    print(f"Visible GPUs: {tf.config.list_physical_devices('GPU')}", flush=True)
    print(json.dumps(asdict(config), indent=2), flush=True)

    solver = Stage1OUPINN(config)
    history: list[tuple[float, ...]] = []
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        points = solver.sample_training_points()
        losses = solver.train_step(*points)
        values = tuple(float(value.numpy()) for value in losses)
        learning_rate = current_learning_rate(solver.optimizer)
        row = (float(epoch), *values, learning_rate)
        history.append(row)

        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {values}")

        if epoch == 1 or epoch % config.print_every == 0 or epoch == config.epochs:
            elapsed = time.perf_counter() - start
            total, strong, relative, boundary, mass, first, second, grad_norm = values
            print(
                f"epoch={epoch:6d} total={total:.3e} strong={strong:.3e} "
                f"relative={relative:.3e} flux={boundary:.3e} mass={mass:.3e} "
                f"m1ode={first:.3e} m2ode={second:.3e} grad={grad_norm:.3e} "
                f"lr={learning_rate:.3e} elapsed={elapsed:.1f}s",
                flush=True,
            )

    with (output_dir / "loss_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
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
        writer.writerows(history)

    checkpoint = tf.train.Checkpoint(model=solver.model, optimizer=solver.optimizer)
    checkpoint.write(str(output_dir / "checkpoint"))
    metrics = evaluate(solver, output_dir)
    metrics["training_seconds"] = float(time.perf_counter() - start)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    gate = "PASS" if metrics["gate_passed"] else "FAIL"
    print("FINAL_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
    print(f"STAGE1_GATE {gate}", flush=True)
    print(f"Artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

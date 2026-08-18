#!/usr/bin/env python3
"""Positive physics-informed neural solver for a 1-D OU Fokker--Planck IVP.

This is Stage 0 of the FP-PINN project.  It validates the Fokker--Planck sign
and factor-of-two convention, automatic second derivatives, positivity,
normalization, zero-flux boundaries, and comparison with an exact solution.

Dimensionless problem
---------------------

    f_t = (v f)_v + f_vv,          v in [-Vmax, Vmax], t in [0, Tmax]
    v f + f_v = 0                  at v = +/- Vmax
    f(0,v) = 0.5 N(-2,0.5^2) + 0.5 N(2,0.5^2)

It is the FP equation associated with dV = -V dt + sqrt(2) dW.  Therefore its
stationary density has unit variance.  The initial condition and positivity
are imposed by the hard ansatz f_theta = f0 exp(t g_theta).
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

# Keep TensorFlow logging readable on a cluster.
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
    vmax: float
    tmax: float
    residual_weight: float
    boundary_weight: float
    mass_weight: float
    moment_weight: float
    seed: int
    print_every: int
    dtype: str
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small first Unity run")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-interior", type=int, default=None)
    parser.add_argument("--n-boundary", type=int, default=None)
    parser.add_argument("--n-moment-times", type=int, default=8)
    parser.add_argument("--n-velocity-quad", type=int, default=None)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--vmax", type=float, default=6.0)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--mass-weight", type=float, default=10.0)
    parser.add_argument("--moment-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", default="outputs/ou_1d")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    quick_defaults = {
        "epochs": 800,
        "n_interior": 2048,
        "n_boundary": 256,
        "n_velocity_quad": 257,
    }
    full_defaults = {
        "epochs": 5000,
        "n_interior": 8192,
        "n_boundary": 512,
        "n_velocity_quad": 513,
    }
    defaults = quick_defaults if args.quick else full_defaults
    return Config(
        epochs=args.epochs or defaults["epochs"],
        n_interior=args.n_interior or defaults["n_interior"],
        n_boundary=args.n_boundary or defaults["n_boundary"],
        n_moment_times=args.n_moment_times,
        n_velocity_quad=args.n_velocity_quad or defaults["n_velocity_quad"],
        width=args.width,
        depth=args.depth,
        learning_rate=args.learning_rate,
        vmax=args.vmax,
        tmax=args.tmax,
        residual_weight=args.residual_weight,
        boundary_weight=args.boundary_weight,
        mass_weight=args.mass_weight,
        moment_weight=args.moment_weight,
        seed=args.seed,
        print_every=args.print_every,
        dtype=args.dtype,
        output_dir=args.output_dir,
    )


class CorrectionNetwork(tf.keras.Model):
    """Small tanh MLP for the bounded log-density correction."""

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
        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer="glorot_normal")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        z = inputs
        for layer in self.hidden:
            z = layer(z)
        return self.output_layer(z)


class OUPINN:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.dtype = tf.float64 if config.dtype == "float64" else tf.float32
        tf.keras.backend.set_floatx(config.dtype)
        self.model = CorrectionNetwork(config.width, config.depth)
        self.optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        self.v_quad = tf.linspace(
            tf.cast(-config.vmax, self.dtype),
            tf.cast(config.vmax, self.dtype),
            config.n_velocity_quad,
        )
        # Build variables before creating a checkpoint.
        _ = self.model(tf.zeros((1, 2), dtype=self.dtype))

    def initial_density(self, v: tf.Tensor) -> tf.Tensor:
        mu0 = tf.cast(2.0, self.dtype)
        variance0 = tf.cast(0.25, self.dtype)
        normalizer = tf.sqrt(tf.cast(2.0 * np.pi, self.dtype) * variance0)
        left = tf.exp(-0.5 * tf.square(v + mu0) / variance0) / normalizer
        right = tf.exp(-0.5 * tf.square(v - mu0) / variance0) / normalizer
        return 0.5 * (left + right)

    def density(self, t: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        cfg = self.config
        t_scaled = 2.0 * t / tf.cast(cfg.tmax, self.dtype) - 1.0
        v_scaled = v / tf.cast(cfg.vmax, self.dtype)
        raw = self.model(tf.concat((t_scaled, v_scaled), axis=1))
        # Smoothly bound the correction and make it exactly zero at t=0.
        cap = tf.cast(12.0, self.dtype)
        correction = (t / tf.cast(cfg.tmax, self.dtype)) * cap * tf.tanh(raw / cap)
        return self.initial_density(v) * tf.exp(correction)

    def pde_residual(self, t: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        # Nested tapes are required for f_vv.
        with tf.GradientTape(persistent=True) as outer:
            outer.watch((t, v))
            with tf.GradientTape(persistent=True) as inner:
                inner.watch((t, v))
                f = self.density(t, v)
                vf = v * f
            f_t = inner.gradient(f, t)
            f_v = inner.gradient(f, v)
            drift_v = inner.gradient(vf, v)
            del inner
        f_vv = outer.gradient(f_v, v)
        del outer
        return f_t - drift_v - f_vv

    def boundary_flux(self, t: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(v)
            f = self.density(t, v)
        f_v = tape.gradient(f, v)
        return v * f + f_v

    def moment_losses(self, times: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        nt = tf.shape(times)[0]
        nv = self.config.n_velocity_quad
        t = tf.repeat(times, repeats=nv, axis=0)
        v = tf.tile(tf.reshape(self.v_quad, (-1, 1)), (nt, 1))
        f = tf.reshape(self.density(t, v), (nt, nv))
        v_rows = tf.tile(tf.reshape(self.v_quad, (1, -1)), (nt, 1))
        dv = tf.cast(2.0 * self.config.vmax / (nv - 1), self.dtype)

        def uniform_trapezoid(values: tf.Tensor) -> tf.Tensor:
            return dv * (
                0.5 * values[:, 0]
                + tf.reduce_sum(values[:, 1:-1], axis=1)
                + 0.5 * values[:, -1]
            )

        mass = uniform_trapezoid(f)
        second = uniform_trapezoid(f * tf.square(v_rows))
        exact_second = 1.0 + 3.25 * tf.exp(-2.0 * tf.reshape(times, (-1,)))
        mass_loss = tf.reduce_mean(tf.square(mass - 1.0))
        moment_loss = tf.reduce_mean(tf.square(second - exact_second))
        return mass_loss, moment_loss

    @tf.function(reduce_retracing=True)
    def train_step(
        self,
        t_interior: tf.Tensor,
        v_interior: tf.Tensor,
        t_boundary: tf.Tensor,
        v_boundary: tf.Tensor,
        t_moment: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        cfg = self.config
        with tf.GradientTape() as parameter_tape:
            residual = self.pde_residual(t_interior, v_interior)
            flux = self.boundary_flux(t_boundary, v_boundary)
            residual_loss = tf.reduce_mean(tf.square(residual))
            boundary_loss = tf.reduce_mean(tf.square(flux))
            mass_loss, moment_loss = self.moment_losses(t_moment)
            total = (
                cfg.residual_weight * residual_loss
                + cfg.boundary_weight * boundary_loss
                + cfg.mass_weight * mass_loss
                + cfg.moment_weight * moment_loss
            )
        gradients = parameter_tape.gradient(total, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return total, residual_loss, boundary_loss, mass_loss, moment_loss

    def sample_training_points(self) -> tuple[tf.Tensor, ...]:
        cfg = self.config
        dtype = self.dtype
        t_interior = tf.random.uniform(
            (cfg.n_interior, 1), 0.0, cfg.tmax, dtype=dtype
        )
        v_interior = tf.random.uniform(
            (cfg.n_interior, 1), -cfg.vmax, cfg.vmax, dtype=dtype
        )
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
        t_moment = tf.random.uniform(
            (cfg.n_moment_times, 1), 0.0, cfg.tmax, dtype=dtype
        )
        return t_interior, v_interior, t_boundary, v_boundary, t_moment


def evaluate(solver: OUPINN, output_dir: pathlib.Path) -> dict[str, float]:
    cfg = solver.config
    times = np.linspace(0.0, cfg.tmax, 41, dtype=np.float64)
    velocity = np.linspace(-cfg.vmax, cfg.vmax, 801, dtype=np.float64)
    tt, vv = np.meshgrid(times, velocity, indexing="ij")
    prediction = solver.density(
        tf.convert_to_tensor(tt.reshape(-1, 1), dtype=solver.dtype),
        tf.convert_to_tensor(vv.reshape(-1, 1), dtype=solver.dtype),
    ).numpy().reshape(tt.shape)
    exact = exact_density(tt, vv)

    relative_l2 = float(np.sqrt(np.sum((prediction - exact) ** 2) / np.sum(exact**2)))
    mass = trapezoidal_integral(prediction, velocity, axis=1)
    second = trapezoidal_integral(
        prediction * velocity[None, :] ** 2, velocity, axis=1
    )
    exact_second = exact_second_moment(times)
    metrics = {
        "relative_l2": relative_l2,
        "max_mass_error": float(np.max(np.abs(mass - 1.0))),
        "max_second_moment_error": float(np.max(np.abs(second - exact_second))),
        "minimum_density": float(np.min(prediction)),
        "initial_condition_linf": float(np.max(np.abs(prediction[0] - exact[0]))),
    }

    final_csv = output_dir / "solution_tfinal.csv"
    with final_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("v", "pinn", "exact", "absolute_error"))
        for v, pred, ref in zip(velocity, prediction[-1], exact[-1]):
            writer.writerow((v, pred, ref, abs(pred - ref)))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        for index in (0, 10, 20, 40):
            axes[0].plot(velocity, prediction[index], label=f"PINN t={times[index]:.2f}")
            axes[0].plot(velocity, exact[index], "--", linewidth=1.2)
        axes[0].set_xlabel("v")
        axes[0].set_ylabel("f")
        axes[0].legend(fontsize=7)
        axes[0].set_title("solid: PINN, dashed: exact")
        axes[1].semilogy(velocity, np.abs(prediction[-1] - exact[-1]) + 1.0e-14)
        axes[1].set_xlabel("v")
        axes[1].set_ylabel("absolute error")
        axes[1].set_title(f"t={cfg.tmax:g}")
        figure.tight_layout()
        figure.savefig(output_dir / "ou_pinn_validation.png", dpi=180)
        plt.close(figure)
    except Exception as exc:  # Plotting is useful but must not destroy a run.
        print(f"Plot skipped: {exc}", flush=True)

    return metrics


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

    solver = OUPINN(config)
    history: list[tuple[float, ...]] = []
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        points = solver.sample_training_points()
        losses = solver.train_step(*points)
        row = (epoch, *(float(value.numpy()) for value in losses))
        history.append(row)
        if epoch == 1 or epoch % config.print_every == 0 or epoch == config.epochs:
            elapsed = time.perf_counter() - start
            _, total, residual, boundary, mass, moment = row
            print(
                f"epoch={epoch:6d} total={total:.3e} pde={residual:.3e} "
                f"flux={boundary:.3e} mass={mass:.3e} m2={moment:.3e} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    with (output_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "total", "pde", "boundary_flux", "mass", "second_moment"))
        writer.writerows(history)

    checkpoint = tf.train.Checkpoint(model=solver.model, optimizer=solver.optimizer)
    checkpoint.write(str(output_dir / "checkpoint"))
    metrics = evaluate(solver, output_dir)
    metrics["training_seconds"] = float(time.perf_counter() - start)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    print("FINAL_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
    print(f"Artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

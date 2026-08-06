#!/usr/bin/env python3
"""Stage-1B causal/RAR refinement for the 1-D OU Fokker--Planck PINN.

This script fine-tunes a completed Stage-1 model without using the analytic
solution in the loss.  It addresses the observed intermediate-time error by
combining causal time slabs, replay over the full interval, a known equilibrium
Maxwellian support envelope, and residual-adaptive refinement (RAR).
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
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
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--n-interior", type=int, default=8192)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--n-moment-times", type=int, default=40)
    parser.add_argument("--n-velocity-quad", type=int, default=513)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--lr-decay-steps", type=int, default=6000)
    parser.add_argument("--lr-decay-rate", type=float, default=0.3)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--strong-weight", type=float, default=1.0)
    parser.add_argument("--relative-weight", type=float, default=0.20)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--mass-weight", type=float, default=30.0)
    parser.add_argument("--first-moment-weight", type=float, default=5.0)
    parser.add_argument("--second-moment-weight", type=float, default=5.0)
    parser.add_argument("--rar-fraction", type=float, default=0.125)
    parser.add_argument("--rar-candidates", type=int, default=4096)
    parser.add_argument("--rar-every", type=int, default=10)
    parser.add_argument("--focus-center", type=float, default=0.35)
    parser.add_argument("--focus-width", type=float, default=0.16)
    parser.add_argument("--equilibrium-floor", type=float, default=0.35)
    parser.add_argument("--time-focus-amplitude", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", default="outputs/stage1b-ou")
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


class Stage1BPin(Stage1OUPINN):
    def __init__(self, config: Config, args: argparse.Namespace) -> None:
        super().__init__(config)
        self.rar_fraction = args.rar_fraction
        self.rar_candidates = args.rar_candidates
        self.rar_every = args.rar_every
        self.focus_center = args.focus_center
        self.focus_width = args.focus_width
        self.equilibrium_floor = args.equilibrium_floor
        self.time_focus_amplitude = args.time_focus_amplitude

        # Replace the Stage-1 optimizer with a fresh refinement schedule.  The
        # parent network weights are restored separately, not the old optimizer.
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            config.learning_rate,
            decay_steps=config.lr_decay_steps,
            decay_rate=config.lr_decay_rate,
            staircase=True,
        )
        self.optimizer = tf.keras.optimizers.Adam(schedule)

        # Chebyshev--Lobatto times give additional moment constraints near both
        # endpoints without using the analytic density.
        index = tf.linspace(
            tf.cast(0.0, self.dtype),
            tf.cast(np.pi, self.dtype),
            config.n_moment_times,
        )
        self.t_moment = tf.reshape(
            0.5 * tf.cast(config.tmax, self.dtype) * (1.0 - tf.cos(index)),
            (-1, 1),
        )

    def equilibrium_density(self, v: tf.Tensor) -> tf.Tensor:
        two_pi = tf.cast(2.0 * np.pi, self.dtype)
        return tf.exp(-0.5 * tf.square(v)) / tf.sqrt(two_pi)

    def causal_horizon(self, refinement_epoch: int) -> float:
        fraction = refinement_epoch / max(self.config.epochs, 1)
        if fraction <= 0.25:
            return 0.35
        if fraction <= 0.50:
            return 0.60
        return 1.00

    def sample_times(self, count: int, horizon: float) -> tf.Tensor:
        dtype = self.dtype
        n_global = count // 4
        n_focus = count // 4
        n_causal = count - n_global - n_focus
        global_times = tf.random.uniform(
            (n_global, 1), 0.0, self.config.tmax, dtype=dtype
        )
        focus_times = tf.clip_by_value(
            tf.random.normal(
                (n_focus, 1),
                mean=tf.cast(self.focus_center, dtype),
                stddev=tf.cast(self.focus_width, dtype),
                dtype=dtype,
            ),
            tf.cast(0.0, dtype),
            tf.cast(self.config.tmax, dtype),
        )
        causal_times = tf.random.uniform(
            (n_causal, 1), 0.0, horizon, dtype=dtype
        )
        times = tf.concat((global_times, focus_times, causal_times), axis=0)
        return tf.random.shuffle(times)

    def sample_velocities(self, count: int) -> tf.Tensor:
        dtype = self.dtype
        n_uniform = count // 4
        n_center = count // 4
        n_thermal = count // 4
        n_peaks = count - n_uniform - n_center - n_thermal
        uniform = tf.random.uniform(
            (n_uniform, 1), -self.config.vmax, self.config.vmax, dtype=dtype
        )
        center = tf.random.normal((n_center, 1), stddev=0.75, dtype=dtype)
        thermal = tf.random.normal((n_thermal, 1), stddev=1.6, dtype=dtype)
        signs = tf.where(
            tf.random.uniform((n_peaks, 1), dtype=dtype) < 0.5,
            tf.cast(-1.0, dtype),
            tf.cast(1.0, dtype),
        )
        peaks = signs * tf.cast(1.7, dtype) + tf.random.normal(
            (n_peaks, 1), stddev=0.55, dtype=dtype
        )
        velocities = tf.concat((uniform, center, thermal, peaks), axis=0)
        velocities = tf.clip_by_value(
            velocities,
            tf.cast(-self.config.vmax, dtype),
            tf.cast(self.config.vmax, dtype),
        )
        return tf.random.shuffle(velocities)

    def draw_points(self, count: int, horizon: float) -> tuple[tf.Tensor, tf.Tensor]:
        return self.sample_times(count, horizon), self.sample_velocities(count)

    @tf.function(reduce_retracing=True)
    def refinement_score(self, t: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        strong, relative = self.residuals(t, v)
        prediction = tf.stop_gradient(self.density(t, v))
        equilibrium = self.equilibrium_density(v)
        support = tf.maximum(
            prediction,
            tf.cast(self.equilibrium_floor, self.dtype) * equilibrium,
        )
        support_weight = tf.clip_by_value(
            support / (tf.reduce_mean(support) + 1.0e-12), 0.20, 8.0
        )
        score = tf.abs(strong) + 0.05 * tf.sqrt(support_weight) * tf.abs(relative)
        return tf.reshape(score, (-1,))

    def sample_training_points(
        self, refinement_epoch: int
    ) -> tuple[tf.Tensor, ...]:
        cfg = self.config
        horizon = self.causal_horizon(refinement_epoch)
        use_rar = refinement_epoch % self.rar_every == 0
        n_rar = int(round(cfg.n_interior * self.rar_fraction)) if use_rar else 0
        n_rar = min(max(n_rar, 0), cfg.n_interior)
        n_base = cfg.n_interior - n_rar

        t_base, v_base = self.draw_points(n_base, horizon)
        if n_rar:
            n_candidates = max(self.rar_candidates, n_rar)
            t_candidates, v_candidates = self.draw_points(n_candidates, horizon)
            scores = self.refinement_score(t_candidates, v_candidates)
            scores = tf.where(
                tf.math.is_finite(scores),
                scores,
                tf.fill(tf.shape(scores), tf.cast(-1.0, scores.dtype)),
            )
            indices = tf.math.top_k(scores, k=n_rar, sorted=False).indices
            t_interior = tf.concat((t_base, tf.gather(t_candidates, indices)), axis=0)
            v_interior = tf.concat((v_base, tf.gather(v_candidates, indices)), axis=0)
        else:
            t_interior, v_interior = t_base, v_base

        permutation = tf.random.shuffle(tf.range(cfg.n_interior))
        t_interior = tf.gather(t_interior, permutation)
        v_interior = tf.gather(v_interior, permutation)

        t_half = self.sample_times(cfg.n_boundary, horizon)
        t_boundary = tf.concat((t_half, t_half), axis=0)
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
                support / (tf.reduce_mean(support) + 1.0e-12), 0.20, 8.0
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


def _model_probe(solver: Stage1OUPINN) -> np.ndarray:
    """Return deterministic density values used to verify a weight transfer."""
    times = np.linspace(0.0, solver.config.tmax, 9, dtype=np.float64)
    velocity = np.linspace(-solver.config.vmax, solver.config.vmax, 129, dtype=np.float64)
    tt, vv = np.meshgrid(times, velocity, indexing="ij")
    values = solver.density(
        tf.convert_to_tensor(tt.reshape(-1, 1), dtype=solver.dtype),
        tf.convert_to_tensor(vv.reshape(-1, 1), dtype=solver.dtype),
    )
    return np.asarray(values.numpy(), dtype=np.float64).reshape(tt.shape)


def restore_stage1_parent_exact(
    init_path: str,
    refinement_solver: Stage1BPin,
    output_dir: pathlib.Path,
) -> dict[str, object]:
    """Restore Stage-1 through its original object graph and audit the result.

    Restoring only a nested model object can silently leave unmatched variables
    when TensorFlow/Keras checkpoint tracking changes.  Recreate the exact
    Stage-1 solver/checkpoint graph, require all live objects to match, audit
    that parent independently, and only then copy its network weights.
    """
    checkpoint_path = pathlib.Path(init_path).expanduser().resolve()
    if not pathlib.Path(str(checkpoint_path) + ".index").is_file():
        raise FileNotFoundError(f"Missing parent checkpoint: {checkpoint_path}.index")

    config_path = checkpoint_path.parent.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing Stage-1 config required for safe restore: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        parent_config = Config(**json.load(handle))

    child_config = refinement_solver.config
    architecture = ("width", "depth", "dtype", "vmax", "tmax")
    mismatches = {
        name: (getattr(parent_config, name), getattr(child_config, name))
        for name in architecture
        if getattr(parent_config, name) != getattr(child_config, name)
    }
    if mismatches:
        raise ValueError(
            "Stage-1/Stage-1B architecture mismatch: " + json.dumps(mismatches)
        )

    parent_solver = Stage1OUPINN(parent_config)
    parent_checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0, dtype=tf.int64, trainable=False),
        model=parent_solver.model,
        optimizer=parent_solver.optimizer,
    )
    restore_status = parent_checkpoint.restore(str(checkpoint_path))
    restore_status.assert_existing_objects_matched()
    restore_status.expect_partial()

    restored_epoch = int(parent_checkpoint.epoch.numpy())
    if restored_epoch != parent_config.epochs:
        raise RuntimeError(
            f"Parent checkpoint epoch={restored_epoch}, expected {parent_config.epochs}"
        )

    audit_dir = output_dir / "parent_restore_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    parent_metrics = evaluate(parent_solver, audit_dir)
    integrity_limits = {
        "relative_l2": 0.15,
        "final_time_relative_l2": 0.05,
        "max_mass_error": 0.02,
        "max_first_moment": 0.01,
        "max_second_moment_error": 0.01,
        "initial_condition_linf": 2.0e-6,
    }
    integrity_checks = {
        name: float(parent_metrics[name]) <= limit
        for name, limit in integrity_limits.items()
    }
    if not all(integrity_checks.values()):
        failed = {
            name: {
                "value": float(parent_metrics[name]),
                "limit": integrity_limits[name],
            }
            for name, passed in integrity_checks.items()
            if not passed
        }
        raise RuntimeError(
            "Stage-1 checkpoint integrity gate failed before refinement: "
            + json.dumps(failed, sort_keys=True)
        )

    parent_probe = _model_probe(parent_solver)
    refinement_solver.model.set_weights(parent_solver.model.get_weights())
    child_probe = _model_probe(refinement_solver)
    transfer_linf = float(np.max(np.abs(parent_probe - child_probe)))
    if transfer_linf > 1.0e-7:
        raise RuntimeError(
            f"Parent-to-refinement weight transfer failed: Linf={transfer_linf:.3e}"
        )

    fingerprint = hashlib.sha256(
        np.asarray(parent_probe, dtype=np.float32).tobytes()
    ).hexdigest()
    audit: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "restored_epoch": restored_epoch,
        "checkpoint_objects_matched": True,
        "density_fingerprint_sha256": fingerprint,
        "parent_to_refinement_linf": transfer_linf,
        "integrity_limits": integrity_limits,
        "integrity_checks": integrity_checks,
        "parent_metrics": parent_metrics,
    }
    with (output_dir / "parent_restore_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2)
    print("PARENT_RESTORE_AUDIT " + json.dumps(audit, sort_keys=True), flush=True)
    return audit


def main() -> None:
    args = parse_args()
    if not args.resume and not args.init_checkpoint:
        raise SystemExit("Provide --init-checkpoint, or use --resume")
    if not 0.0 <= args.rar_fraction < 1.0:
        raise SystemExit("--rar-fraction must be in [0,1)")
    if args.rar_every < 1:
        raise SystemExit("--rar-every must be at least 1")

    config = make_config(args)
    output_dir = pathlib.Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    print(f"TensorFlow: {tf.__version__}", flush=True)
    print(f"Visible GPUs: {tf.config.list_physical_devices('GPU')}", flush=True)
    print(json.dumps({**asdict(config), **vars(args)}, indent=2), flush=True)

    solver = Stage1BPin(config, args)
    refinement_checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0, dtype=tf.int64, trainable=False),
        model=solver.model,
        optimizer=solver.optimizer,
    )
    manager = tf.train.CheckpointManager(
        refinement_checkpoint,
        str(output_dir / "checkpoints"),
        max_to_keep=3,
    )

    parent_audit: dict[str, object] | None = None
    if args.resume:
        if manager.latest_checkpoint is None:
            raise FileNotFoundError(
                f"No Stage-1B checkpoint exists in {output_dir / 'checkpoints'}"
            )
        restore_status = refinement_checkpoint.restore(manager.latest_checkpoint)
        restore_status.assert_existing_objects_matched()
        restore_status.expect_partial()
        print(f"Resumed Stage-1B from {manager.latest_checkpoint}", flush=True)
    else:
        parent_audit = restore_stage1_parent_exact(
            args.init_checkpoint, solver, output_dir
        )
        print(
            f"Safely loaded Stage-1 parent weights from {args.init_checkpoint}",
            flush=True,
        )

    if args.evaluate_only:
        metrics = evaluate(solver, output_dir)
        metrics["parent_checkpoint"] = args.init_checkpoint
        metrics["parent_restore_audit"] = parent_audit
        metrics["method"] = "causal_time_slabs+equilibrium_envelope+RAR"
        metrics["evaluated_checkpoint"] = (
            manager.latest_checkpoint if args.resume else args.init_checkpoint
        )
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        old_plot = output_dir / "stage1_validation.png"
        if old_plot.exists():
            old_plot.replace(output_dir / "stage1b_validation.png")
        gate = "PASS" if metrics["gate_passed"] else "FAIL"
        print("FINAL_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
        print(f"STAGE1B_GATE {gate}", flush=True)
        print(f"Artifacts: {output_dir}", flush=True)
        return

    start_epoch = int(refinement_checkpoint.epoch.numpy()) + 1
    history_path = output_dir / "loss_history.csv"
    append_history = args.resume and history_path.exists()
    history_handle = history_path.open(
        "a" if append_history else "w", newline="", encoding="utf-8"
    )
    writer = csv.writer(history_handle)
    if not append_history:
        writer.writerow(
            (
                "refinement_epoch",
                "causal_horizon",
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
        for epoch in range(start_epoch, config.epochs + 1):
            points = solver.sample_training_points(epoch)
            losses = solver.train_step(*points)
            values = tuple(float(value.numpy()) for value in losses)
            learning_rate = current_learning_rate(solver.optimizer)
            horizon = solver.causal_horizon(epoch)
            writer.writerow((epoch, horizon, *values, learning_rate))
            refinement_checkpoint.epoch.assign(epoch)

            if not np.all(np.isfinite(values)):
                manager.save(checkpoint_number=epoch)
                history_handle.flush()
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {values}")

            if epoch % config.checkpoint_every == 0:
                saved = manager.save(checkpoint_number=epoch)
                history_handle.flush()
                gc.collect()
                print(f"Checkpoint: {saved}", flush=True)

            if epoch == start_epoch or epoch % config.print_every == 0:
                total, strong, relative, boundary, mass, first, second, grad = values
                print(
                    f"stage1b_epoch={epoch:6d} horizon={horizon:.2f} "
                    f"total={total:.3e} strong={strong:.3e} relative={relative:.3e} "
                    f"flux={boundary:.3e} mass={mass:.3e} m1ode={first:.3e} "
                    f"m2ode={second:.3e} grad={grad:.3e} lr={learning_rate:.3e} "
                    f"elapsed={time.perf_counter()-start:.1f}s",
                    flush=True,
                )
                history_handle.flush()
    finally:
        history_handle.close()

    manager.save(checkpoint_number=int(refinement_checkpoint.epoch.numpy()))
    metrics = evaluate(solver, output_dir)
    metrics["training_seconds"] = float(time.perf_counter() - start)
    metrics["parent_checkpoint"] = args.init_checkpoint
    metrics["parent_restore_audit"] = parent_audit
    metrics["method"] = "causal_time_slabs+equilibrium_envelope+RAR"
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "config_stage1b.json").open("w", encoding="utf-8") as handle:
        json.dump({**asdict(config), **vars(args)}, handle, indent=2)

    old_plot = output_dir / "stage1_validation.png"
    if old_plot.exists():
        old_plot.replace(output_dir / "stage1b_validation.png")
    gate = "PASS" if metrics["gate_passed"] else "FAIL"
    print("FINAL_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
    print(f"STAGE1B_GATE {gate}", flush=True)
    print(f"Artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

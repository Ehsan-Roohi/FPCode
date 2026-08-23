#!/usr/bin/env python3
"""Heat-flux G1 stage: structure-preserving PINN on a deterministic quadrature.

Differences from train_stage2.py (G0), each motivated by the audit of commit
97f9ea8 / checkpoint epoch-12500:

1. Deterministic (cx, rho) quadrature instead of importance-sampled Monte
   Carlo clouds.  The G0 evaluator's 3.6 % <-> 6.4 % discrepancy was sampling
   noise of the evaluation cloud (one-sigma ~4 pp); here every integral is a
   spectrally accurate quadrature, so the loss and all gate numbers are
   reproducible to round-off.
2. Mass, momentum and energy are exact at every time by an exponential tilt
   (structure_model.StructuredDensityModel).  The G0 optimiser had traded
   3-4.5 % invariant drift against the pointwise residual; that trade is no
   longer available.
3. The loss is the f-weighted mean square of the log residual *only*.  No
   soft invariant penalties, no weak third-moment loss, no finite-difference
   heat-flux-rate loss, no analytic-history loss: the (4/3) nu decay of Qx is
   the quantity under test and must not enter training.  (An ablation flag
   re-enables the operator's own weak heat-flux identity; it stays off in the
   qualification run.)
4. An explicit third-Hermite heat-flux mode b(t) cx(|c|^2 - 5) with b(0) = 0.
5. Random per-epoch shift of the cx grid (a continuum of quadrature panels)
   and a finer, wider held-out grid for evaluation.

The network, the bridge ansatz, the exact initial condition, the 9x9 closure
and the residual operator are those of train_stage2.py, so a G0 weight file
can be loaded as a warm start (--resume-base-weights).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for path in (str(PARENT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_stage2 import Config as BaseConfig  # noqa: E402
from axisym_quadrature import build_quadrature  # noqa: E402
from structure_model import (  # noqa: E402
    G1Config,
    QuadratureTensors,
    StructuredDensityModel,
    assemble_slices,
    heat_flux_rate_residual,
    invariant_features_tf,
    log_residual,
    quadrature_tensors,
    weighted_residual_loss,
)
from evaluate_g1 import evaluate_model  # noqa: E402


@dataclass
class G1TrainConfig:
    output_dir: str
    reference: str
    epochs: int = 20_000
    n_time_batch: int = 16
    time_power: float = 1.0
    n_cx: int = 129
    n_rho: int = 32
    cx_half_width: float = 8.0
    rho_max: float = 8.0
    shift_cx_grid: bool = True
    eval_n_cx: int = 257
    eval_n_rho: int = 64
    eval_cx_half_width: float = 9.0
    eval_rho_max: float = 9.0
    width: int = 128
    depth: int = 5
    learning_rate: float = 2.0e-4
    lr_decay_steps: int = 7_000
    lr_decay_rate: float = 0.3
    gradient_clip_norm: float = 1.0
    correction_cap: float = 12.0
    bridge_rate: float = 1.0
    closure_regularization: float = 1.0e-7
    stop_gradient_closure: bool = False
    use_heat_flux_mode: bool = True
    heat_flux_mode_cap: float = 0.02
    heat_flux_mode_width: int = 16
    tilt_newton_steps: int = 4
    tilt_beta_cap: float = 0.5
    tilt_penalty_weight: float = 10.0       # gauge fixing: keeps the raw network's own invariants near exact
    heat_flux_rate_weight: float = 0.0      # ablation only; 0 in the qualification run
    nu: float = 1.0
    tmax: float = 1.0
    print_every: int = 100
    checkpoint_every: int = 1_000
    seed: int = 20260901
    resume_base_weights: str | None = None  # G0 DensityModel weights (warm start)
    resume_weights: str | None = None       # G1 StructuredDensityModel weights
    evaluate_only: bool = False
    strict_gate: bool = False


def parse_args() -> G1TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = G1TrainConfig(output_dir="", reference="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference", required=True, help="axisym_fp_reference.py reference.npz")
    for name, field in G1TrainConfig.__dataclass_fields__.items():
        if name in ("output_dir", "reference"):
            continue
        default = getattr(defaults, name)
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, dest=name, action="store_true", default=default)
            parser.add_argument("--no-" + name.replace("_", "-"), dest=name, action="store_false")
        elif default is None:
            parser.add_argument(flag, dest=name, default=None)
        else:
            parser.add_argument(flag, dest=name, type=type(default), default=default)
    args = parser.parse_args()
    return G1TrainConfig(**vars(args))


def base_config_from(config: G1TrainConfig) -> BaseConfig:
    return BaseConfig(
        case="heat_flux", output_dir=config.output_dir, reference=config.reference,
        width=config.width, depth=config.depth, correction_cap=config.correction_cap,
        bridge_rate=config.bridge_rate, closure_regularization=config.closure_regularization,
        axisymmetric_heat_flux=True, antithetic_heat_flux_quadrature=False,
        stop_gradient_closure=config.stop_gradient_closure, nu=config.nu, tmax=config.tmax,
        seed=config.seed,
    )


def g1_config_from(config: G1TrainConfig) -> G1Config:
    return G1Config(
        use_heat_flux_mode=config.use_heat_flux_mode, heat_flux_mode_cap=config.heat_flux_mode_cap,
        heat_flux_mode_width=config.heat_flux_mode_width, tilt_newton_steps=config.tilt_newton_steps,
        tilt_beta_cap=config.tilt_beta_cap, tmax=config.tmax, nu=config.nu,
    )


def shifted_quadrature(base: QuadratureTensors, shift: tf.Tensor, spacing: float) -> QuadratureTensors:
    """Translate the cx grid by shift * spacing (weights unchanged)."""
    offset = tf.stack([tf.cast(shift, tf.float64) * spacing, 0.0, 0.0])
    nodes64 = base.nodes64 + offset[None, :]
    r2 = tf.reduce_sum(tf.square(nodes64), axis=1)
    return QuadratureTensors(
        nodes32=tf.cast(nodes64, tf.float32), nodes64=nodes64, weights64=base.weights64,
        psi64=invariant_features_tf(nodes64), phi64=nodes64[:, 0] * r2, size=base.size,
    )


def make_train_step(model: StructuredDensityModel, optimizer, config: G1TrainConfig, base_quad: QuadratureTensors):
    spacing = 2.0 * config.cx_half_width / (config.n_cx - 1)
    nt = config.n_time_batch

    @tf.function(reduce_retracing=True)
    def train_step() -> dict[str, tf.Tensor]:
        # Stratified random times: one sample in each of nt equal strata of
        # [0, tmax] (optionally warped towards t = 0 with time_power > 1).
        strata = (tf.cast(tf.range(nt), tf.float32) + tf.random.uniform([nt])) / nt
        t_slices = config.tmax * tf.pow(strata, config.time_power)
        if config.shift_cx_grid:
            quad = shifted_quadrature(base_quad, tf.random.uniform([]), spacing)
        else:
            quad = base_quad
        with tf.GradientTape() as tape:
            state = assemble_slices(model, t_slices, quad)
            residual = log_residual(state, quad, config.nu, config.stop_gradient_closure)
            pde_loss = weighted_residual_loss(state, residual)
            # Gauge fixing.  log f = log f~ + beta.psi is invariant under
            # log f~ -> log f~ + gamma(t).psi, beta -> beta - gamma, so the PDE
            # loss has three flat directions per time slice along which the
            # raw network's own mass/momentum/energy can wander while the
            # tilt silently repairs them.  Penalising |beta|^2 removes the
            # flat directions without constraining f at all: every density
            # with exact invariants is representable with beta = 0.
            tilt_loss = tf.cast(tf.reduce_mean(tf.reduce_sum(tf.square(state.beta), axis=1)), tf.float32)
            if config.heat_flux_rate_weight > 0.0:
                _, rate_residual = heat_flux_rate_residual(state, quad, config.nu)
                rate_loss = tf.cast(tf.reduce_mean(tf.square(rate_residual / (4.0 / 3.0 * config.nu * 0.25))), tf.float32)
            else:
                rate_loss = tf.constant(0.0, tf.float32)
            total = pde_loss + config.tilt_penalty_weight * tilt_loss + config.heat_flux_rate_weight * rate_loss
            tf.debugging.assert_all_finite(total, "Non-finite G1 training loss")
        gradients = tape.gradient(total, model.trainable_variables)
        if any(gradient is None for gradient in gradients):
            missing = [variable.name for gradient, variable in zip(gradients, model.trainable_variables) if gradient is None]
            raise RuntimeError("Disconnected trainable variables: " + ", ".join(missing))
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Non-finite G1 gradient")
        clipped, grad_norm = tf.clip_by_global_norm(gradients, config.gradient_clip_norm)
        optimizer.apply_gradients(zip(clipped, model.trainable_variables))
        raw = state.raw_moments64
        return {
            "total": total, "pde": pde_loss, "tilt": tilt_loss, "heat_flux_rate": rate_loss, "grad_norm": grad_norm,
            "max_abs_beta": tf.cast(tf.reduce_max(tf.abs(state.beta)), tf.float32),
            "max_abs_beta_rate": tf.cast(tf.reduce_max(tf.abs(state.beta_rate)), tf.float32),
            "raw_mass_drift": tf.cast(tf.reduce_max(tf.abs(raw["mass"] - 1.0)), tf.float32),
            "raw_energy_drift": tf.cast(tf.reduce_max(tf.abs(raw["dm2"] - 3.0)), tf.float32),
            "max_coefficient": tf.reduce_max(tf.abs(state.coefficients)),
            "max_abs_lambda": tf.reduce_max(tf.abs(state.lam)),
            "max_stress_anisotropy": tf.reduce_max(tf.abs(state.moments["pij"][:, 0] - state.moments["pij"][:, 3])),
        }

    return train_step


def portable_reload_check(model: StructuredDensityModel, weights: Path, base: BaseConfig, g1: G1Config) -> float:
    rng = np.random.default_rng(base.seed + 1)
    points = np.stack([rng.normal(size=2048) * 1.5, np.abs(rng.normal(size=2048)) * 1.5, np.zeros(2048)], axis=1)
    t = tf.fill([2048, 1], 0.731)
    before = model.raw_log_density(t, tf.constant(points, tf.float32)).numpy()
    reloaded = StructuredDensityModel(base, g1)
    reloaded.build_model()
    reloaded.load_weights(str(weights))
    after = reloaded.raw_log_density(t, tf.constant(points, tf.float32)).numpy()
    return float(np.max(np.abs(before - after)))


def main() -> None:
    config = parse_args()
    tf.keras.backend.set_floatx("float32")
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = base_config_from(config)
    g1 = g1_config_from(config)
    full_config: dict[str, Any] = {**asdict(config), **asdict(g1), "base_config": asdict(base)}
    (output / "config.json").write_text(json.dumps(full_config, indent=2) + "\n")

    model = StructuredDensityModel(base, g1)
    model.build_model()
    if config.resume_base_weights and config.resume_weights:
        raise SystemExit("set only one of --resume-base-weights and --resume-weights")
    if config.resume_base_weights:
        model.base.load_weights(config.resume_base_weights)
        print(f"Warm start: loaded G0 DensityModel weights into model.base from {config.resume_base_weights}", flush=True)
    if config.resume_weights:
        model.load_weights(config.resume_weights)
        print(f"Loaded G1 weights: {config.resume_weights}", flush=True)

    base_quad = quadrature_tensors(build_quadrature(
        cx_half_width=config.cx_half_width, n_cx=config.n_cx, rho_max=config.rho_max, n_rho=config.n_rho,
    ))
    print(f"Training quadrature: {config.n_cx} x {config.n_rho} = {base_quad.size} nodes, "
          f"{config.n_time_batch} time slices per step -> {base_quad.size * config.n_time_batch} residual points", flush=True)

    checkpoints = output / "checkpoints_h5"
    checkpoints.mkdir(exist_ok=True)
    if not config.evaluate_only:
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            config.learning_rate, config.lr_decay_steps, config.lr_decay_rate, staircase=True,
        )
        optimizer = tf.keras.optimizers.Adam(schedule)
        train_step = make_train_step(model, optimizer, config, base_quad)
        model.save_weights(checkpoints / "epoch-000000.weights.h5")
        history: list[dict[str, float]] = []
        started = time.perf_counter()
        for epoch in range(1, config.epochs + 1):
            result = {key: float(value.numpy()) for key, value in train_step().items()}
            result["epoch"] = epoch
            result["elapsed_s"] = time.perf_counter() - started
            history.append(result)
            if epoch == 1 or epoch % config.print_every == 0:
                print("g1 " + " ".join([
                    f"epoch={epoch:6d}", f"pde={result['pde']:.3e}", f"tilt={result['tilt']:.2e}", f"grad={result['grad_norm']:.2e}",
                    f"beta={result['max_abs_beta']:.2e}", f"dbeta={result['max_abs_beta_rate']:.2e}",
                    f"rawmass={result['raw_mass_drift']:.2e}", f"rawenergy={result['raw_energy_drift']:.2e}",
                    f"aniso={result['max_stress_anisotropy']:.2e}", f"coef={result['max_coefficient']:.2e}",
                    f"elapsed={result['elapsed_s']:.0f}s",
                ]), flush=True)
            if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
                model.save_weights(checkpoints / f"epoch-{epoch:06d}.weights.h5")
        with (output / "loss_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    final_weights = output / "g1_final.weights.h5"
    model.save_weights(final_weights)
    model.base.save_weights(output / "g1_final_base_only.weights.h5")   # G0-compatible raw network
    reload_linf = portable_reload_check(model, final_weights, base, g1)
    print(f"Portable reload L_inf: {reload_linf:.3e}", flush=True)
    metrics = evaluate_model(model, full_config, config.reference, output / "final_evaluation",
                             portable_reload_linf=reload_linf)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print("G1_FINAL_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
    print(f"G1_STATUS {metrics['status']}", flush=True)
    print(f"Artifacts: {output}", flush=True)
    if config.strict_gate and metrics["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

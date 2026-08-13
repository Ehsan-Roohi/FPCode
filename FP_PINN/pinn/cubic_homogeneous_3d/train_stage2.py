#!/usr/bin/env python3
"""Self-consistent PINN for homogeneous 3-D cubic Fokker--Planck relaxation.

The network represents log(f), uses an exact initial-condition ansatz, and
recomputes the nine closure coefficients plus the analytic cubic coefficient
from differentiable importance-sampled moments at every optimizer step.  No
particle-reference values enter the training loss; the reference is used only
for the final independent validation gate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cubic_operator import (
    CASE_NAMES,
    equilibrium_logpdf,
    initial_logpdf,
    moments_from_samples,
    proposal_logpdf,
    sample_proposal,
    solve_closure,
)


@dataclass
class Config:
    case: str
    output_dir: str
    reference: str
    epochs: int = 30_000
    n_time_batch: int = 8
    n_velocity_per_time: int = 1024
    width: int = 128
    depth: int = 5
    learning_rate: float = 5.0e-4
    lr_decay_steps: int = 10_000
    lr_decay_rate: float = 0.3
    gradient_clip_norm: float = 5.0
    correction_cap: float = 12.0
    bridge_rate: float = 1.0
    pde_weight: float = 1.0
    mass_weight: float = 30.0
    momentum_weight: float = 20.0
    energy_weight: float = 20.0
    heat_flux_weight: float = 10.0
    heat_flux_scale: float = 0.25
    tail_fraction: float = 0.20
    tail_variance: float = 4.0
    fixed_velocity_quadrature: bool = False
    axisymmetric_heat_flux: bool = True
    antithetic_heat_flux_quadrature: bool = True
    stop_gradient_closure: bool = False
    closure_regularization: float = 1.0e-7
    nu: float = 1.0
    tmax: float = 1.0
    print_every: int = 250
    checkpoint_every: int = 2500
    evaluation_samples: int = 65_536
    marginal_quadrature_order: int = 18
    seed: int = 20260808
    resume_weights: str | None = None
    evaluate_only: bool = False
    strict_gate: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_NAMES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--epochs", type=int, default=30_000)
    parser.add_argument("--n-time-batch", type=int, default=8)
    parser.add_argument("--n-velocity-per-time", type=int, default=1024)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--lr-decay-steps", type=int, default=10_000)
    parser.add_argument("--lr-decay-rate", type=float, default=0.3)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--correction-cap", type=float, default=12.0)
    parser.add_argument("--bridge-rate", type=float, default=1.0)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--mass-weight", type=float, default=30.0)
    parser.add_argument("--momentum-weight", type=float, default=20.0)
    parser.add_argument("--energy-weight", type=float, default=20.0)
    parser.add_argument(
        "--heat-flux-weight", type=float, default=10.0,
        help="Weight of the particle-free weak third-moment residual",
    )
    parser.add_argument(
        "--heat-flux-scale", type=float, default=0.25,
        help="Reference scale used to nondimensionalize the weak heat-flux residual",
    )
    parser.add_argument(
        "--tail-fraction", type=float, default=0.20,
        help="Fraction of training velocities drawn from the broad tail proposal",
    )
    parser.add_argument("--tail-variance", type=float, default=4.0)
    parser.add_argument(
        "--fixed-velocity-quadrature", action="store_true",
        help="Reuse one deterministic velocity cloud to reduce closure noise",
    )
    parser.add_argument(
        "--axisymmetric-heat-flux",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict the heat-flux correction to (t,cx,cy^2+cz^2)",
    )
    parser.add_argument(
        "--antithetic-heat-flux-quadrature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the four transverse sign reflections of every heat-flux velocity",
    )
    parser.add_argument(
        "--stop-gradient-closure", action="store_true",
        help="Use the old Picard-style closure instead of differentiating the 9x9 solve",
    )
    parser.add_argument("--closure-regularization", type=float, default=1.0e-7)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=2500)
    parser.add_argument("--evaluation-samples", type=int, default=65_536)
    parser.add_argument("--marginal-quadrature-order", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--resume-weights")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--strict-gate", action="store_true",
        help="Return exit status 2 when a numerical validation gate fails",
    )
    config = Config(**vars(parser.parse_args()))
    if not 0.0 < config.tail_fraction < 1.0:
        parser.error("--tail-fraction must lie strictly between zero and one")
    if config.tail_variance <= 1.0:
        parser.error("--tail-variance must be greater than one")
    if config.heat_flux_scale <= 0.0:
        parser.error("--heat-flux-scale must be positive")
    if (
        config.case == "heat_flux"
        and config.antithetic_heat_flux_quadrature
        and config.n_velocity_per_time % 4
    ):
        parser.error("--n-velocity-per-time must be divisible by four for antithetic heat-flux quadrature")
    if (
        config.case == "heat_flux"
        and config.antithetic_heat_flux_quadrature
        and config.evaluation_samples % 4
    ):
        parser.error("--evaluation-samples must be divisible by four for antithetic heat-flux quadrature")
    return config


def _tf_normal_logpdf(x: tf.Tensor, mean: float, variance: float) -> tf.Tensor:
    dtype = x.dtype
    return -0.5 * (
        tf.math.log(tf.cast(2.0 * np.pi * variance, dtype))
        + tf.square(x - tf.cast(mean, dtype)) / tf.cast(variance, dtype)
    )


def tf_equilibrium_logpdf(c: tf.Tensor) -> tf.Tensor:
    return -1.5 * tf.math.log(tf.cast(2.0 * np.pi, c.dtype)) - 0.5 * tf.reduce_sum(
        tf.square(c), axis=-1, keepdims=True
    )


def tf_initial_logpdf(case: str, c: tf.Tensor) -> tf.Tensor:
    if case == "equilibrium":
        return tf_equilibrium_logpdf(c)
    if case == "stress":
        variances = tf.constant([1.6, 0.9, 0.5], dtype=c.dtype)
        return tf.reduce_sum(
            -0.5 * (
                tf.math.log(tf.cast(2.0 * np.pi, c.dtype) * variances)
                + tf.square(c) / variances
            ), axis=-1, keepdims=True,
        )
    log_a = _tf_normal_logpdf(c[:, 0:1], 1.0, 0.5) + tf.math.log(
        tf.cast(1.0 / 3.0, c.dtype)
    )
    log_b = _tf_normal_logpdf(c[:, 0:1], -0.5, 0.5) + tf.math.log(
        tf.cast(2.0 / 3.0, c.dtype)
    )
    log_x = tf.reduce_logsumexp(tf.concat([log_a, log_b], axis=1), axis=1, keepdims=True)
    return log_x + _tf_normal_logpdf(c[:, 1:2], 0.0, 1.0) + _tf_normal_logpdf(
        c[:, 2:3], 0.0, 1.0
    )


def tf_training_proposal_logpdf(config: Config, c: tf.Tensor) -> tf.Tensor:
    """Tail-enriched proposal used only for PINN quadrature.

    The independent particle solution is not used here.  Exact mixture
    reweighting keeps all moment estimates unbiased.
    """
    dtype = c.dtype
    tail_fraction = tf.cast(config.tail_fraction, dtype)
    core_fraction = 0.5 * (1.0 - tail_fraction)
    tail_variance = tf.cast(config.tail_variance, dtype)
    tail_logpdf = (
        -1.5 * tf.math.log(tf.cast(2.0 * np.pi, dtype) * tail_variance)
        - 0.5 * tf.reduce_sum(tf.square(c), axis=1, keepdims=True) / tail_variance
    )
    terms = tf.concat(
        [
            tf_initial_logpdf(config.case, c) + tf.math.log(core_fraction),
            tf_equilibrium_logpdf(c) + tf.math.log(core_fraction),
            tail_logpdf + tf.math.log(tail_fraction),
        ],
        axis=1,
    )
    return tf.reduce_logsumexp(terms, axis=1, keepdims=True)


class DensityModel(tf.keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden = [
            tf.keras.layers.Dense(
                config.width, activation=tf.nn.tanh,
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=config.seed + index),
            )
            for index in range(config.depth)
        ]
        self.final = tf.keras.layers.Dense(
            1, kernel_initializer="zeros", bias_initializer="zeros"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        value = inputs
        for layer in self.hidden:
            value = layer(value)
        return self.final(value)

    def log_density(self, t: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        t = tf.cast(t, tf.float32)
        c = tf.cast(c, tf.float32)
        r2 = tf.reduce_sum(tf.square(c), axis=1, keepdims=True)
        velocity_features = c / 3.0
        if self.config.case == "heat_flux" and self.config.axisymmetric_heat_flux:
            # The positive skew initial state and the homogeneous FP operator
            # are invariant under every rotation in the transverse y-z plane.
            # Keep the legacy five-feature shape so existing H5 weights remain
            # loadable, but remove the two symmetry-breaking coordinates.  The
            # correction then depends only on (t, cx, cy^2+cz^2).
            velocity_features = tf.concat(
                [velocity_features[:, 0:1], tf.zeros_like(velocity_features[:, 1:3])],
                axis=1,
            )
        features = tf.concat(
            [t / self.config.tmax, velocity_features, r2 / 9.0], axis=1
        )
        raw = self(features)
        correction = t * self.config.correction_cap * tf.tanh(
            raw / self.config.correction_cap
        )
        log_f0 = tf_initial_logpdf(self.config.case, c)
        log_m = tf_equilibrium_logpdf(c)
        alpha = 1.0 - tf.exp(-self.config.bridge_rate * t)
        alpha = tf.clip_by_value(alpha, 0.0, 1.0 - 1.0e-7)
        # Evaluate the bridge mixture entirely in log space.  The previous
        # exp -> weighted sum -> log path produced non-finite second
        # derivatives in broad-tail quadrature before the density underflowed.
        log_one_minus_alpha = tf.math.log1p(-alpha)
        safe_alpha = tf.maximum(
            alpha, tf.constant(1.0e-30, dtype=alpha.dtype)
        )
        log_density_base = tf.reduce_logsumexp(
            tf.concat(
                [
                    log_f0 + log_one_minus_alpha,
                    log_m + tf.math.log(safe_alpha),
                ],
                axis=1,
            ),
            axis=1,
            keepdims=True,
        )
        # Preserve the exact initial-condition ansatz at t=0.
        log_density_base = tf.where(alpha > 0.0, log_density_base, log_f0)
        return log_density_base + correction


def sample_tf_proposal(
    config: Config,
    velocity_grid: tf.Tensor | None = None,
    velocity_log_q: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    nt, nv = config.n_time_batch, config.n_velocity_per_time
    total = nt * nv
    # Half uniform and half early-time-focused samples reduce the initial layer error.
    n_first = nt // 2
    times = tf.concat(
        [tf.random.uniform((n_first, 1), 0.0, config.tmax),
         config.tmax * tf.square(tf.random.uniform((nt - n_first, 1), 0.0, 1.0))],
        axis=0,
    )
    times = tf.sort(times, axis=0)
    if velocity_grid is None:
        antithetic = (
            config.case == "heat_flux" and config.antithetic_heat_flux_quadrature
        )
        sample_total = total // 4 if antithetic else total
        selector = tf.random.uniform((sample_total, 1))
        core_fraction = 0.5 * (1.0 - config.tail_fraction)
        choose_initial = selector < core_fraction
        choose_equilibrium = tf.logical_and(
            selector >= core_fraction, selector < 2.0 * core_fraction
        )
        equilibrium = tf.random.normal((sample_total, 3))
        tail = tf.sqrt(tf.cast(config.tail_variance, tf.float32)) * tf.random.normal(
            (sample_total, 3)
        )
        if config.case == "equilibrium":
            initial = tf.random.normal((sample_total, 3))
        elif config.case == "stress":
            initial = tf.random.normal((sample_total, 3)) * tf.sqrt(
                tf.constant([1.6, 0.9, 0.5], dtype=tf.float32)
            )
        else:
            choose_a = tf.random.uniform((sample_total, 1)) < (1.0 / 3.0)
            x_mean = tf.where(choose_a, 1.0, -0.5)
            initial = tf.concat(
                [x_mean + tf.sqrt(0.5) * tf.random.normal((sample_total, 1)),
                 tf.random.normal((sample_total, 2))], axis=1,
            )
        c = tf.where(choose_initial, initial, tf.where(choose_equilibrium, equilibrium, tail))
        if antithetic:
            c = tf.reshape(c, (nt, nv // 4, 3))
            c = tf.concat(
                [
                    c,
                    c * tf.constant([1.0, -1.0, 1.0], dtype=c.dtype),
                    c * tf.constant([1.0, 1.0, -1.0], dtype=c.dtype),
                    c * tf.constant([1.0, -1.0, -1.0], dtype=c.dtype),
                ],
                axis=1,
            )
        else:
            c = tf.reshape(c, (nt, nv, 3))
        flat_c = tf.reshape(c, (-1, 3))
        log_q = tf.reshape(
            tf_training_proposal_logpdf(config, flat_c), (nt, nv, 1)
        )
    else:
        c = velocity_grid
        if velocity_log_q is None:
            raise ValueError("velocity_log_q is required with velocity_grid")
        log_q = velocity_log_q
    t = tf.repeat(times[:, None, :], repeats=nv, axis=1)
    return t, c, log_q


def moment_tensors(c: tf.Tensor, ratio: tf.Tensor) -> dict[str, tf.Tensor]:
    # c: [nt,nv,3], ratio=f/q: [nt,nv,1].
    mass = tf.reduce_mean(ratio, axis=1)[:, 0]
    normalized = ratio / tf.maximum(tf.reduce_sum(ratio, axis=1, keepdims=True), 1.0e-20)
    mean = tf.reduce_sum(normalized * c, axis=1)
    v = c - mean[:, None, :]
    x, y, z = v[:, :, 0], v[:, :, 1], v[:, :, 2]
    r2 = tf.reduce_sum(tf.square(v), axis=2)
    w = normalized[:, :, 0]

    def avg(value: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(w * value, axis=1)

    pij = tf.stack(
        [avg(x*x), avg(x*y), avg(x*z), avg(y*y), avg(y*z), avg(z*z)], axis=1
    )
    q = tf.stack([avg(x*r2), avg(y*r2), avg(z*r2)], axis=1)
    m3 = tf.stack(
        [avg(x**3), avg(x*x*y), avg(x*x*z), avg(x*y*y), avg(x*y*z),
         avg(x*z*z), avg(y**3), avg(y*y*z), avg(y*z*z), avg(z**3)], axis=1
    )
    m4 = tf.stack(
        [avg(x*x*r2), avg(x*y*r2), avg(x*z*r2), avg(y*y*r2),
         avg(y*z*r2), avg(z*z*r2)], axis=1
    )
    r4 = r2*r2
    m5 = tf.stack([avg(x*r4), avg(y*r4), avg(z*r4)], axis=1)
    dm2 = avg(r2)
    dm4 = avg(r4)
    return {
        "mass": mass, "mean": mean, "pij": pij, "q": q, "m3": m3,
        "m4": m4, "m5": m5, "dm2": dm2, "dm4": dm4,
    }


def closure_tf(m: dict[str, tf.Tensor], config: Config) -> tuple[tf.Tensor, tf.Tensor]:
    p, q, m3, m4, m5 = m["pij"], m["q"], m["m3"], m["m4"], m["m5"]
    d2, d4 = m["dm2"], m["dm4"]
    z = tf.zeros_like(d2)
    rows = [
        tf.stack([2*p[:,0],2*p[:,1],2*p[:,2],z,z,z,2*q[:,0],z,z],1),
        tf.stack([p[:,1],p[:,0]+p[:,3],p[:,4],p[:,1],p[:,2],z,q[:,1],q[:,0],z],1),
        tf.stack([p[:,2],p[:,4],p[:,0]+p[:,5],z,p[:,1],p[:,2],q[:,2],z,q[:,0]],1),
        tf.stack([z,2*p[:,1],z,2*p[:,3],2*p[:,4],z,z,2*q[:,1],z],1),
        tf.stack([z,p[:,2],p[:,1],p[:,4],p[:,3]+p[:,5],p[:,4],z,q[:,2],q[:,1]],1),
        tf.stack([z,z,2*p[:,2],z,2*p[:,4],2*p[:,5],z,z,2*q[:,2]],1),
        tf.stack([q[:,0]+2*m3[:,0],q[:,1]+4*m3[:,1],q[:,2]+4*m3[:,2],
                  2*m3[:,3],4*m3[:,4],2*m3[:,5],
                  d4-d2*d2+2*m4[:,0]-2*d2*p[:,0],
                  2*m4[:,1]-2*d2*p[:,1],2*m4[:,2]-2*d2*p[:,2]],1),
        tf.stack([2*m3[:,1],q[:,0]+4*m3[:,3],4*m3[:,4],q[:,1]+2*m3[:,6],
                  q[:,2]+4*m3[:,7],2*m3[:,8],2*m4[:,1]-2*d2*p[:,1],
                  d4-d2*d2+2*m4[:,3]-2*d2*p[:,3],2*m4[:,4]-2*d2*p[:,4]],1),
        tf.stack([2*m3[:,2],4*m3[:,4],q[:,0]+4*m3[:,5],2*m3[:,7],
                  q[:,1]+4*m3[:,8],q[:,2]+2*m3[:,9],2*m4[:,2]-2*d2*p[:,2],
                  2*m4[:,4]-2*d2*p[:,4],d4-d2*d2+2*m4[:,5]-2*d2*p[:,5]],1),
    ]
    lhs = tf.stack(rows, axis=1)
    third = d2 / 3.0
    dev2 = (
        tf.square(p[:,0]-third)+tf.square(p[:,3]-third)+tf.square(p[:,5]-third)
        +2.0*(tf.square(p[:,1])+tf.square(p[:,2])+tf.square(p[:,4]))
    )
    tiny = tf.cast(1.0e-12, lhs.dtype)
    lam = -config.nu * dev2 / tf.maximum(d2, tiny) ** 3.5
    rhs0 = -2.0 * lam[:, None] * m4
    rq0 = -lam*(3*m5[:,0]-d2*q[:,0]-2*(p[:,0]*q[:,0]+p[:,1]*q[:,1]+p[:,2]*q[:,2]))
    rq1 = -lam*(3*m5[:,1]-d2*q[:,1]-2*(p[:,1]*q[:,0]+p[:,3]*q[:,1]+p[:,4]*q[:,2]))
    rq2 = -lam*(3*m5[:,2]-d2*q[:,2]-2*(p[:,2]*q[:,0]+p[:,4]*q[:,1]+p[:,5]*q[:,2]))
    # nubol=2*nu, consistent with tau=1/nu=2*mu/p in the legacy solver.
    q_rate = (3.0*config.nu - (4.0/3.0)*config.nu) * q
    rhs = tf.concat([rhs0, tf.stack([rq0,rq1,rq2],axis=1)+q_rate], axis=1)
    one = tf.cast(1.0, lhs.dtype)
    scale = tf.maximum(one, tf.linalg.norm(lhs, axis=[1,2]) / 9.0)
    system = lhs + config.closure_regularization * scale[:,None,None] * tf.eye(
        9, batch_shape=tf.shape(lhs)[:1], dtype=lhs.dtype
    )
    vector = tf.linalg.solve(system, rhs[:,:,None])[:,:,0]
    vector = tf.clip_by_value(
        vector, tf.cast(-25.0, vector.dtype), tf.cast(25.0, vector.dtype)
    )
    return vector, lam


def _matrix_from_vector(vector: tf.Tensor) -> tf.Tensor:
    return tf.stack(
        [tf.stack([vector[:,0],vector[:,1],vector[:,2]],1),
         tf.stack([vector[:,1],vector[:,3],vector[:,4]],1),
         tf.stack([vector[:,2],vector[:,4],vector[:,5]],1)], axis=1,
    )


def weak_heat_flux_loss(
    c: tf.Tensor,
    ratio: tf.Tensor,
    relative_residual: tf.Tensor,
    mean: tf.Tensor,
    scale: float,
) -> tf.Tensor:
    """Squared weak FP defect for the third central moment Q=<v |v|^2>.

    This is a projection of the PDE residual itself, not a fit to the particle
    history.  Per-time normalization prevents late-time small-Q samples from
    being hidden by the bulk density residual.
    """
    normalized = ratio / tf.maximum(
        tf.reduce_sum(ratio, axis=1, keepdims=True),
        tf.cast(1.0e-30, ratio.dtype),
    )
    peculiar = c - mean[:, None, :]
    r2 = tf.reduce_sum(tf.square(peculiar), axis=2, keepdims=True)
    heat_basis = peculiar * r2
    defect = tf.reduce_sum(
        normalized * heat_basis * relative_residual, axis=1
    )
    return tf.reduce_mean(
        tf.reduce_sum(tf.square(defect / tf.cast(scale, defect.dtype)), axis=1)
    )


def make_train_step(model: DensityModel, optimizer: tf.keras.optimizers.Optimizer, config: Config):
    fixed_c: tf.Tensor | None = None
    fixed_log_q: tf.Tensor | None = None
    if config.fixed_velocity_quadrature:
        # Common random numbers remove optimizer-to-optimizer closure jitter.
        _, fixed_c, fixed_log_q = sample_tf_proposal(config)
        fixed_c = tf.stop_gradient(fixed_c)
        fixed_log_q = tf.stop_gradient(fixed_log_q)

    @tf.function(reduce_retracing=True)
    def train_step() -> dict[str, tf.Tensor]:
        t_grid, c_grid, log_q = sample_tf_proposal(config, fixed_c, fixed_log_q)
        flat_t = tf.reshape(t_grid, (-1,1))
        flat_c = tf.reshape(c_grid, (-1,3))
        with tf.GradientTape() as parameter_tape:
            with tf.GradientTape(persistent=True) as second_tape:
                second_tape.watch([flat_t, flat_c])
                with tf.GradientTape(persistent=True) as first_tape:
                    first_tape.watch([flat_t, flat_c])
                    log_f = model.log_density(flat_t, flat_c)
                h_t = first_tape.gradient(log_f, flat_t)
                grad_h = first_tape.gradient(log_f, flat_c)
                del first_tape
            hessian_h = second_tape.batch_jacobian(
                grad_h, flat_c, experimental_use_pfor=True
            )
            lap_h = tf.linalg.trace(hessian_h)[:, None]
            tf.debugging.assert_all_finite(
                lap_h, "Non-finite velocity-space Laplacian of log density"
            )
            del second_tape
            ratio = tf.exp(
                tf.clip_by_value(tf.reshape(log_f, tf.shape(log_q)) - log_q, -40.0, 40.0)
            )
            # Fifth-order moments and the small 9x9 solve are accumulated in
            # float64; the network and pointwise derivatives remain float32.
            moments64 = moment_tensors(
                tf.cast(c_grid, tf.float64), tf.cast(ratio, tf.float64)
            )
            coefficients64, lam64 = closure_tf(moments64, config)
            moments = {
                name: tf.cast(value, tf.float32)
                for name, value in moments64.items()
            }
            coefficients = tf.cast(coefficients64, tf.float32)
            lam = tf.cast(lam64, tf.float32)
            if config.stop_gradient_closure:
                coefficients_used = tf.stop_gradient(coefficients)
                lam_used = tf.stop_gradient(lam)
            else:
                coefficients_used = coefficients
                lam_used = lam
            matrix = _matrix_from_vector(coefficients_used)
            gamma = coefficients_used[:,6:9]
            peculiar = c_grid - moments["mean"][:,None,:]
            r2 = tf.reduce_sum(tf.square(peculiar), axis=2)
            linear = tf.einsum("tij,tvj->tvi", matrix, peculiar)
            nonlinear = (
                linear + gamma[:,None,:]*(r2-moments["dm2"][:,None])[:,:,None]
                + lam_used[:,None,None]*(peculiar*r2[:,:,None]-moments["q"][:,None,:])
            )
            drift = -config.nu*peculiar + nonlinear
            divergence = (
                -3.0*config.nu + tf.linalg.trace(matrix)[:,None]
                +2.0*tf.reduce_sum(gamma[:,None,:]*peculiar,axis=2)
                +5.0*lam_used[:,None]*r2
            )
            flat_drift = tf.reshape(drift, (-1,3))
            diffusion = tf.repeat(
                config.nu * moments["dm2"] / 3.0,
                repeats=config.n_velocity_per_time,
            )[:, None]
            relative_residual = (
                h_t + tf.reshape(divergence,(-1,1))
                + tf.reduce_sum(flat_drift*grad_h,axis=1,keepdims=True)
                - diffusion * (lap_h+tf.reduce_sum(tf.square(grad_h),axis=1,keepdims=True))
            )
            relative_residual = tf.reshape(relative_residual, tf.shape(ratio))
            importance = tf.stop_gradient(ratio / tf.maximum(tf.reduce_mean(ratio),1.0e-12))
            delta = tf.constant(5.0, tf.float32)
            pde_loss = tf.reduce_mean(
                importance * tf.square(delta) *
                (tf.sqrt(1.0+tf.square(relative_residual/delta))-1.0)
            )
            if config.case == "heat_flux":
                heat_loss = weak_heat_flux_loss(
                    c_grid, ratio, relative_residual, moments["mean"],
                    config.heat_flux_scale,
                )
            else:
                heat_loss = tf.constant(0.0, tf.float32)
            mass_loss = tf.reduce_mean(tf.square(moments["mass"]-1.0))
            momentum_loss = tf.reduce_mean(tf.square(moments["mean"]))
            energy_loss = tf.reduce_mean(tf.square(moments["dm2"]-3.0))
            total = (
                config.pde_weight*pde_loss + config.mass_weight*mass_loss
                +config.momentum_weight*momentum_loss+config.energy_weight*energy_loss
                +config.heat_flux_weight*heat_loss
            )
        gradients = parameter_tape.gradient(total, model.trainable_variables)
        finite_gradients = [tf.where(tf.math.is_finite(g),g,tf.zeros_like(g)) for g in gradients]
        finite_gradients, grad_norm = tf.clip_by_global_norm(
            finite_gradients, config.gradient_clip_norm
        )
        optimizer.apply_gradients(zip(finite_gradients, model.trainable_variables))
        return {
            "total": total, "pde": pde_loss, "mass": mass_loss,
            "momentum": momentum_loss, "energy": energy_loss, "heat_flux": heat_loss,
            "grad_norm": grad_norm, "max_coefficient": tf.reduce_max(tf.abs(coefficients)),
            "max_abs_lambda": tf.reduce_max(tf.abs(lam)),
        }
    return train_step


def _predict_log_density(model: DensityModel, t: float, c: np.ndarray, batch: int = 65_536) -> np.ndarray:
    pieces = []
    for start in range(0, c.shape[0], batch):
        values = c[start:start+batch].astype(np.float32)
        times = np.full((values.shape[0],1), t, dtype=np.float32)
        pieces.append(model.log_density(times, values).numpy()[:,0])
    return np.concatenate(pieces)


def _evaluate_marginal(
    model: DensityModel, time_value: float, centers: np.ndarray, axis: int, order: int
) -> np.ndarray:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    transverse = np.sqrt(2.0) * nodes
    a, b = np.meshgrid(transverse, transverse, indexing="ij")
    quad_weight = 2.0 * np.outer(weights, weights) * np.exp(
        nodes[:,None]**2 + nodes[None,:]**2
    )
    result = np.empty_like(centers)
    other = [index for index in range(3) if index != axis]
    for index, center in enumerate(centers):
        points = np.zeros((order*order,3),dtype=np.float64)
        points[:,axis] = center
        points[:,other[0]] = a.ravel()
        points[:,other[1]] = b.ravel()
        density = np.exp(_predict_log_density(model,time_value,points))
        result[index] = np.sum(quad_weight.ravel()*density)
    return result


def _evaluation_proposal(config: Config, rng: np.random.Generator) -> np.ndarray:
    """Return the independent validation proposal, paired when symmetry permits.

    Pairing is a quadrature variance reduction only.  It does not introduce
    particle-reference information into training or evaluation.
    """
    if config.case != "heat_flux" or not config.antithetic_heat_flux_quadrature:
        return sample_proposal(config.case, config.evaluation_samples, rng)
    base = sample_proposal(config.case, config.evaluation_samples // 4, rng)
    signs = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
         [1.0, 1.0, -1.0], [1.0, -1.0, -1.0]],
        dtype=np.float64,
    )
    return np.concatenate([base * sign for sign in signs], axis=0)


def evaluate(model: DensityModel, config: Config, output: Path) -> dict[str, Any]:
    reference = np.load(config.reference)
    times = reference["time"]
    rng = np.random.default_rng(config.seed + 9001)
    proposal = _evaluation_proposal(config, rng)
    log_q = proposal_logpdf(config.case, proposal)
    model_mass, model_mean, model_dm2, model_pij, model_q = [], [], [], [], []
    minimum_density = np.inf
    for time_value in times:
        log_f = _predict_log_density(model,float(time_value),proposal)
        density = np.exp(log_f)
        ratio = np.exp(np.clip(log_f-log_q,-100.0,100.0))
        weights = ratio / proposal.shape[0]
        moments = moments_from_samples(proposal,weights=weights)
        model_mass.append(float(np.mean(ratio)))
        model_mean.append(moments.mean)
        model_dm2.append(moments.dm2)
        model_pij.append(moments.pij)
        model_q.append(moments.q)
        minimum_density = min(minimum_density,float(np.min(density)))
    model_mean=np.asarray(model_mean); model_dm2=np.asarray(model_dm2)
    model_pij=np.asarray(model_pij); model_q=np.asarray(model_q); model_mass=np.asarray(model_mass)

    ref_pij, ref_q = reference["pij"], reference["q"]
    ref_dm2 = reference["dm2"]
    theta = model_dm2/3.0
    pred_dev=model_pij.copy(); ref_dev=ref_pij.copy()
    pred_dev[:,[0,3,5]]-=theta[:,None]
    ref_dev[:,[0,3,5]]-=ref_dm2[:,None]/3.0
    stress_error=float(np.linalg.norm(pred_dev-ref_dev)/max(np.linalg.norm(ref_dev),1.0e-12))
    heat_error=float(np.linalg.norm(model_q-ref_q)/max(np.linalg.norm(ref_q),1.0e-12))
    active_heat_error=heat_error
    transverse_heat_flux_relative=0.0
    reference_transverse_heat_flux_relative=0.0
    if config.case == "heat_flux":
        active_heat_error=float(
            np.linalg.norm(model_q[:,0]-ref_q[:,0])
            / max(np.linalg.norm(ref_q[:,0]),1.0e-12)
        )
        active_heat_scale=max(float(np.max(np.abs(ref_q[:,0]))),1.0e-12)
        transverse_heat_flux_relative=float(
            np.max(np.linalg.norm(model_q[:,1:3],axis=1))/active_heat_scale
        )
        reference_transverse_heat_flux_relative=float(
            np.max(np.linalg.norm(ref_q[:,1:3],axis=1))/active_heat_scale
        )

    centers=reference["histogram_centers"]
    selected=np.unique([0,len(times)//2,len(times)-1])
    pred_marginals=[]; ref_marginals=[]
    for time_index in selected:
        for axis,name in enumerate(("marginal_x","marginal_y","marginal_z")):
            pred_marginals.append(_evaluate_marginal(
                model,float(times[time_index]),centers,axis,config.marginal_quadrature_order
            ))
            ref_marginals.append(reference[name][time_index])
    pred_marginals=np.asarray(pred_marginals); ref_marginals=np.asarray(ref_marginals)
    marginal_error=float(
        np.linalg.norm(pred_marginals-ref_marginals)/max(np.linalg.norm(ref_marginals),1.0e-12)
    )

    initial_points=sample_proposal(config.case,4096,rng)
    initial_prediction=np.exp(_predict_log_density(model,0.0,initial_points))
    initial_exact=np.exp(initial_logpdf(config.case,initial_points))
    initial_linf=float(np.max(np.abs(initial_prediction-initial_exact)))
    relevant_error = stress_error if config.case == "stress" else active_heat_error
    relevant_threshold = 0.20 if config.case == "heat_flux" else 0.15
    if config.case == "equilibrium":
        relevant_error=max(float(np.max(np.linalg.norm(pred_dev,axis=1))),
                           float(np.max(np.linalg.norm(model_q,axis=1))))
        relevant_threshold=0.06
    checks={
        "marginal_relative_l2": marginal_error < (0.15 if config.case=="heat_flux" else 0.12),
        "max_mass_error": float(np.max(np.abs(model_mass-1.0))) < 0.03,
        "max_momentum_norm": float(np.max(np.linalg.norm(model_mean,axis=1))) < 0.03,
        "max_energy_error": float(np.max(np.abs(model_dm2-3.0))) < 0.04,
        "case_relaxation_error": relevant_error < relevant_threshold,
        "initial_condition_linf": initial_linf < 2.0e-6,
        "positive_density": minimum_density >= 0.0,
    }
    if config.case == "heat_flux":
        checks["transverse_heat_flux_symmetry"] = transverse_heat_flux_relative < 0.02
    metrics: dict[str,Any]={
        "case":config.case,"marginal_relative_l2":marginal_error,
        "stress_history_relative_l2":stress_error,"heat_flux_history_relative_l2":heat_error,
        "case_relaxation_error":relevant_error,
        "max_mass_error":float(np.max(np.abs(model_mass-1.0))),
        "max_momentum_norm":float(np.max(np.linalg.norm(model_mean,axis=1))),
        "max_energy_error":float(np.max(np.abs(model_dm2-3.0))),
        "minimum_density":minimum_density,"initial_condition_linf":initial_linf,
        "gate_checks":checks,"gate_passed":bool(all(checks.values())),
    }
    if config.case == "heat_flux":
        metrics.update({
            "heat_flux_active_axis_relative_l2":active_heat_error,
            "max_transverse_heat_flux_relative":transverse_heat_flux_relative,
            "reference_transverse_heat_flux_relative":reference_transverse_heat_flux_relative,
        })

    np.savez_compressed(
        output/"validation.npz",time=times,model_mass=model_mass,model_mean=model_mean,
        model_dm2=model_dm2,model_pij=model_pij,model_q=model_q,reference_pij=ref_pij,
        reference_q=ref_q,selected_time_indices=selected,histogram_centers=centers,
        predicted_marginals=pred_marginals,reference_marginals=ref_marginals,
    )
    with (output/"moments_by_time.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["time","mass","mean_x","mean_y","mean_z","dm2",
                         "p_xx","p_xy","p_xz","p_yy","p_yz","p_zz","q_x","q_y","q_z"])
        for i,t in enumerate(times):
            writer.writerow([t,model_mass[i],*model_mean[i],model_dm2[i],*model_pij[i],*model_q[i]])
    make_validation_plot(config,output,times,centers,selected,pred_marginals,ref_marginals,
                         model_mass,model_mean,model_dm2,model_pij,model_q,ref_pij,ref_q)
    return metrics


def make_validation_plot(
    config:Config,output:Path,times:np.ndarray,centers:np.ndarray,selected:np.ndarray,
    predicted:np.ndarray,reference:np.ndarray,mass:np.ndarray,mean:np.ndarray,dm2:np.ndarray,
    pij:np.ndarray,q:np.ndarray,ref_pij:np.ndarray,ref_q:np.ndarray,
) -> None:
    plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":0.25})
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.5),constrained_layout=True)
    active_axis=0
    for j,index in enumerate(selected):
        row=j*3+active_axis
        axes[0,0].plot(centers,predicted[row],label=fr"PINN $t={times[index]:.2f}$")
        axes[0,0].plot(centers,reference[row],"--",lw=1.0,label=fr"particle $t={times[index]:.2f}$")
    axes[0,0].set(xlabel=r"$c_x$",ylabel=r"marginal $f_x$",title="Distribution relaxation")
    axes[0,0].legend(fontsize=6,ncol=2)
    if config.case=="stress":
        axes[0,1].plot(times,pij[:,0]-pij[:,3],label="PINN")
        axes[0,1].plot(times,ref_pij[:,0]-ref_pij[:,3],"--",label="particle")
        axes[0,1].set(ylabel=r"$P_{xx}-P_{yy}$",title="Stress relaxation")
    elif config.case=="heat_flux":
        axes[0,1].plot(times,q[:,0],label="PINN")
        axes[0,1].plot(times,ref_q[:,0],"--",label="particle")
        axes[0,1].set(ylabel=r"$Q_x=2q_x/\rho$",title="Heat-flux relaxation")
    else:
        dev=np.sqrt((pij[:,0]-1)**2+(pij[:,3]-1)**2+(pij[:,5]-1)**2+2*(pij[:,1]**2+pij[:,2]**2+pij[:,4]**2))
        axes[0,1].semilogy(times,np.maximum(dev,1e-12),label="stress norm")
        axes[0,1].semilogy(times,np.maximum(np.linalg.norm(q,axis=1),1e-12),label="heat flux norm")
        axes[0,1].set(title="Equilibrium invariance")
    axes[0,1].set_xlabel(r"$t/\tau$"); axes[0,1].legend()
    axes[1,0].semilogy(times,np.maximum(np.abs(mass-1),1e-12),label="mass")
    axes[1,0].semilogy(times,np.maximum(np.linalg.norm(mean,axis=1),1e-12),label="momentum")
    axes[1,0].semilogy(times,np.maximum(np.abs(dm2-3),1e-12),label="energy")
    axes[1,0].set(xlabel=r"$t/\tau$",ylabel="absolute error",title="Conservation audit")
    axes[1,0].legend()
    if config.case=="heat_flux":
        axes[1,1].semilogy(times,np.maximum(np.abs(q[:,1]),1e-12),label=r"PINN $|Q_y|$")
        axes[1,1].semilogy(times,np.maximum(np.abs(q[:,2]),1e-12),label=r"PINN $|Q_z|$")
        axes[1,1].semilogy(times,np.maximum(np.linalg.norm(ref_q[:,1:3],axis=1),1e-12),
                            "--",label="particle transverse noise")
        axes[1,1].set(xlabel=r"$t/\tau$",ylabel="absolute magnitude",
                      title="Transverse-symmetry audit")
        axes[1,1].legend(fontsize=7)
    else:
        axes[1,1].axis("off")
        axes[1,1].text(0.02,0.95,
            "Stage 2: homogeneous cubic FP\n"+f"case: {config.case}\n"
            "PINN: exact IC + positive log-density\nclosure: 9 x 9 C/Gamma + cubic lambda\n"
            "validation: independent particle reference",va="top",family="monospace")
    fig.savefig(output/"stage2_validation.pdf",bbox_inches="tight")
    fig.savefig(output/"stage2_validation.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config=parse_args()
    tf.keras.backend.set_floatx("float32")
    tf.random.set_seed(config.seed); np.random.seed(config.seed)
    output=Path(config.output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    (output/"config.json").write_text(json.dumps(asdict(config),indent=2)+"\n")
    model=DensityModel(config)
    model.log_density(tf.zeros((1,1)),tf.zeros((1,3)))
    if config.resume_weights:
        model.load_weights(config.resume_weights)
        print(f"Loaded portable weights: {config.resume_weights}",flush=True)
    if not config.evaluate_only:
        schedule=tf.keras.optimizers.schedules.ExponentialDecay(
            config.learning_rate,config.lr_decay_steps,config.lr_decay_rate,staircase=True
        )
        optimizer=tf.keras.optimizers.Adam(schedule)
        train_step=make_train_step(model,optimizer,config)
        checkpoints=output/"checkpoints_h5"; checkpoints.mkdir(exist_ok=True)
        history=[]; started=time.perf_counter()
        for epoch in range(1,config.epochs+1):
            result={key:float(value.numpy()) for key,value in train_step().items()}
            result["epoch"]=epoch; history.append(result)
            if epoch==1 or epoch%config.print_every==0:
                print("stage2 " + " ".join(
                    [f"case={config.case}",f"epoch={epoch:6d}",f"total={result['total']:.3e}",
                     f"pde={result['pde']:.3e}",f"mass={result['mass']:.3e}",
                     f"mom={result['momentum']:.3e}",f"energy={result['energy']:.3e}",
                     f"qweak={result['heat_flux']:.3e}",
                     f"grad={result['grad_norm']:.3e}",f"elapsed={time.perf_counter()-started:.1f}s"]),flush=True)
            if epoch%config.checkpoint_every==0 or epoch==config.epochs:
                model.save_weights(checkpoints/f"epoch-{epoch:06d}.weights.h5")
        with (output/"loss_history.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    final_weights=output/"stage2_final.weights.h5"; model.save_weights(final_weights)
    audit_points=sample_proposal(config.case,2048,np.random.default_rng(config.seed+1))
    before=_predict_log_density(model,0.731,audit_points)
    reloaded=DensityModel(config); reloaded.log_density(tf.zeros((1,1)),tf.zeros((1,3)))
    reloaded.load_weights(final_weights)
    reload_linf=float(np.max(np.abs(before-_predict_log_density(reloaded,0.731,audit_points))))
    metrics=evaluate(reloaded,config,output); metrics["portable_reload_linf"]=reload_linf
    metrics["portable_reload_passed"]=reload_linf<1.0e-7
    metrics["gate_passed"]=bool(metrics["gate_passed"] and metrics["portable_reload_passed"])
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")
    print("FINAL_METRICS "+json.dumps(metrics,sort_keys=True),flush=True)
    print(f"Artifacts: {output}",flush=True)
    if config.strict_gate and not metrics["gate_passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()

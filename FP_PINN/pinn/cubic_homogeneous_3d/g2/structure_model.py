"""Structure-preserving density ansatz for the axisymmetric stress G2 stage.

G2 carries the deterministic quadrature and exact exponential invariant tilt
validated in G1 to an independent second-moment benchmark.  The initial state
is an axisymmetric anisotropic Gaussian with variances (1.6, 0.7, 0.7), so its
trace is exactly three and its deviatoric stress amplitude is 0.9.

Ansatz (all symbols as in train_stage2.py):

    log f~(t, c) = log[(1-a) f0 + a M] + t cap tanh(raw/cap) + s(t) phi2(c)
    log f (t, c) = log f~(t, c) + beta(t) . psi(c),    psi = (1, cx, |c|^2 - 3)

* phi2(c) = cx^2 - (cy^2+cz^2)/2 is the traceless second Hermite stress mode.
  Its bounded amplitude s(t) vanishes at t=0, preserving the exact initial
  condition.  The qualification loss never sees the analytic exp(-2 nu t)
  stress history; that law is reserved for held-out evaluation.
* beta(t) in R^3 is *not* a trainable parameter.  For every time slice it is
  the unique solution of the convex moment-matching problem

        int f~ exp(beta.psi) psi dc = (1, 0, 0),

  computed with a few Newton steps on the deterministic (cx, rho) quadrature.
  Mass, momentum and energy of f are therefore exact (to quadrature and Newton
  tolerance, ~1e-12) at every time, independently of the network.  This is the
  classical exponential tilt: the minimum-KL correction of f~ that restores the
  three collision invariants.  The finite-grid beta converges to the
  continuum tilt as the deterministic quadrature is refined.  For a generic
  untrained network the train/fine grids agree to sub-1e-3 relative accuracy;
  the effect of the remaining quadrature difference on Qx is measured again
  for every trained checkpoint and is subject to its own frozen gate.
* beta depends on t through the whole slice, so its time derivative is
  obtained by implicit differentiation of the constraint (``tilt_time_rate``)
  instead of by per-node automatic differentiation, which would be wrong.

Velocity derivatives of the tilt are analytic: grad(beta.psi) = (beta_1, 0, 0)
+ 2 beta_2 c and lap(beta.psi) = 6 beta_2.

Nothing here samples anything: all integrals are deterministic quadratures
(see axisym_quadrature.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for path in (str(PARENT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_stage2 import (  # noqa: E402
    Config as BaseConfig,
    DensityModel,
    _matrix_from_vector,
    closure_tf,
    tf_equilibrium_logpdf,
)
from g1.axisym_quadrature import AxisymQuadrature  # noqa: E402

INVARIANT_TARGET = (1.0, 0.0, 0.0)   # mass, x-momentum, (energy - 3)


@dataclass
class G2Config:
    """Options of the structure-preserving ansatz (stored in config.json)."""

    use_stress_mode: bool = True
    stress_mode_cap: float = 0.10
    stress_mode_width: int = 16
    tilt_newton_steps: int = 4
    tilt_beta_cap: float = 0.5
    tmax: float = 1.0
    nu: float = 1.0


# --------------------------------------------------------------------------
# Deterministic quadrature as TensorFlow constants
# --------------------------------------------------------------------------
@dataclass
class QuadratureTensors:
    """One (cx, rho) quadrature as the tensors needed by the residual."""

    nodes32: tf.Tensor     # [nv, 3] float32 (cx, rho, 0), fed to the network
    nodes64: tf.Tensor     # [nv, 3] float64, used for every integral
    weights64: tf.Tensor   # [nv] float64, includes the 2 pi rho Jacobian
    psi64: tf.Tensor       # [nv, 3] float64, (1, cx, |c|^2 - 3)
    phi64: tf.Tensor       # [nv] float64, cx^2-rho^2/2 (deviatoric stress basis)
    size: int = field(default=0)

    @property
    def cx64(self) -> tf.Tensor:
        return self.nodes64[:, 0]

    @property
    def rho64(self) -> tf.Tensor:
        return self.nodes64[:, 1]


def invariant_features_tf(c: tf.Tensor) -> tf.Tensor:
    """psi(c) = (1, cx, |c|^2 - 3) as float64, shape [..., 3]."""
    c = tf.cast(c, tf.float64)
    r2 = tf.reduce_sum(tf.square(c), axis=-1)
    return tf.stack([tf.ones_like(r2), c[..., 0], r2 - 3.0], axis=-1)


def stress_mode_tf(c: tf.Tensor) -> tf.Tensor:
    """Traceless Hermite mode cx^2-(cy^2+cz^2)/2, shape [n, 1]."""
    return tf.square(c[:, 0:1]) - 0.5 * tf.reduce_sum(tf.square(c[:, 1:3]), axis=1, keepdims=True)


def axisym_stress_initial_logpdf_tf(c: tf.Tensor) -> tf.Tensor:
    """Unit-mass Gaussian with covariance diag(1.6, 0.7, 0.7)."""
    variances = tf.constant([1.6, 0.7, 0.7], dtype=c.dtype)
    return tf.reduce_sum(
        -0.5 * (
            tf.math.log(tf.cast(2.0 * np.pi, c.dtype) * variances)
            + tf.square(c) / variances
        ),
        axis=-1,
        keepdims=True,
    )


def quadrature_tensors(quad: AxisymQuadrature) -> QuadratureTensors:
    nodes64 = tf.constant(quad.nodes, tf.float64)
    r2 = tf.reduce_sum(tf.square(nodes64), axis=1)
    return QuadratureTensors(
        nodes32=tf.constant(quad.nodes, tf.float32),
        nodes64=nodes64,
        weights64=tf.constant(quad.weights, tf.float64),
        psi64=invariant_features_tf(nodes64),
        phi64=tf.square(nodes64[:, 0]) - 0.5 * tf.square(nodes64[:, 1]),
        size=quad.size,
    )


# --------------------------------------------------------------------------
# Axisymmetric moments
# --------------------------------------------------------------------------
def axisym_moment_tensors_tf(cx: tf.Tensor, rho: tf.Tensor, wf: tf.Tensor) -> dict[str, tf.Tensor]:
    """Full 3-D central moments of an axisymmetric density from (cx, rho) nodes.

    cx, rho: [nv] float64 node coordinates; wf: [nt, nv] float64, quadrature
    weight (including the 2 pi rho Jacobian) times density.

    The generic train_stage2.moment_tensors must NOT be applied to the nodes
    (cx, rho, 0): it subtracts a mean in every coordinate, and the "mean" of
    rho >= 0 is not zero, so it returns Var(rho) instead of <rho^2>.  Here the
    transverse moments are reconstructed from the axisymmetric identities
    <cy^2> = <cz^2> = <rho^2>/2, <cx cy^2> = <cx rho^2>/2, etc.  The dictionary
    layout is that of moment_tensors so that closure_tf can be reused.
    """
    cx = tf.cast(cx, tf.float64)
    rho = tf.cast(rho, tf.float64)
    wf = tf.cast(wf, tf.float64)
    mass = tf.reduce_sum(wf, axis=1)
    normalized = wf / tf.maximum(mass, 1.0e-300)[:, None]
    mean_x = tf.reduce_sum(normalized * cx[None, :], axis=1)
    vx = cx[None, :] - mean_x[:, None]
    rho2 = tf.square(rho)[None, :]
    r2 = tf.square(vx) + rho2

    def avg(values: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(normalized * values, axis=1)

    zero = tf.zeros_like(mass)
    pxx = avg(tf.square(vx))
    half_transverse = 0.5 * avg(rho2)
    q_x = avg(vx * r2)
    x3 = avg(vx * vx * vx)
    half_x_rho2 = 0.5 * avg(vx * rho2)
    x2r2 = avg(tf.square(vx) * r2)
    half_rho2r2 = 0.5 * avg(rho2 * r2)
    m5x = avg(vx * r2 * r2)
    return {
        "mass": mass,
        "mean": tf.stack([mean_x, zero, zero], axis=1),
        "pij": tf.stack([pxx, zero, zero, half_transverse, zero, half_transverse], axis=1),
        "q": tf.stack([q_x, zero, zero], axis=1),
        "m3": tf.stack([x3, zero, zero, half_x_rho2, zero, half_x_rho2, zero, zero, zero, zero], axis=1),
        "m4": tf.stack([x2r2, zero, zero, half_rho2r2, zero, half_rho2r2], axis=1),
        "m5": tf.stack([m5x, zero, zero], axis=1),
        "dm2": avg(r2),
        "dm4": avg(r2 * r2),
    }


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class StressHead(tf.keras.layers.Layer):
    """s(t) = t*cap*tanh(MLP(t/tmax)/cap), with s(0)=0."""

    def __init__(self, width: int, cap: float, tmax: float, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.cap = float(cap)
        self.tmax = float(tmax)
        self.dense_1 = tf.keras.layers.Dense(
            width, activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed + 101),
        )
        self.dense_2 = tf.keras.layers.Dense(
            width, activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed + 102),
        )
        self.out = tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros")

    def call(self, t: tf.Tensor) -> tf.Tensor:
        t = tf.cast(t, tf.float32)
        raw = self.out(self.dense_2(self.dense_1(t / self.tmax)))
        return t * self.cap * tf.tanh(raw / self.cap)


class AxisymStressDensityModel(DensityModel):
    """G0-compatible network with an exact axisymmetric stress initial state."""

    def log_density(self, t: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        t = tf.cast(t, tf.float32)
        c = tf.cast(c, tf.float32)
        r2 = tf.reduce_sum(tf.square(c), axis=1, keepdims=True)
        velocity_features = tf.concat(
            [c[:, 0:1] / 3.0, tf.zeros_like(c[:, 1:3])], axis=1,
        )
        features = tf.concat([t / self.config.tmax, velocity_features, r2 / 9.0], axis=1)
        raw = self(features)
        correction = t * self.config.correction_cap * tf.tanh(raw / self.config.correction_cap)
        log_f0 = axisym_stress_initial_logpdf_tf(c)
        log_m = tf_equilibrium_logpdf(c)
        alpha = tf.clip_by_value(1.0 - tf.exp(-self.config.bridge_rate * t), 0.0, 1.0 - 1.0e-7)
        safe_alpha = tf.maximum(alpha, tf.constant(1.0e-30, dtype=alpha.dtype))
        bridge = tf.reduce_logsumexp(
            tf.concat([
                log_f0 + tf.math.log1p(-alpha),
                log_m + tf.math.log(safe_alpha),
            ], axis=1),
            axis=1,
            keepdims=True,
        )
        bridge = tf.where(alpha > 0.0, bridge, log_f0)
        return bridge + correction


class StructuredDensityModel(tf.keras.Model):
    """Axisymmetric stress base + stress mode + exact invariant tilt."""

    def __init__(self, base_config: BaseConfig, g2: G2Config):
        super().__init__()
        self.base_config = base_config
        self.g1 = g2  # assemble_slices uses the common tilt options through this name
        self.base = AxisymStressDensityModel(base_config)
        self.head = (
            StressHead(g2.stress_mode_width, g2.stress_mode_cap, g2.tmax, base_config.seed)
            if g2.use_stress_mode else None
        )

    # -- pointwise part -----------------------------------------------------
    def raw_log_density(self, t: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        """log f~ (before the tilt); a pure pointwise function of (t, c)."""
        t = tf.cast(t, tf.float32)
        c = tf.cast(c, tf.float32)
        log_f = self.base.log_density(t, c)
        if self.head is not None:
            log_f = log_f + self.head(t) * stress_mode_tf(c)
        return log_f

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Keras entry point: inputs = [n, 4] rows (t, cx, cy, cz) -> log f~ [n, 1]."""
        inputs = tf.cast(inputs, tf.float32)
        return self.raw_log_density(inputs[:, 0:1], inputs[:, 1:4])

    def build_model(self) -> None:
        """Create every weight (base network, head) so that save/load_weights work."""
        self(tf.zeros([2, 4]))

    # -- tilt ---------------------------------------------------------------
    def solve_tilt(
        self, log_f_raw: tf.Tensor, psi: tf.Tensor, weights: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Newton solve of  sum_v w_v f~_v exp(beta.psi_v) psi_v = target.

        log_f_raw: [nt, nv] float64, psi: [nv, 3] float64, weights: [nv] float64.
        Returns beta [nt, 3], the Jacobian J [nt, 3, 3] of the moment map at
        beta and the tilted log density [nt, nv] (float64).  The loop is
        unrolled, so parameter gradients flow through the solve.
        """
        target = tf.constant(INVARIANT_TARGET, tf.float64)
        nt = tf.shape(log_f_raw)[0]
        beta = tf.zeros([nt, 3], tf.float64)
        cap = tf.constant(self.g1.tilt_beta_cap, tf.float64)
        for _ in range(self.g1.tilt_newton_steps):
            log_f = log_f_raw + tf.matmul(beta, psi, transpose_b=True)
            wf = weights[None, :] * tf.exp(log_f)
            moments = tf.matmul(wf, psi)                                  # [nt, 3]
            jac = tf.einsum("tv,vi,vj->tij", wf, psi, psi)                 # [nt, 3, 3]
            step = tf.linalg.solve(jac, (moments - target)[:, :, None])[:, :, 0]
            beta = tf.clip_by_value(beta - step, -cap, cap)
        log_f = log_f_raw + tf.matmul(beta, psi, transpose_b=True)
        wf = weights[None, :] * tf.exp(log_f)
        jac = tf.einsum("tv,vi,vj->tij", wf, psi, psi)
        return beta, jac, log_f

    @staticmethod
    def tilt_time_rate(
        jac: tf.Tensor, wf: tf.Tensor, psi: tf.Tensor, raw_time_derivative: tf.Tensor,
    ) -> tf.Tensor:
        """d beta / dt from implicit differentiation of the moment constraint.

        sum_v w_v f_v psi_v (d_t log f~_v + beta'.psi_v) = 0
            =>  J beta' = - sum_v w_v f_v psi_v d_t log f~_v.
        wf: [nt, nv] (w f, tilted), raw_time_derivative: [nt, nv]; returns [nt, 3].
        """
        rhs = -tf.matmul(wf * raw_time_derivative, psi)                 # [nt, 3]
        return tf.linalg.solve(jac, rhs[:, :, None])[:, :, 0]


def pointwise_derivatives(
    model: StructuredDensityModel, flat_t: tf.Tensor, flat_c: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """log f~, d_t log f~, grad_c log f~, lap_c log f~ at the nodes (float32).

    Identical tape structure to train_stage2.make_train_step.  Because the
    network is axisymmetric the 3-D Laplacian evaluated at (cx, rho, 0) equals
    the axisymmetric Laplacian d_xx + d_rr + d_r / rho.
    """
    with tf.GradientTape(persistent=True) as second_tape:
        second_tape.watch([flat_t, flat_c])
        with tf.GradientTape(persistent=True) as first_tape:
            first_tape.watch([flat_t, flat_c])
            log_f = model.raw_log_density(flat_t, flat_c)
        h_t = first_tape.gradient(log_f, flat_t)
        grad_h = first_tape.gradient(log_f, flat_c)
        del first_tape
    hessian = second_tape.batch_jacobian(grad_h, flat_c, experimental_use_pfor=True)
    del second_tape
    lap_h = tf.linalg.trace(hessian)[:, None]
    return log_f, h_t, grad_h, lap_h


@dataclass
class SliceState:
    """Everything the residual needs on one batch of time slices."""

    log_f: tf.Tensor          # [nt, nv] float64 tilted log density
    wf: tf.Tensor             # [nt, nv] float64 quadrature weight * f
    beta: tf.Tensor           # [nt, 3]
    beta_rate: tf.Tensor      # [nt, 3]
    moments: dict             # float32 moments of the tilted density
    moments64: dict
    coefficients: tf.Tensor   # [nt, 9] closure vector (float32)
    lam: tf.Tensor            # [nt]
    h_t: tf.Tensor            # [nt, nv, 1] full d_t log f (float32)
    grad_h: tf.Tensor         # [nt, nv, 3]
    lap_h: tf.Tensor          # [nt, nv, 1]
    raw_moments64: dict       # moments of f~ (diagnostic: how much the tilt does)


def assemble_slices(
    model: StructuredDensityModel,
    t_slices: tf.Tensor,       # [nt]
    quad: QuadratureTensors,
) -> SliceState:
    nodes = quad.nodes32
    weights64 = quad.weights64
    psi64 = quad.psi64
    nt = tf.shape(t_slices)[0]
    nv = tf.shape(nodes)[0]
    flat_t = tf.reshape(tf.repeat(tf.cast(t_slices, tf.float32)[:, None], nv, axis=1), [-1, 1])
    flat_c = tf.reshape(tf.tile(nodes[None, :, :], [nt, 1, 1]), [-1, 3])
    log_f_raw32, h_t_raw, grad_h_raw, lap_h_raw = pointwise_derivatives(model, flat_t, flat_c)
    tf.debugging.assert_all_finite(lap_h_raw, "Non-finite velocity-space Laplacian of log density")
    log_f_raw = tf.cast(tf.reshape(log_f_raw32, [nt, nv]), tf.float64)

    beta, jac, log_f = model.solve_tilt(log_f_raw, psi64, weights64)
    wf = weights64[None, :] * tf.exp(log_f)
    raw_wf = weights64[None, :] * tf.exp(log_f_raw)
    h_t_raw64 = tf.cast(tf.reshape(h_t_raw, [nt, nv]), tf.float64)
    beta_rate = model.tilt_time_rate(jac, wf, psi64, h_t_raw64)

    # Moments and closure of the tilted density in float64.  The closure is
    # the validated train_stage2.closure_tf; only the moment assembly is
    # replaced by the axisymmetric version (see axisym_moment_tensors_tf).
    moments64 = axisym_moment_tensors_tf(quad.cx64, quad.rho64, wf)
    coefficients64, lam64 = closure_tf(moments64, model.base_config)
    raw_moments64 = axisym_moment_tensors_tf(quad.cx64, quad.rho64, raw_wf)

    # Full derivatives of log f = log f~ + beta.psi (tilt derivatives analytic).
    psi32 = tf.cast(psi64, tf.float32)                                   # [nv, 3]
    beta32 = tf.cast(beta, tf.float32)
    beta_rate32 = tf.cast(beta_rate, tf.float32)
    h_t = tf.reshape(h_t_raw, [nt, nv, 1]) + tf.matmul(beta_rate32, psi32, transpose_b=True)[:, :, None]
    c32 = tf.tile(nodes[None, :, :], [nt, 1, 1])
    grad_tilt = (
        tf.concat([beta32[:, 1:2], tf.zeros_like(beta32[:, 1:2]), tf.zeros_like(beta32[:, 1:2])], axis=1)[:, None, :]
        + 2.0 * beta32[:, 2:3, None] * c32
    )
    grad_h = tf.reshape(grad_h_raw, [nt, nv, 3]) + grad_tilt
    lap_h = tf.reshape(lap_h_raw, [nt, nv, 1]) + 6.0 * beta32[:, 2:3, None]

    return SliceState(
        log_f=log_f, wf=wf, beta=beta, beta_rate=beta_rate,
        moments={k: tf.cast(v, tf.float32) for k, v in moments64.items()},
        moments64=moments64,
        coefficients=tf.cast(coefficients64, tf.float32), lam=tf.cast(lam64, tf.float32),
        h_t=h_t, grad_h=grad_h, lap_h=lap_h, raw_moments64=raw_moments64,
    )


def log_residual(
    state: SliceState, quad: QuadratureTensors, nu: float, stop_gradient_closure: bool = False,
) -> tf.Tensor:
    """Residual of the cubic FP equation divided by f, shape [nt, nv].

    R = d_t log f + div a + a . grad log f - D (lap log f + |grad log f|^2),
    exactly the operator of train_stage2.make_train_step.
    """
    nodes = quad.nodes32
    nt = tf.shape(state.h_t)[0]
    c32 = tf.tile(nodes[None, :, :], [nt, 1, 1])
    coefficients = state.coefficients
    lam = state.lam
    if stop_gradient_closure:
        coefficients = tf.stop_gradient(coefficients)
        lam = tf.stop_gradient(lam)
    matrix = _matrix_from_vector(coefficients)
    gamma = coefficients[:, 6:9]
    moments = state.moments
    peculiar = c32 - moments["mean"][:, None, :]
    r2 = tf.reduce_sum(tf.square(peculiar), axis=2)
    linear = tf.einsum("tij,tvj->tvi", matrix, peculiar)
    nonlinear = (
        linear + gamma[:, None, :] * (r2 - moments["dm2"][:, None])[:, :, None]
        + lam[:, None, None] * (peculiar * r2[:, :, None] - moments["q"][:, None, :])
    )
    drift = -nu * peculiar + nonlinear
    divergence = (
        -3.0 * nu + tf.linalg.trace(matrix)[:, None]
        + 2.0 * tf.reduce_sum(gamma[:, None, :] * peculiar, axis=2)
        + 5.0 * lam[:, None] * r2
    )
    diffusion = (nu * moments["dm2"] / 3.0)[:, None]
    grad_h = state.grad_h
    lap_h = state.lap_h[:, :, 0]
    return (
        state.h_t[:, :, 0] + divergence
        + tf.reduce_sum(drift * grad_h, axis=2)
        - diffusion * (lap_h + tf.reduce_sum(tf.square(grad_h), axis=2))
    )


def weighted_residual_loss(state: SliceState, residual: tf.Tensor) -> tf.Tensor:
    """f-weighted mean square of the log residual: sum_tv w f R^2 / sum_tv w f.

    sum_v w f R^2 = int (d_t f - L f)^2 / f dc, the natural (Fisher) norm of
    the density residual; the weight itself carries no parameter gradient so
    that the optimiser cannot lower the loss by moving mass away from where
    the residual is large.
    """
    wf = tf.cast(tf.stop_gradient(state.wf), tf.float32)
    return tf.reduce_sum(wf * tf.square(residual)) / tf.reduce_sum(wf)


def stress_rate_residual(state: SliceState, quad: QuadratureTensors, nu: float) -> tuple[tf.Tensor, tf.Tensor]:
    """Weak identity d(pxx-pperp)/dt + 2 nu (pxx-pperp)=0.

    It is disabled in qualification training and exists only for a labelled
    ablation, so the analytic stress history cannot leak into the main fit.
    """
    phi = quad.phi64
    delta = tf.reduce_sum(state.wf * phi[None, :], axis=1)
    ddelta = tf.reduce_sum(state.wf * phi[None, :] * tf.cast(state.h_t[:, :, 0], tf.float64), axis=1)
    return delta, ddelta + 2.0 * nu * delta


__all__ = [
    "G2Config",
    "StressHead",
    "INVARIANT_TARGET",
    "QuadratureTensors",
    "SliceState",
    "StructuredDensityModel",
    "assemble_slices",
    "axisym_moment_tensors_tf",
    "axisym_stress_initial_logpdf_tf",
    "stress_mode_tf",
    "stress_rate_residual",
    "invariant_features_tf",
    "log_residual",
    "pointwise_derivatives",
    "quadrature_tensors",
    "weighted_residual_loss",
]

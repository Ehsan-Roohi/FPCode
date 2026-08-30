"""Structural tests for the heat-flux G1 stage.

Run from FP_PINN/pinn/cubic_homogeneous_3d:

    python -m unittest discover -s g1/tests -t . -v

Every test is deterministic and finishes in well under a minute on a CPU.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
STAGE_DIR = HERE.parent.parent
for path in (str(STAGE_DIR), str(STAGE_DIR / "g1")):
    if path not in sys.path:
        sys.path.insert(0, path)

from cubic_operator import moments_from_samples  # noqa: E402
from train_stage2 import Config as BaseConfig, tf_initial_logpdf  # noqa: E402
from axisym_quadrature import build_quadrature, axisymmetric_moments, heat_flux_mode, invariant_features  # noqa: E402
from structure_model import (  # noqa: E402
    G1Config,
    StructuredDensityModel,
    assemble_slices,
    axisym_moment_tensors_tf,
    quadrature_tensors,
)


def exact_initial_density(cx: np.ndarray, rho: np.ndarray) -> np.ndarray:
    gx = (1.0 / 3.0) * np.exp(-((cx - 1.0) ** 2)) / np.sqrt(np.pi) + (2.0 / 3.0) * np.exp(-((cx + 0.5) ** 2)) / np.sqrt(np.pi)
    return gx * np.exp(-0.5 * rho**2) / (2.0 * np.pi)


def small_model(seed: int = 3) -> StructuredDensityModel:
    base = BaseConfig(case="heat_flux", output_dir="", reference="", width=16, depth=2, seed=seed)
    model = StructuredDensityModel(base, G1Config(heat_flux_mode_width=8))
    model.build_model()
    # Give the network and the head non-trivial weights so that the tests
    # exercise a generic (not identically zero) correction.
    rng = np.random.default_rng(seed)
    for variable in model.trainable_variables:
        variable.assign(rng.normal(scale=0.3, size=variable.shape).astype(np.float32))
    return model


class QuadratureTests(unittest.TestCase):
    def test_initial_state_moments_are_exact(self):
        quad = build_quadrature(8.0, 129, 8.0, 32)
        f = exact_initial_density(quad.nodes[:, 0], quad.nodes[:, 1])
        m = axisymmetric_moments(quad, f)
        self.assertAlmostEqual(m["mass"], 1.0, places=12)
        self.assertAlmostEqual(m["mean_x"], 0.0, places=12)
        self.assertAlmostEqual(m["dm2"], 3.0, places=11)
        self.assertAlmostEqual(m["qx"], 0.25, places=12)
        self.assertAlmostEqual(m["pyy"], 1.0, places=11)
        self.assertAlmostEqual(m["dm4"], 14.625, places=9)

    def test_shifted_panels_have_same_accuracy(self):
        for shift in (0.0, 0.3, 0.77):
            quad = build_quadrature(8.0, 129, 8.0, 32, cx_shift=shift)
            f = exact_initial_density(quad.nodes[:, 0], quad.nodes[:, 1])
            self.assertAlmostEqual(axisymmetric_moments(quad, f)["qx"], 0.25, places=11)

    def test_axisymmetric_moment_tensors_match_three_dimensional_moments(self):
        quad = build_quadrature(8.0, 129, 8.0, 32)
        f = exact_initial_density(quad.nodes[:, 0], quad.nodes[:, 1])
        wf = tf.constant((quad.weights * f)[None, :], tf.float64)
        m = {k: v.numpy()[0] for k, v in axisym_moment_tensors_tf(
            tf.constant(quad.nodes[:, 0]), tf.constant(quad.nodes[:, 1]), wf).items()}
        rng = np.random.default_rng(0)
        n = 400_000
        a = rng.uniform(size=n) < 1.0 / 3.0
        c = np.stack([np.where(a, 1.0, -0.5) + np.sqrt(0.5) * rng.normal(size=n),
                      rng.normal(size=n), rng.normal(size=n)], axis=1)
        mc = moments_from_samples(c)
        np.testing.assert_allclose(m["pij"], mc.pij, atol=2.0e-2)
        np.testing.assert_allclose(m["q"], mc.q, atol=2.0e-2)
        np.testing.assert_allclose(m["m3"], mc.m3, atol=2.0e-2)
        np.testing.assert_allclose(m["m4"], mc.m4, atol=1.0e-1)
        np.testing.assert_allclose(m["m5"], mc.m5, atol=1.5e-1)
        # Exact values of the initial state.
        np.testing.assert_allclose(m["pij"], [1, 0, 0, 1, 0, 1], atol=1.0e-10)
        np.testing.assert_allclose(m["q"], [0.25, 0, 0], atol=1.0e-10)
        np.testing.assert_allclose(m["m4"], [4.625, 0, 0, 5, 0, 5], atol=1.0e-9)
        self.assertAlmostEqual(m["dm4"], 14.625, places=9)

    def test_heat_flux_mode_is_orthogonal_to_invariants(self):
        quad = build_quadrature(9.0, 257, 9.0, 64)
        # Pure-numpy Maxwellian: train_stage2's tf_equilibrium_logpdf rounds its
        # normalisation constant through float32 (tf.cast of a Python float),
        # which is harmless for training but spoils a 1e-9 test.
        maxwellian = np.exp(-0.5 * np.sum(quad.nodes**2, axis=1)) / (2.0 * np.pi) ** 1.5
        phi3 = heat_flux_mode(quad.nodes)
        psi = invariant_features(quad.nodes)
        for i in range(3):
            self.assertAlmostEqual(float(np.sum(quad.weights * maxwellian * phi3 * psi[:, i])), 0.0, places=10)
        r2 = np.sum(quad.nodes**2, axis=1)
        self.assertAlmostEqual(float(np.sum(quad.weights * maxwellian * phi3 * quad.nodes[:, 0] * r2)), 10.0, places=9)


class ModelStructureTests(unittest.TestCase):
    def test_hard_initial_condition(self):
        model = small_model()
        quad = build_quadrature(8.0, 129, 8.0, 32)
        nodes = tf.constant(quad.nodes, tf.float32)
        log_raw = model.raw_log_density(tf.zeros([quad.size, 1]), nodes).numpy().ravel()
        log_f0 = tf_initial_logpdf("heat_flux", nodes).numpy().ravel()
        np.testing.assert_allclose(log_raw, log_f0, atol=1.0e-6)
        self.assertEqual(float(tf.reduce_max(tf.abs(model.head(tf.zeros([4, 1]))))), 0.0)

    def test_exact_axisymmetry(self):
        model = small_model()
        rng = np.random.default_rng(1)
        cx, rho, angle = rng.normal(size=64), np.abs(rng.normal(size=64)) * 1.5, rng.uniform(0, 2 * np.pi, size=64)
        a = np.stack([cx, rho, np.zeros(64)], axis=1)
        b = np.stack([cx, rho * np.cos(angle), rho * np.sin(angle)], axis=1)
        t = tf.fill([64, 1], 0.61)
        la = model.raw_log_density(t, tf.constant(a, tf.float32)).numpy()
        lb = model.raw_log_density(t, tf.constant(b, tf.float32)).numpy()
        np.testing.assert_allclose(la, lb, atol=2.0e-5)

    def test_tilt_restores_invariants_exactly(self):
        model = small_model()
        quad = quadrature_tensors(build_quadrature(8.0, 129, 8.0, 32))
        times = tf.constant([0.05, 0.4, 0.9], tf.float32)
        state = assemble_slices(model, times, quad)
        tilted = state.moments64
        np.testing.assert_allclose(tilted["mass"].numpy(), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(tilted["mean"].numpy()[:, 0], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(tilted["dm2"].numpy(), 3.0, atol=1.0e-11)
        # The untrained random network does NOT conserve anything by itself.
        raw = state.raw_moments64
        self.assertGreater(float(np.max(np.abs(raw["mass"].numpy() - 1.0))), 1.0e-4)

    def test_tilt_is_quadrature_converged(self):
        model = small_model()
        times = tf.constant([0.3, 0.8], tf.float32)
        betas = []
        for (n_cx, n_rho, width) in ((129, 32, 8.0), (257, 64, 9.0)):
            state = assemble_slices(model, times, quadrature_tensors(build_quadrature(width, n_cx, width, n_rho)))
            betas.append(state.beta.numpy())
        # beta is the solution of a quadrature approximation to the continuum
        # moment equations, so two finite grids need not give bitwise-identical
        # values.  The deliberately rough random network used here gives a
        # measured max relative difference of about 6.5e-4 on Unity.  Require
        # convergence at the 1e-3 level; the evaluator separately freezes the
        # physically relevant train/fine Qx discrepancy at <= 0.5 percentage
        # points for every checkpoint.
        np.testing.assert_allclose(betas[0], betas[1], rtol=1.0e-3, atol=2.0e-5)

    def test_implicit_time_derivative_of_tilt(self):
        model = small_model()
        quad = quadrature_tensors(build_quadrature(8.0, 129, 8.0, 32))
        # Different GPU matmul kernels round the float32 network output
        # differently.  A small finite difference of beta therefore amplifies
        # harmless round-off by 1/eps and made this regression test depend on
        # the allocated GPU.  Differentiate the independently unrolled Newton
        # solve instead.  This still tests the production implicit formula,
        # but uses an analytic/autodiff reference rather than a subtractive
        # float32 finite difference.
        time = tf.Variable([0.5], dtype=tf.float32)
        with tf.GradientTape() as tape:
            flat_t = tf.repeat(time[:, None], quad.size, axis=0)
            log_f_raw = tf.cast(
                tf.reshape(model.raw_log_density(flat_t, quad.nodes32), [1, quad.size]),
                tf.float64,
            )
            beta, _, _ = model.solve_tilt(log_f_raw, quad.psi64, quad.weights64)
        autodiff = tape.jacobian(beta, time).numpy()[0, :, 0]
        state = assemble_slices(model, tf.constant([0.5], tf.float32), quad)
        implicit = state.beta_rate.numpy()[0]
        np.testing.assert_allclose(implicit, autodiff, rtol=2.0e-4, atol=2.0e-5)

    def test_autodiff_laplacian_equals_axisymmetric_laplacian(self):
        model = small_model()
        quad = quadrature_tensors(build_quadrature(8.0, 129, 8.0, 32))
        state = assemble_slices(model, tf.constant([0.45], tf.float32), quad)
        # Independently form the cylindrical axisymmetric Laplacian of the raw
        # log density at a few interior nodes.  Use nested autodiff rather than
        # a second finite difference of float32 network values: the latter is
        # dominated by GPU-dependent cancellation after division by h**2.
        # This route differentiates with respect to the independent (cx, rho)
        # coordinates, whereas pointwise_derivatives traces the full 3-D
        # Cartesian Hessian.  Their equality is the identity under test.
        nodes = quad.nodes64.numpy()
        grad_h = state.grad_h.numpy()[0]
        lap_h = state.lap_h.numpy()[0, :, 0]
        beta = state.beta.numpy()[0]
        idx = np.arange(0, nodes.shape[0], 397)[1:]
        cx = tf.Variable(nodes[idx, 0].astype(np.float32))
        rho = tf.Variable(nodes[idx, 1].astype(np.float32))
        with tf.GradientTape(persistent=True) as outer:
            outer.watch([cx, rho])
            with tf.GradientTape(persistent=True) as inner:
                inner.watch([cx, rho])
                c = tf.stack([cx, rho, tf.zeros_like(rho)], axis=1)
                raw = model.raw_log_density(tf.fill([len(idx), 1], 0.45), c)[:, 0]
            hx = inner.gradient(raw, cx)
            hr = inner.gradient(raw, rho)
            del inner
        hxx = outer.gradient(hx, cx)
        hrr = outer.gradient(hr, rho)
        del outer

        cx_np = cx.numpy().astype(np.float64)
        rho_np = rho.numpy().astype(np.float64)
        hx_np = hx.numpy().astype(np.float64)
        hr_np = hr.numpy().astype(np.float64)
        lap_axisym = (
            hxx.numpy().astype(np.float64)
            + hrr.numpy().astype(np.float64)
            + hr_np / rho_np
            + 6.0 * beta[2]
        )
        grad_axisym_x = hx_np + beta[1] + 2.0 * beta[2] * cx_np
        grad_axisym_r = hr_np + 2.0 * beta[2] * rho_np
        np.testing.assert_allclose(lap_h[idx], lap_axisym, rtol=2.0e-4, atol=2.0e-4)
        np.testing.assert_allclose(grad_h[idx, 0], grad_axisym_x, rtol=1.0e-4, atol=1.0e-4)
        np.testing.assert_allclose(grad_h[idx, 1], grad_axisym_r, rtol=1.0e-4, atol=1.0e-4)
        np.testing.assert_allclose(grad_h[idx, 2], 0.0, atol=1.0e-6)

    def test_portable_reload(self):
        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.weights.h5"
            model.save_weights(path)
            clone = StructuredDensityModel(model.base_config, model.g1)
            clone.build_model()
            clone.load_weights(path)
        c = tf.constant(np.random.default_rng(5).normal(size=(32, 3)), tf.float32)
        t = tf.fill([32, 1], 0.7)
        np.testing.assert_array_equal(model.raw_log_density(t, c).numpy(), clone.raw_log_density(t, c).numpy())

    def test_g0_weights_load_into_base(self):
        """A G0 DensityModel weight file must load into model.base unchanged."""
        base = BaseConfig(case="heat_flux", output_dir="", reference="", width=16, depth=2, seed=7)
        from train_stage2 import DensityModel
        g0 = DensityModel(base)
        g0.log_density(tf.zeros((1, 1)), tf.zeros((1, 3)))
        rng = np.random.default_rng(7)
        for variable in g0.trainable_variables:
            variable.assign(rng.normal(scale=0.3, size=variable.shape).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g0.weights.h5"
            g0.save_weights(path)
            model = StructuredDensityModel(base, G1Config(use_heat_flux_mode=False))
            model.build_model()
            model.base.load_weights(path)
        c = tf.constant(np.random.default_rng(8).normal(size=(16, 3)), tf.float32)
        t = tf.fill([16, 1], 0.33)
        np.testing.assert_array_equal(g0.log_density(t, c).numpy(), model.raw_log_density(t, c).numpy())


if __name__ == "__main__":
    unittest.main()

"""Deterministic structural tests for axisymmetric stress G2."""

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
for path in (str(STAGE_DIR), str(STAGE_DIR / "g2")):
    if path not in sys.path:
        sys.path.insert(0, path)

from g1.axisym_quadrature import build_quadrature, axisymmetric_moments, invariant_features  # noqa: E402
from train_stage2 import Config as BaseConfig  # noqa: E402
from structure_model import (  # noqa: E402
    G2Config,
    StructuredDensityModel,
    assemble_slices,
    axisym_moment_tensors_tf,
    axisym_stress_initial_logpdf_tf,
    quadrature_tensors,
    stress_mode_tf,
)


def exact_initial_density(cx: np.ndarray, rho: np.ndarray) -> np.ndarray:
    norm = (2.0 * np.pi) ** (-1.5) / np.sqrt(1.6 * 0.7**2)
    return norm * np.exp(-0.5 * (cx**2 / 1.6 + rho**2 / 0.7))


def small_model(seed: int = 13) -> StructuredDensityModel:
    base = BaseConfig(case="stress", output_dir="", reference="", width=16, depth=2, seed=seed)
    model = StructuredDensityModel(base, G2Config(stress_mode_width=8))
    model.build_model()
    rng = np.random.default_rng(seed)
    for variable in model.trainable_variables:
        variable.assign(rng.normal(scale=0.2, size=variable.shape).astype(np.float32))
    return model


class StressQuadratureTests(unittest.TestCase):
    def test_initial_state_moments(self):
        exact = {
            "mass": 1.0, "mean_x": 0.0, "dm2": 3.0,
            "pxx": 1.6, "pyy": 0.7, "qx": 0.0, "dm4": 16.08,
        }
        grids = {
            "train": build_quadrature(8.0, 129, 8.0, 32),
            "fine": build_quadrature(9.0, 257, 9.0, 64),
        }
        moments = {
            name: axisymmetric_moments(
                quad, exact_initial_density(quad.nodes[:, 0], quad.nodes[:, 1])
            )
            for name, quad in grids.items()
        }

        # These are finite-domain quadratures, not symbolic Gaussian
        # integrals.  Freeze tolerances above the measured truncation error of
        # each production grid and separately require geometric convergence
        # on the held-out fine grid.  The old decimal-places assertions
        # incorrectly demanded ~1e-11 from the train grid, whose deterministic
        # mass truncation error is 2.62e-10 and fourth-moment error is 1.22e-6.
        train_atol = {
            "mass": 5.0e-10, "mean_x": 1.0e-12, "dm2": 3.0e-8,
            "pxx": 3.0e-8, "pyy": 1.0e-12, "qx": 1.0e-12,
            "dm4": 2.0e-6,
        }
        fine_atol = {
            "mass": 2.0e-12, "mean_x": 1.0e-12, "dm2": 2.0e-10,
            "pxx": 2.0e-10, "pyy": 1.0e-12, "qx": 1.0e-12,
            "dm4": 2.0e-8,
        }
        for key, target in exact.items():
            self.assertLessEqual(abs(moments["train"][key] - target), train_atol[key])
            self.assertLessEqual(abs(moments["fine"][key] - target), fine_atol[key])
        for key in ("mass", "dm2", "pxx", "dm4"):
            train_error = abs(moments["train"][key] - exact[key])
            fine_error = abs(moments["fine"][key] - exact[key])
            self.assertLess(fine_error, 0.02 * train_error)

    def test_stress_mode_is_orthogonal_to_invariants(self):
        quad = build_quadrature(9.0, 257, 9.0, 64)
        nodes = quad.nodes
        maxwellian = np.exp(-0.5 * np.sum(nodes**2, axis=1)) / (2.0 * np.pi) ** 1.5
        phi2 = (stress_mode_tf(tf.constant(nodes, tf.float64)).numpy().ravel())
        psi = invariant_features(nodes)
        for index in range(3):
            self.assertAlmostEqual(float(np.sum(quad.weights * maxwellian * phi2 * psi[:, index])), 0.0, places=9)
        self.assertGreater(float(np.sum(quad.weights * maxwellian * phi2**2)), 1.0)


class StressModelTests(unittest.TestCase):
    def test_hard_initial_condition_and_axisymmetry(self):
        model = small_model()
        rng = np.random.default_rng(4)
        cx = rng.normal(size=128)
        rho = np.abs(rng.normal(size=128))
        angle = rng.uniform(0.0, 2.0 * np.pi, size=128)
        a = np.stack([cx, rho, np.zeros_like(rho)], axis=1)
        b = np.stack([cx, rho * np.cos(angle), rho * np.sin(angle)], axis=1)
        zeros = tf.zeros([128, 1])
        log0 = model.raw_log_density(zeros, tf.constant(a, tf.float32)).numpy()
        exact = axisym_stress_initial_logpdf_tf(tf.constant(a, tf.float32)).numpy()
        np.testing.assert_allclose(log0, exact, atol=1.0e-6)
        self.assertEqual(float(tf.reduce_max(tf.abs(model.head(tf.zeros([4, 1]))))), 0.0)
        time = tf.fill([128, 1], 0.63)
        np.testing.assert_allclose(
            model.raw_log_density(time, tf.constant(a, tf.float32)).numpy(),
            model.raw_log_density(time, tf.constant(b, tf.float32)).numpy(),
            atol=2.0e-5,
        )

    def test_exact_cx_reflection_and_zero_heat_flux(self):
        """Every weight set must preserve the even stress-case parity."""
        model = small_model(seed=19)
        rng = np.random.default_rng(20)
        c = rng.normal(size=(256, 3)).astype(np.float32)
        reflected = c.copy()
        reflected[:, 0] *= -1.0
        time = tf.fill([len(c), 1], 0.61)
        forward = model.raw_log_density(time, tf.constant(c)).numpy()
        backward = model.raw_log_density(time, tf.constant(reflected)).numpy()
        np.testing.assert_array_equal(forward, backward)

        # The invariant tilt must retain the parity on the same symmetric
        # quadrature used in production.  Consequently qx is round-off zero,
        # far below the frozen relative gate of 1e-6 / STRESS_DELTA0.
        quad = quadrature_tensors(build_quadrature(8.0, 65, 8.0, 24))
        times = tf.constant([0.15, 0.55, 0.95], tf.float32)
        nt = int(times.shape[0])
        nv = quad.size
        flat_t = tf.reshape(tf.repeat(times[:, None], nv, axis=1), [-1, 1])
        flat_c = tf.reshape(tf.tile(quad.nodes32[None, :, :], [nt, 1, 1]), [-1, 3])
        log_raw = tf.cast(tf.reshape(model.raw_log_density(flat_t, flat_c), [nt, nv]), tf.float64)
        _, _, log_f = model.solve_tilt(log_raw, quad.psi64, quad.weights64)
        wf = quad.weights64[None, :] * tf.exp(log_f)
        moments = axisym_moment_tensors_tf(quad.cx64, quad.rho64, wf)
        self.assertLess(float(tf.reduce_max(tf.abs(moments["q"][:, 0]))), 1.0e-12)

    def test_tilt_restores_collision_invariants(self):
        model = small_model()
        quad = quadrature_tensors(build_quadrature(8.0, 129, 8.0, 32))
        state = assemble_slices(model, tf.constant([0.05, 0.4, 0.9], tf.float32), quad)
        np.testing.assert_allclose(state.moments64["mass"].numpy(), 1.0, atol=1.0e-12)
        np.testing.assert_allclose(state.moments64["mean"].numpy()[:, 0], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(state.moments64["dm2"].numpy(), 3.0, atol=1.0e-11)

    def test_portable_reload(self):
        model = small_model()
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "g2.weights.h5"
            model.save_weights(weights)
            clone = StructuredDensityModel(model.base_config, model.g1)
            clone.build_model()
            clone.load_weights(weights)
        c = tf.constant(np.random.default_rng(5).normal(size=(32, 3)), tf.float32)
        t = tf.fill([32, 1], 0.7)
        np.testing.assert_array_equal(model.raw_log_density(t, c).numpy(), clone.raw_log_density(t, c).numpy())


if __name__ == "__main__":
    unittest.main()

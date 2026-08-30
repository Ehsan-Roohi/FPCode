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
        quad = build_quadrature(8.0, 129, 8.0, 32)
        moments = axisymmetric_moments(quad, exact_initial_density(quad.nodes[:, 0], quad.nodes[:, 1]))
        self.assertAlmostEqual(moments["mass"], 1.0, places=11)
        self.assertAlmostEqual(moments["mean_x"], 0.0, places=12)
        self.assertAlmostEqual(moments["dm2"], 3.0, places=10)
        self.assertAlmostEqual(moments["pxx"], 1.6, places=10)
        self.assertAlmostEqual(moments["pyy"], 0.7, places=10)
        self.assertAlmostEqual(moments["qx"], 0.0, places=12)
        self.assertAlmostEqual(moments["dm4"], 16.08, places=8)

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

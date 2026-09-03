import unittest

import numpy as np
import tensorflow as tf

from f1_fp import (F1_GATES, analytic_dougherty_collision_tf,
                   collision_invariants_numpy,
                   conservative_dougherty_collision_tf,
                   conservative_collision_numpy, first_derivative_matrix,
                   raw_dougherty_collision_numpy)


class F1FokkerPlanckTests(unittest.TestCase):
    def test_registered_gates_are_fixed(self):
        self.assertEqual(F1_GATES["maximum_flux_relative_spread"], 1.0e-2)
        self.assertEqual(F1_GATES["collision_invariant_relative_rms"], 5.0e-6)
        self.assertEqual(F1_GATES["collision_projection_relative_rms"], 5.0e-2)

    def test_nonuniform_derivative_is_second_order_exact(self):
        x = np.array([0.0, 0.2, 0.9, 1.7, 3.0])
        derivative = first_derivative_matrix(x)
        np.testing.assert_allclose(derivative @ (2.0 + 3.0*x - 0.4*x*x),
                                   3.0 - 0.8*x, rtol=2e-13, atol=2e-13)

    def test_axisymmetric_operator_has_correct_shapes_and_is_finite(self):
        vx = np.linspace(-6.0, 6.0, 25)
        s = np.linspace(0.0, 36.0, 19) ** 1.15
        density = np.array([1.0, 1.7])
        velocity = np.array([1.3, 0.8])
        temperature = np.array([1.0, 1.6])
        f = np.stack([
            density[i]*(2*np.pi*temperature[i])**-1.5*np.exp(
                -((vx[:, None]-velocity[i])**2+s[None, :])/(2*temperature[i]))
            for i in range(2)])
        raw = raw_dougherty_collision_numpy(
            f, vx, s, density, velocity, temperature)
        self.assertEqual(raw.shape, f.shape)
        self.assertTrue(np.isfinite(raw).all())

    def test_discrete_projection_restores_collision_invariants(self):
        vx = np.linspace(-5.0, 5.0, 21)
        s = np.linspace(0.0, 25.0, 17)
        weights = np.ones((len(vx), len(s)))
        f = np.exp(-0.5*((vx[:, None]-0.7)**2+s[None, :]))[None, ...]
        raw = (0.1 + 0.2*vx[:, None] - 0.03*s[None, :])[None, ...] * f
        corrected = conservative_collision_numpy(raw, f, vx, s, weights)
        invariant = collision_invariants_numpy(corrected, vx, s, weights)
        np.testing.assert_allclose(invariant, 0.0, atol=2e-11, rtol=0.0)

    def test_tensorflow_collision_shapes_and_invariants(self):
        vx = np.linspace(-5.0, 5.0, 21).astype(np.float32)
        s = np.linspace(0.0, 25.0, 17).astype(np.float32)
        weight = np.ones((len(vx), len(s)), np.float32).reshape(-1)
        distribution = np.stack([
            np.exp(-0.5*((vx[:, None]-u)**2+s[None, :]))
            for u in (0.4, 0.8)]).reshape(2, -1).astype(np.float32)
        values = {"u": tf.constant([0.4, 0.8]),
                  "temperature": tf.constant([1.0, 1.0])}
        corrected, invariant, correction = conservative_dougherty_collision_tf(
            tf.constant(distribution), values, tf.constant(vx), tf.constant(s),
            tf.constant(weight),
            tf.constant(first_derivative_matrix(vx).astype(np.float32)),
            tf.constant(first_derivative_matrix(s).astype(np.float32)))
        self.assertEqual(tuple(corrected.shape), distribution.shape)
        self.assertTrue(np.isfinite(corrected.numpy()).all())
        self.assertLess(float(invariant), 2.0e-5)
        self.assertTrue(np.isfinite(float(correction)))

    def test_analytic_maxwellian_is_dougherty_equilibrium(self):
        vx = tf.constant(np.linspace(-7.0, 7.0, 41).astype(np.float32))
        s = tf.constant(np.tile(np.linspace(0.0, 49.0, 29), 41).astype(np.float32))
        velocity = tf.constant([0.7, 1.1])
        temperature = tf.constant([1.0, 1.4])
        density = tf.constant([1.0, 1.8])
        flat_vx = tf.repeat(vx, 29)
        weights = tf.ones_like(flat_vx)

        def log_builder(v, radial2):
            return (tf.math.log(density[:, None])
                    -1.5*tf.math.log(2*np.pi*temperature[:, None])
                    -0.5*((v-velocity[:, None])**2+radial2)/temperature[:, None])

        log_f = log_builder(flat_vx[None, :], s[None, :])
        f = tf.exp(log_f)
        collision, invariant, correction = analytic_dougherty_collision_tf(
            log_builder, f, {"u": velocity, "temperature": temperature},
            flat_vx, s, weights)
        self.assertLess(float(tf.reduce_max(tf.abs(collision))), 2.0e-6)
        self.assertLess(float(invariant), 2.0e-5)
        self.assertTrue(np.isfinite(float(correction)))


if __name__ == "__main__":
    unittest.main()

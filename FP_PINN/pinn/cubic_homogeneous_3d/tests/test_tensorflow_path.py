from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

try:
    import tensorflow as tf
except ModuleNotFoundError:  # The repository's lightweight CPU test environment.
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is tested inside the Unity dsmc-gpu environment")
class TensorFlowPathTests(unittest.TestCase):
    def test_heat_flux_model_is_axisymmetric_with_legacy_weight_shape(self) -> None:
        from train_stage2 import Config, DensityModel

        config = Config(
            case="heat_flux", output_dir="unused", reference="unused",
            width=16, depth=2, axisymmetric_heat_flux=True,
        )
        model = DensityModel(config)
        model.log_density(tf.zeros((1, 1)), tf.zeros((1, 3)))
        generator = tf.random.Generator.from_seed(9182)
        for variable in model.trainable_variables:
            variable.assign(generator.normal(variable.shape, dtype=variable.dtype))

        times = tf.fill((4, 1), tf.constant(0.63, tf.float32))
        velocities = tf.constant(
            [[0.4, 1.2, -0.7], [-0.9, 0.3, 1.8],
             [1.4, -1.1, -0.2], [-1.2, 0.8, -1.5]],
            dtype=tf.float32,
        )
        rotations = tf.stack(
            [velocities[:, 0], -velocities[:, 2], velocities[:, 1]], axis=1
        )
        reflected = velocities * tf.constant([1.0, -1.0, -1.0], tf.float32)
        baseline = model.log_density(times, velocities).numpy()
        np.testing.assert_allclose(
            model.log_density(times, rotations).numpy(), baseline,
            rtol=2e-6, atol=2e-6,
        )
        np.testing.assert_allclose(
            model.log_density(times, reflected).numpy(), baseline,
            rtol=2e-6, atol=2e-6,
        )

    def test_heat_flux_quadrature_is_antithetic(self) -> None:
        from train_stage2 import Config, sample_tf_proposal

        config = Config(
            case="heat_flux", output_dir="unused", reference="unused",
            n_time_batch=3, n_velocity_per_time=64,
            antithetic_heat_flux_quadrature=True,
        )
        time_grid, velocity_grid, log_q = sample_tf_proposal(config)
        quarter = config.n_velocity_per_time // 4
        signs = np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
             [1.0, 1.0, -1.0], [1.0, -1.0, -1.0]],
            dtype=np.float32,
        )
        for index, sign in enumerate(signs):
            target = slice(index * quarter, (index + 1) * quarter)
            np.testing.assert_allclose(
                velocity_grid.numpy()[:, target, :],
                velocity_grid.numpy()[:, :quarter, :] * sign,
                rtol=0.0, atol=0.0,
            )
            np.testing.assert_allclose(
                time_grid.numpy()[:, target, :], time_grid.numpy()[:, :quarter, :],
                rtol=0.0, atol=0.0,
            )
            np.testing.assert_allclose(
                log_q.numpy()[:, target, :], log_q.numpy()[:, :quarter, :],
                rtol=2e-6, atol=2e-6,
            )

    def test_fixed_heat_flux_quadrature_cycles_independent_panels(self) -> None:
        from train_stage2 import Config, build_fixed_quadrature_panels

        config = Config(
            case="heat_flux", output_dir="unused", reference="unused",
            n_time_batch=2, n_velocity_per_time=64,
            fixed_velocity_quadrature=True, quadrature_panels=3,
            antithetic_heat_flux_quadrature=True,
        )
        velocity, log_q = build_fixed_quadrature_panels(config)
        self.assertEqual(tuple(velocity.shape), (3, 2, 64, 3))
        self.assertEqual(tuple(log_q.shape), (3, 2, 64, 1))
        self.assertFalse(np.array_equal(velocity.numpy()[0], velocity.numpy()[1]))
        quarter = config.n_velocity_per_time // 4
        np.testing.assert_array_equal(
            velocity.numpy()[:, :, quarter:2 * quarter, :],
            velocity.numpy()[:, :, :quarter, :] * [1.0, -1.0, 1.0],
        )

    def test_zero_pde_defect_has_zero_weak_heat_flux_loss(self) -> None:
        from train_stage2 import weak_heat_flux_loss

        c = tf.constant(
            [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
            dtype=tf.float32,
        )
        ratio = tf.ones((1, 4, 1), tf.float32)
        residual = tf.zeros_like(ratio)
        mean = tf.zeros((1, 3), tf.float32)
        loss = weak_heat_flux_loss(c, ratio, residual, mean, scale=0.25)
        self.assertEqual(float(loss.numpy()), 0.0)

    def test_tensorflow_closure_matches_numpy(self) -> None:
        from cubic_operator import moments_from_samples, sample_initial, solve_closure
        from train_stage2 import Config, closure_tf

        rng = np.random.default_rng(918)
        for case in ("stress", "heat_flux"):
            values = sample_initial(case, 200_000, rng)
            m = moments_from_samples(values)
            config = Config(case=case, output_dir="unused", reference="unused")
            closure_np = solve_closure(m, regularization=config.closure_regularization)
            for dtype in (tf.float32, tf.float64):
                with self.subTest(case=case, dtype=dtype.name):
                    tensors = {
                        "pij": tf.constant(m.pij[None, :], dtype),
                        "q": tf.constant(m.q[None, :], dtype),
                        "m3": tf.constant(m.m3[None, :], dtype),
                        "m4": tf.constant(m.m4[None, :], dtype),
                        "m5": tf.constant(m.m5[None, :], dtype),
                        "dm2": tf.constant([m.dm2], dtype),
                        "dm4": tf.constant([m.dm4], dtype),
                    }
                    vector_tf, lambda_tf = closure_tf(tensors, config)
                    np.testing.assert_allclose(
                        vector_tf.numpy()[0], closure_np.vector,
                        rtol=2e-4, atol=2e-5,
                    )
                    self.assertAlmostEqual(
                        float(lambda_tf.numpy()[0]), closure_np.cubic_lambda,
                        places=6,
                    )

    def test_tail_log_density_hessian_is_finite(self) -> None:
        from train_stage2 import Config, DensityModel

        config = Config(
            case="heat_flux", output_dir="unused", reference="unused",
            width=16, depth=2,
        )
        model = DensityModel(config)
        times = tf.fill((6, 1), tf.constant(0.5, tf.float32))
        velocities = tf.constant(
            [
                [12.0, 0.0, 0.0],
                [-12.0, 0.0, 0.0],
                [0.0, 12.0, 0.0],
                [0.0, 0.0, -12.0],
                [8.0, 8.0, 8.0],
                [10.0, -7.0, 5.0],
            ],
            dtype=tf.float32,
        )
        with tf.GradientTape(persistent=True) as second_tape:
            second_tape.watch(velocities)
            with tf.GradientTape() as first_tape:
                first_tape.watch(velocities)
                log_f = model.log_density(times, velocities)
            grad_h = first_tape.gradient(log_f, velocities)
        hessian_h = second_tape.batch_jacobian(
            grad_h, velocities, experimental_use_pfor=True
        )
        del second_tape
        for name, value in (
            ("log density", log_f),
            ("log-density gradient", grad_h),
            ("log-density Hessian", hessian_h),
        ):
            self.assertTrue(np.all(np.isfinite(value.numpy())), name)

    def test_gauss_hermite_pseudo_proposal_recovers_exact_moments(self) -> None:
        from cubic_operator import equilibrium_logpdf, initial_logpdf, moments_from_samples
        from heat_flux_g0 import HEAT_FLUX_INITIAL_QX
        from train_stage2 import gauss_hermite_pseudo_proposal

        points, log_q = gauss_hermite_pseudo_proposal(16)
        count = points.shape[0]
        equilibrium_ratio = np.exp(equilibrium_logpdf(points) - log_q[:, 0])
        self.assertAlmostEqual(float(np.mean(equilibrium_ratio)), 1.0, places=6)
        equilibrium = moments_from_samples(
            points, weights=equilibrium_ratio / count
        )
        self.assertAlmostEqual(float(equilibrium.dm2), 3.0, places=5)

        heat_ratio = np.exp(initial_logpdf("heat_flux", points) - log_q[:, 0])
        heat = moments_from_samples(points, weights=heat_ratio / count)
        self.assertAlmostEqual(float(heat.mass), 1.0, places=6)
        self.assertAlmostEqual(float(heat.dm2), 3.0, places=5)
        self.assertAlmostEqual(
            float(heat.q[0]), HEAT_FLUX_INITIAL_QX, places=5
        )

    def test_one_training_step_is_finite(self) -> None:
        from train_stage2 import Config, DensityModel, make_train_step

        with tempfile.TemporaryDirectory() as temp:
            config = Config(
                case="heat_flux", output_dir=temp, reference="unused", epochs=1,
                n_time_batch=2, n_velocity_per_time=128, width=16, depth=2,
            )
            model = DensityModel(config)
            model.log_density(tf.zeros((1, 1)), tf.zeros((1, 3)))
            anchor = DensityModel(config)
            anchor.log_density(tf.zeros((1, 1)), tf.zeros((1, 3)))
            anchor.set_weights(model.get_weights())
            anchor.trainable = False
            optimizer = tf.keras.optimizers.Adam(1.0e-4)
            result = make_train_step(model, optimizer, config, anchor)()
            self.assertIn("heat_flux_history", result)
            self.assertIn("resume_anchor", result)
            self.assertAlmostEqual(float(result["resume_anchor"].numpy()), 0.0, places=7)
            for name, value in result.items():
                self.assertTrue(np.isfinite(float(value.numpy())), name)


if __name__ == "__main__":
    unittest.main()

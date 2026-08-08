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
    def test_tensorflow_closure_matches_numpy(self) -> None:
        from cubic_operator import moments_from_samples, sample_initial, solve_closure
        from train_stage2 import Config, closure_tf

        rng = np.random.default_rng(918)
        for case in ("stress", "heat_flux"):
            values = sample_initial(case, 200_000, rng)
            m = moments_from_samples(values)
            config = Config(case=case, output_dir="unused", reference="unused")
            tensors = {
                "pij": tf.constant(m.pij[None, :], tf.float32),
                "q": tf.constant(m.q[None, :], tf.float32),
                "m3": tf.constant(m.m3[None, :], tf.float32),
                "m4": tf.constant(m.m4[None, :], tf.float32),
                "m5": tf.constant(m.m5[None, :], tf.float32),
                "dm2": tf.constant([m.dm2], tf.float32),
                "dm4": tf.constant([m.dm4], tf.float32),
            }
            vector_tf, lambda_tf = closure_tf(tensors, config)
            closure_np = solve_closure(m, regularization=config.closure_regularization)
            np.testing.assert_allclose(vector_tf.numpy()[0], closure_np.vector, rtol=2e-4, atol=2e-5)
            self.assertAlmostEqual(float(lambda_tf.numpy()[0]), closure_np.cubic_lambda, places=6)

    def test_one_training_step_is_finite(self) -> None:
        from train_stage2 import Config, DensityModel, make_train_step

        with tempfile.TemporaryDirectory() as temp:
            config = Config(
                case="heat_flux", output_dir=temp, reference="unused", epochs=1,
                n_time_batch=2, n_velocity_per_time=128, width=16, depth=2,
            )
            model = DensityModel(config)
            model.log_density(tf.zeros((1, 1)), tf.zeros((1, 3)))
            optimizer = tf.keras.optimizers.Adam(1.0e-4)
            result = make_train_step(model, optimizer, config)()
            for name, value in result.items():
                self.assertTrue(np.isfinite(float(value.numpy())), name)


if __name__ == "__main__":
    unittest.main()


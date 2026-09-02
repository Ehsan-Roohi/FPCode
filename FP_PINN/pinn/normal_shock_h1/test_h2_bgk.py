import tempfile
import unittest
from pathlib import Path

import numpy as np

from h2_bgk import (H2_GATES, KNUDSEN_EFFECTIVE, PSI_MAX, hermite_modes,
                    compact_quadrature_arrays, conservative_fields_numpy,
                    moments_numpy, positive_log_tilt_numpy,
                    raw_projection_targets)


class H2BGKTests(unittest.TestCase):
    def test_registered_constants_are_fixed(self):
        self.assertEqual(KNUDSEN_EFFECTIVE, 1.0 / 80.0)
        self.assertEqual(PSI_MAX, 0.65)
        self.assertEqual(H2_GATES["qx_core_relative_l2"], 0.2)

    def test_axisymmetric_compression_preserves_represented_moments(self):
        root = Path(tempfile.mkdtemp()) / "quadrature.npz"
        grid = np.array([-1.0, 0.0, 1.0])
        v = np.array([(x, y, z) for x in grid for y in grid for z in grid])
        w = np.ones(len(v)) * 0.125
        np.savez(root, v=v, w=w)
        vx, r2, wc = compact_quadrature_arrays(root)
        f_compact = np.exp(-0.4 * ((vx - 0.2) ** 2 + r2))[None, :]
        f_full = np.exp(-0.4 * ((v[:, 0] - 0.2) ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2))[None, :]
        compact = moments_numpy(f_compact, vx, r2, wc)
        full = moments_numpy(f_full, v[:, 0], v[:, 1] ** 2 + v[:, 2] ** 2, w)
        for key in compact:
            np.testing.assert_allclose(compact[key], full[key], rtol=2e-14, atol=2e-14)

    def test_hermite_parity(self):
        vx = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        r2 = np.full_like(vx, 0.7)
        q, sigma = hermite_modes(vx, r2, 0.0, 1.0)
        np.testing.assert_allclose(q, -q[::-1])
        np.testing.assert_allclose(sigma, sigma[::-1])

    def test_conservative_macro_fields_have_machine_precision_fluxes(self):
        rho = np.linspace(1.0, 2.2, 19)
        temperature = np.linspace(1.0, 2.05, 19)
        flux = np.array([2.4, 8.7, 14.2])
        m = conservative_fields_numpy(rho, temperature, flux)
        mass = rho*m["u"]
        momentum = rho*m["u"]**2 + rho*temperature + m["sigma_xx"]
        energy = m["qx"] + m["u"]*(
            0.5*rho*m["u"]**2 + 2.5*rho*temperature + m["sigma_xx"])
        np.testing.assert_allclose(mass, flux[0], rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(momentum, flux[1], rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(energy, flux[2], rtol=2e-15, atol=2e-15)

    def test_five_projection_targets_encode_conservative_fields(self):
        rho = np.array([1.1, 1.7])
        temperature = np.array([1.2, 1.8])
        flux = np.array([2.4, 8.7, 14.2])
        fields = conservative_fields_numpy(rho, temperature, flux)
        target = raw_projection_targets(fields, flux, velocity_scale=2.0)
        np.testing.assert_allclose(target[:, 0], rho)
        np.testing.assert_allclose(target[:, 1]*2.0, flux[0])
        np.testing.assert_allclose(target[:, 3]*4.0, flux[1])
        np.testing.assert_allclose(target[:, 4]*8.0, flux[2])

    def test_bounded_log_tilt_is_positive_without_exploding(self):
        value = positive_log_tilt_numpy(
            np.array([-1.0e4, -75.0, -3.0]),
            np.array([-1.0e4, -10.0, 1.0e4]))
        self.assertTrue(np.all(value > 0.0))
        self.assertGreaterEqual(value.min(), np.exp(-80.0))
        self.assertLessEqual(value.max(), np.exp(9.0))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np

from h2_bgk import (H2_GATES, KNUDSEN_EFFECTIVE, PSI_MAX, hermite_modes,
                    compact_quadrature_arrays, moments_numpy)


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


if __name__ == "__main__":
    unittest.main()


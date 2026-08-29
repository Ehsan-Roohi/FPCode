"""Fast compatibility checks for the deterministic axisymmetric FV reference."""

from __future__ import annotations

import sys
import unittest
from unittest import mock
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE_DIR = HERE.parent.parent
for path in (str(STAGE_DIR), str(STAGE_DIR / "g1")):
    if path not in sys.path:
        sys.path.insert(0, path)

import axisym_fp_reference as reference  # noqa: E402
from axisym_fp_reference import AxisymGrid, initial_axisym, marginals  # noqa: E402


class AxisymReferenceCompatibilityTests(unittest.TestCase):
    def test_numpy_1_trapz_fallback(self):
        coordinates = np.linspace(0.0, 1.0, 101)
        # NumPy 1.x does not define ``trapezoid`` at all, so the mock must be
        # allowed to create the missing attribute before exercising fallback.
        with mock.patch.object(reference.np, "trapezoid", None, create=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                integral = reference.trapezoidal_integral(coordinates, coordinates)
        self.assertAlmostEqual(integral, 0.5, places=12)

    def test_marginals_are_finite_with_numpy_1_compatible_api(self):
        grid = AxisymGrid(L=6.0, nx=48, nr=32)
        density = initial_axisym(grid.X, grid.R)
        density /= grid.moments(density).mass
        centers = np.linspace(-4.0, 4.0, 41)

        marginal_x, marginal_y = marginals(grid, density, centers)

        self.assertEqual(marginal_x.shape, centers.shape)
        self.assertEqual(marginal_y.shape, centers.shape)
        self.assertTrue(np.all(np.isfinite(marginal_x)))
        self.assertTrue(np.all(np.isfinite(marginal_y)))
        self.assertTrue(np.all(marginal_x >= 0.0))
        self.assertTrue(np.all(marginal_y >= 0.0))


if __name__ == "__main__":
    unittest.main()

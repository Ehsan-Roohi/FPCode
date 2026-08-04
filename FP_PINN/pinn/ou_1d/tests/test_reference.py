"""Fast CPU checks for the analytic OU reference and sign convention."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from reference import exact_density, exact_second_moment, initial_density


class OrnsteinUhlenbeckReferenceTests(unittest.TestCase):
    def test_initial_condition_is_exact(self) -> None:
        v = np.linspace(-7.0, 7.0, 1001)
        np.testing.assert_allclose(exact_density(0.0, v), initial_density(v))

    def test_mass_is_one(self) -> None:
        v = np.linspace(-9.0, 9.0, 40001)
        for t in (0.0, 0.1, 0.5, 1.0):
            mass = np.trapezoid(exact_density(t, v), v)
            self.assertAlmostEqual(float(mass), 1.0, places=9)

    def test_second_moment(self) -> None:
        v = np.linspace(-9.0, 9.0, 40001)
        for t in (0.0, 0.25, 0.75, 1.0):
            numerical = np.trapezoid(v**2 * exact_density(t, v), v)
            self.assertAlmostEqual(
                float(numerical), float(exact_second_moment(t)), places=8
            )

    def test_exact_solution_satisfies_ou_pde(self) -> None:
        # Check f_t - (v f)_v - f_vv = 0 away from the truncated boundaries.
        t = 0.4
        v = np.linspace(-5.0, 5.0, 401)
        h = 2.0e-4
        f = exact_density(t, v)
        ft = (exact_density(t + h, v) - exact_density(t - h, v)) / (2.0 * h)
        drift_v = (
            (v + h) * exact_density(t, v + h)
            - (v - h) * exact_density(t, v - h)
        ) / (2.0 * h)
        fvv = (
            exact_density(t, v + h) - 2.0 * f + exact_density(t, v - h)
        ) / h**2
        residual = ft - drift_v - fvv
        self.assertLess(float(np.max(np.abs(residual))), 2.0e-6)


if __name__ == "__main__":
    unittest.main()


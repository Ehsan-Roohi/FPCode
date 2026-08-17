from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from cubic_operator import (
    ALL_CASE_NAMES,
    OOD_SUITE_CASES,
    analytic_initial_summary,
    case_default_nu,
    case_has_heat_flux,
    heat_flux_mixture_parameters,
    moments_from_samples,
    sample_initial,
)


class OODCaseTests(unittest.TestCase):
    def test_suite_has_control_and_required_perturbations(self) -> None:
        self.assertEqual(len(OOD_SUITE_CASES), 9)
        self.assertEqual(OOD_SUITE_CASES[0], "heat_flux")
        self.assertIn("ood_hf_shape_w020", OOD_SUITE_CASES)
        self.assertIn("ood_hf_nu050", OOD_SUITE_CASES)
        self.assertIn("ood_hf_nu200", OOD_SUITE_CASES)
        self.assertIn("ood_stress_strong", OOD_SUITE_CASES)
        self.assertIn("ood_coupled_axisym", OOD_SUITE_CASES)

    def test_every_case_has_unit_temperature_and_positive_mixture(self) -> None:
        for case in ALL_CASE_NAMES:
            with self.subTest(case=case):
                summary = analytic_initial_summary(case)
                self.assertAlmostEqual(float(np.sum(summary["pij"][[0, 3, 5]])), 3.0)
                self.assertGreater(case_default_nu(case), 0.0)
                if case_has_heat_flux(case):
                    weight, mean_a, mean_b, variance = heat_flux_mixture_parameters(case)
                    self.assertGreater(weight, 0.0)
                    self.assertLess(weight, 0.5)
                    self.assertGreater(mean_a, 0.0)
                    self.assertLess(mean_b, 0.0)
                    self.assertGreater(variance, 0.0)

    def test_monte_carlo_initial_moments_match_registry(self) -> None:
        rng = np.random.default_rng(20260816)
        for case in OOD_SUITE_CASES:
            with self.subTest(case=case):
                expected = analytic_initial_summary(case)
                measured = moments_from_samples(sample_initial(case, 120_000, rng))
                np.testing.assert_allclose(measured.mean, expected["mean"], atol=2.0e-2)
                np.testing.assert_allclose(measured.pij, expected["pij"], atol=3.5e-2)
                np.testing.assert_allclose(measured.q, expected["q"], atol=4.5e-2)


if __name__ == "__main__":
    unittest.main()

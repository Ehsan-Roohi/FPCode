from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from heat_flux_g0 import (  # noqa: E402
    HEAT_FLUX_DECAY_RATE_FACTOR,
    HEAT_FLUX_INITIAL_QX,
    analytic_heat_flux_history,
    effective_prandtl_number,
    fit_decay_rate,
    heat_flux_gate_summary,
    relative_l2,
)


class HeatFluxG0Tests(unittest.TestCase):
    def test_analytic_history_has_exact_initial_value_and_rate(self) -> None:
        times = np.array([0.0, 0.25, 0.5, 1.0])
        history = analytic_heat_flux_history(times)
        self.assertEqual(history[0], HEAT_FLUX_INITIAL_QX)
        self.assertAlmostEqual(
            history[-1],
            HEAT_FLUX_INITIAL_QX * np.exp(-HEAT_FLUX_DECAY_RATE_FACTOR),
            places=15,
        )
        self.assertAlmostEqual(fit_decay_rate(times, history), 4.0 / 3.0, places=14)
        self.assertAlmostEqual(
            effective_prandtl_number(fit_decay_rate(times, history)),
            2.0 / 3.0,
            places=14,
        )

    def test_frozen_gate_boundaries_pass_at_equality(self) -> None:
        self.assertEqual(heat_flux_gate_summary(0.0500001)["level"], "NO_GO")
        self.assertEqual(heat_flux_gate_summary(0.05)["level"], "CONTINUATION_PASS")
        self.assertEqual(heat_flux_gate_summary(0.02)["level"], "PRIMARY_PASS")
        self.assertEqual(heat_flux_gate_summary(0.01)["level"], "PUBLICATION_PASS")
        self.assertEqual(heat_flux_gate_summary(float("nan"))["level"], "NO_GO")

    def test_relative_l2_is_zero_for_analytic_identity(self) -> None:
        values = analytic_heat_flux_history(np.linspace(0.0, 1.0, 21))
        self.assertEqual(relative_l2(values, values), 0.0)


if __name__ == "__main__":
    unittest.main()

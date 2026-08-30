"""Fast compatibility and physics checks for the G2 FV reference."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE_DIR = HERE.parent.parent
for path in (str(STAGE_DIR), str(STAGE_DIR / "g2")):
    if path not in sys.path:
        sys.path.insert(0, path)

import axisym_stress_reference as reference  # noqa: E402


class StressReferenceTests(unittest.TestCase):
    def test_numpy_1_trapz_fallback(self):
        x = np.linspace(0.0, 1.0, 101)
        with mock.patch.object(reference.np, "trapezoid", None, create=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                value = reference.trapezoidal_integral(x, x)
        self.assertAlmostEqual(value, 0.5, places=12)

    def test_short_reference_tracks_analytic_stress(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(
                output_dir=temporary, L=7.0, nx=80, nr=40, dt=1.0e-3,
                tmax=0.1, save_interval=0.05, nu=1.0, hist_bins=80, hist_vmax=5.0,
            )
            metrics = reference.run(args)
        self.assertLess(metrics["stress_delta_vs_exact_relative_l2"], 0.005)
        self.assertLess(metrics["max_heat_flux_norm"], 1.0e-12)


if __name__ == "__main__":
    unittest.main()

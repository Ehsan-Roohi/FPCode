from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from cubic_operator import OOD_SUITE_CASES, case_default_nu


class ReferenceSmokeTests(unittest.TestCase):
    def test_all_cases_write_finite_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            for case in OOD_SUITE_CASES:
                destination = Path(temp) / case
                subprocess.run(
                    [
                        sys.executable, str(HERE / "reference_particle.py"),
                        "--case", case, "--output-dir", str(destination),
                        "--particles", "4000", "--dt", "0.01", "--tmax", "0.02",
                        "--save-every", "1", "--print-every", "10",
                        "--nu", str(case_default_nu(case)),
                    ],
                    check=True,
                    cwd=HERE,
                    stdout=subprocess.DEVNULL,
                )
                result = np.load(destination / "reference.npz")
                self.assertEqual(result["time"].shape, (3,))
                self.assertTrue(np.all(np.isfinite(result["pij"])))
                self.assertLess(np.max(np.abs(result["dm2"] - 3.0)), 1.0e-12)


if __name__ == "__main__":
    unittest.main()

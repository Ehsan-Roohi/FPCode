import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from h2_reference import load_reference, split_indices, heldout_metrics


class H2ReferenceTests(unittest.TestCase):
    def make_reference(self, neural=False):
        d = Path(tempfile.mkdtemp()) / "ref.csv"
        x = np.linspace(-40, 40, 101)
        rho = 1 + .5 * (1 + np.tanh(x / 4))
        u = 2 / rho
        temperature = 1 + .3 * (1 + np.tanh(x / 5))
        qx = -np.exp(-(x / 5) ** 2)
        sigma = np.exp(-(x / 6) ** 2)
        np.savetxt(d, np.column_stack((x, rho, u, temperature, qx, sigma)), delimiter=",",
                   header=",".join(("x", "rho", "u", "temperature", "qx", "sigma_xx")), comments="")
        d.with_suffix(".csv.json").write_text(json.dumps({"solver": "conservative DVM", "mach": 2.0, "neural": neural}))
        return d

    def test_reference_and_disjoint_split(self):
        ref = load_reference(self.make_reference(), expected_mach=2.0)
        s = split_indices(len(ref.x))
        self.assertEqual(len(np.intersect1d(np.union1d(s["macro"], s["moments"]), s["held_out"])), 0)
        m = heldout_metrics({k: getattr(ref, k) for k in ("rho", "u", "temperature", "qx", "sigma_xx")}, ref, s["held_out"])
        self.assertEqual(max(m.values()), 0.0)

    def test_neural_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "neural prediction"):
            load_reference(self.make_reference(neural=True), expected_mach=2.0)

    def test_mach_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Mach"):
            load_reference(self.make_reference(), expected_mach=5.0)


if __name__ == "__main__":
    unittest.main()

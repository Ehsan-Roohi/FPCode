import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from h2_reference import (KNOWN_FULLSTATES, heldout_metrics, load_reference,
                          split_indices, validation_regions)


class H2ReferenceTests(unittest.TestCase):
    def make_reference(self, neural=False):
        d = Path(tempfile.mkdtemp()) / "ref.csv"
        x = np.linspace(-40, 40, 101)
        rho = 1 + .5*(1+np.tanh(x/4))
        u = 2/rho
        temperature = 1 + .3*(1+np.tanh(x/5))
        qx = -np.exp(-(x/5)**2)
        sigma = np.exp(-(x/6)**2)
        np.savetxt(d, np.column_stack((x, rho, u, temperature, qx, sigma)), delimiter=",",
                   header=",".join(("x", "rho", "u", "temperature", "qx", "sigma_xx")), comments="")
        d.with_suffix(".csv.json").write_text(json.dumps(
            {"solver": "conservative DVM", "mach": 2.0, "neural": neural}))
        return d

    def test_reference_and_disjoint_split(self):
        ref = load_reference(self.make_reference(), expected_mach=2.0)
        s = split_indices(len(ref.x))
        self.assertEqual(len(np.intersect1d(np.union1d(s["macro"], s["moments"]), s["held_out"])), 0)
        regions = validation_regions(ref.x, s["held_out"])
        self.assertGreater(len(regions["held_out_core"]), 16)
        m = heldout_metrics({k: getattr(ref, k) for k in
                             ("rho", "u", "temperature", "qx", "sigma_xx")},
                            ref, regions["held_out_core"])
        self.assertEqual(max(m.values()), 0.0)

    def test_neural_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "neural prediction"):
            load_reference(self.make_reference(neural=True), expected_mach=2.0)

    def test_mach_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Mach"):
            load_reference(self.make_reference(), expected_mach=5.0)

    def test_unknown_fullstate_is_rejected_before_pickle(self):
        d = Path(tempfile.mkdtemp()) / "fullstate.npz"
        n, nv = 40, 3
        np.savez(d, x_mfp=np.linspace(-40, 40, n), f=np.ones((n, nv)),
                 v=np.zeros((nv, 3)), w=np.ones(nv), rho=np.ones(n), ux=np.ones(n),
                 T=np.ones(n), qx=np.zeros(n), sig=np.zeros(n),
                 states=np.array({"unsafe": "ignored"}, dtype=object))
        with self.assertRaisesRegex(ValueError, "unregistered full-state"):
            load_reference(d, expected_mach=2.0)

    def test_registered_fullstate_uses_numeric_members_only(self):
        d = Path(tempfile.mkdtemp()) / "fullstate.npz"
        n, nv = 40, 3
        np.savez(d, x_mfp=np.linspace(-40, 40, n), f=np.ones((n, nv)),
                 v=np.zeros((nv, 3)), w=np.ones(nv), rho=np.ones(n), ux=np.ones(n),
                 T=np.ones(n), qx=np.zeros(n), sig=np.zeros(n),
                 states=np.array({"legacy": True}, dtype=object))
        digest = hashlib.sha256(d.read_bytes()).hexdigest()
        KNOWN_FULLSTATES[digest] = {"solver": "independent conservative BGK DVM",
                                    "collision": "BGK", "mach": 2.0, "neural": False}
        try:
            ref = load_reference(d, expected_mach=2.0)
            self.assertEqual(ref.metadata["sha256"], digest)
            self.assertEqual(len(ref.x), n)
        finally:
            KNOWN_FULLSTATES.pop(digest)

    def test_numpy_gate_values_are_normalized_for_json(self):
        gates = {
            "endpoint": bool(np.float64(1e-4) < 5e-3),
            "tail": bool(np.float64(1e-3) < 5e-3),
        }
        self.assertEqual(json.loads(json.dumps(gates)),
                         {"endpoint": True, "tail": True})


if __name__ == "__main__":
    unittest.main()

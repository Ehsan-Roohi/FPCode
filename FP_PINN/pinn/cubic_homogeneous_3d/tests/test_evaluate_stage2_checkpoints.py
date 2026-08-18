from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

try:
    from evaluate_stage2_checkpoints import discover_checkpoints
except ModuleNotFoundError:  # TensorFlow is optional in lightweight CI.
    discover_checkpoints = None


@unittest.skipIf(discover_checkpoints is None, "TensorFlow checkpoint tooling unavailable")
class CheckpointDiscoveryTests(unittest.TestCase):
    def test_resume_baseline_is_part_of_global_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = root / "checkpoints_h5"
            checkpoints.mkdir()
            for path in (
                root / "resume_input.weights.h5",
                root / "stage2_final.weights.h5",
                checkpoints / "epoch-002000.weights.h5",
            ):
                path.write_bytes(b"portable")
            relative = [path.relative_to(root).as_posix() for path in discover_checkpoints(root)]
            self.assertEqual(relative[0], "resume_input.weights.h5")
            self.assertIn("checkpoints_h5/epoch-002000.weights.h5", relative)


if __name__ == "__main__":
    unittest.main()

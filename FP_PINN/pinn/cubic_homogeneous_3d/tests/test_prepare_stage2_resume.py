from __future__ import annotations

import sys
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from prepare_stage2_resume import allowed_relative_path


class ResumeArchiveTests(unittest.TestCase):
    def test_accepts_packaged_stage2_members(self) -> None:
        expected = {
            "config.json": Path("config.json"),
            "reference_particle/reference.npz": Path("reference_particle/reference.npz"),
            "checkpoints_h5/epoch-002500.weights.h5": Path(
                "checkpoints_h5/epoch-002500.weights.h5"
            ),
            "resume_input.weights.h5": Path("resume_input.weights.h5"),
            "prefix/stage2_best.weights.h5": Path("stage2_best.weights.h5"),
            "prefix/stage2_final.weights.h5": Path("stage2_final.weights.h5"),
        }
        for member, relative in expected.items():
            with self.subTest(member=member):
                self.assertEqual(allowed_relative_path(member), relative)

    def test_rejects_unrelated_or_unsafe_members(self) -> None:
        for member in (
            "../config.json",
            "/config.json",
            "loss_history.csv",
            "checkpoints_h5/not-a-weight.txt",
        ):
            with self.subTest(member=member):
                self.assertIsNone(allowed_relative_path(member))


if __name__ == "__main__":
    unittest.main()

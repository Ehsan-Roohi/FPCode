from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import package_stage2_ood_v4


class PackageStage2OODV4Tests(unittest.TestCase):
    def test_archive_is_atomic_and_surfaces_aggregate_results(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_root = root / "outputs" / "stage2-v4-123"
            aggregate = run_root / "aggregate"
            case = run_root / "heat_flux"
            aggregate.mkdir(parents=True)
            case.mkdir()
            (aggregate / "stage2_v4_ood_summary.json").write_text("{}\n")
            (case / "case_status.json").write_text("{}\n")
            archive = root / "FP_PINN_STAGE2_V4_OOD_JOB123_COMPLETE.zip"
            argv = [
                "package_stage2_ood_v4.py",
                "--run-root", str(run_root),
                "--archive", str(archive),
                "--repo-root", str(root),
                "--array-job-id", "123",
                "--collector-job-id", "124",
            ]
            with mock.patch.object(sys, "argv", argv):
                package_stage2_ood_v4.main()
            self.assertTrue(archive.is_file())
            self.assertFalse((root / f".{archive.name}.partial").exists())
            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
            self.assertIn("stage2_v4_ood_summary.json", names)
            self.assertIn("cases/heat_flux/case_status.json", names)
            self.assertIn("run_metadata.json", names)
            self.assertIn("MANIFEST.json", names)
            self.assertTrue(any(name.startswith("source_snapshot/") for name in names))


if __name__ == "__main__":
    unittest.main()

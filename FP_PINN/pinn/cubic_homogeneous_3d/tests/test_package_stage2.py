from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


HERE = Path(__file__).resolve().parents[1]


class PackageStage2Tests(unittest.TestCase):
    def test_archive_is_atomic_flat_and_contains_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = Path(raw_temp)
            case_output = temp / "case"
            case_output.mkdir()
            (case_output / "metrics.json").write_text('{"gate_passed": true}\n')
            nested = case_output / "reference_particle"
            nested.mkdir()
            (nested / "reference_metrics.json").write_text("{}\n")
            log = temp / "slurm.out"
            log.write_text("completed\n")
            archive = temp / "result.zip"

            subprocess.run(
                [
                    sys.executable, str(HERE / "package_stage2.py"),
                    "--case-output", str(case_output),
                    "--archive", str(archive),
                    "--repo-root", str(HERE),
                    "--job-id", "42_2", "--array-job-id", "42",
                    "--task-id", "2", "--case", "heat_flux",
                    "--log", str(log),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(archive.is_file())
            self.assertFalse(archive.with_name(f".{archive.name}.partial").exists())
            with zipfile.ZipFile(archive) as bundle:
                self.assertIn("metrics.json", bundle.namelist())
                self.assertIn(
                    "reference_particle/reference_metrics.json", bundle.namelist()
                )
                self.assertIn("slurm_logs/slurm.out", bundle.namelist())
                metadata = json.loads(bundle.read("run_metadata.json"))
            self.assertEqual(metadata["array_job_id"], "42")
            self.assertEqual(metadata["case"], "heat_flux")


if __name__ == "__main__":
    unittest.main()

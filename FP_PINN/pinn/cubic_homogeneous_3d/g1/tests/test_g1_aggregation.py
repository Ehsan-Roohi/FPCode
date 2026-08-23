"""Regression tests for the preregistered three-seed G1 verdict."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
G1_DIR = HERE.parent
if str(G1_DIR) not in sys.path:
    sys.path.insert(0, str(G1_DIR))

from aggregate_g1_seeds import EXPECTED_SCRATCH_TASKS, aggregate_tasks  # noqa: E402


def write_task(root: Path, name: str, seed: int, qx_error: float, status: str = "PASS") -> None:
    task = root / name
    task.mkdir(parents=True)
    (task / "config.json").write_text(json.dumps({"seed": seed, "resume_base_weights": None}) + "\n")
    metrics = {
        "status": status,
        "primary_pass": status == "PASS",
        "publication_pass": False,
        "qx_analytic_l2_fine": qx_error,
        "qx_quadrature_uncertainty_pp": 0.1,
        "decay_rate": 4.0 / 3.0,
        "marginal_relative_l2": 0.01,
        "raw_max_mass_error": 0.01,
        "raw_max_energy_error": 0.01,
        "stress_anisotropy_max_abs_error": 0.005,
        "residual_rms_max": 0.02,
        "field_relative_l2_max": 0.03,
        "gate_checks": {"qx_analytic_l2_primary": status == "PASS"},
    }
    (task / "metrics.json").write_text(json.dumps(metrics) + "\n")


class SeedAggregationTests(unittest.TestCase):
    def test_three_distinct_passing_seeds_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (name, seed) in enumerate(EXPECTED_SCRATCH_TASKS):
                write_task(root, name, seed, 0.012 + 0.002 * index)
            summary = aggregate_tasks(root, 1.0)
        self.assertEqual(summary["overall_status"], "PASS")
        self.assertTrue(summary["seed_agreement"]["all_expected_tasks_present"])
        self.assertTrue(summary["seed_agreement"]["scratch_seeds_unique"])

    def test_missing_seed_is_no_go(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, seed in EXPECTED_SCRATCH_TASKS[:2]:
                write_task(root, name, seed, 0.012)
            summary = aggregate_tasks(root, 1.0)
        self.assertEqual(summary["overall_status"], "NO_GO")
        self.assertEqual(summary["seed_agreement"]["n_scratch_seeds"], 2)
        self.assertIsNone(summary["seed_agreement"]["qx_l2_spread_pp"])

    def test_duplicate_seed_is_no_go(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, _ in EXPECTED_SCRATCH_TASKS:
                write_task(root, name, 7, 0.012)
            summary = aggregate_tasks(root, 1.0)
        self.assertEqual(summary["overall_status"], "NO_GO")
        self.assertFalse(summary["seed_agreement"]["scratch_seeds_unique"])

    def test_one_failed_gate_is_no_go(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (name, seed) in enumerate(EXPECTED_SCRATCH_TASKS):
                write_task(root, name, seed, 0.012, status="NO_GO" if index == 2 else "PASS")
            summary = aggregate_tasks(root, 1.0)
        self.assertEqual(summary["overall_status"], "NO_GO")
        self.assertFalse(summary["seed_agreement"]["all_seeds_pass"])


if __name__ == "__main__":
    unittest.main()

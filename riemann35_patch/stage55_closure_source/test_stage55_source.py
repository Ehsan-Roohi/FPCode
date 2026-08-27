#!/usr/bin/env python3
"""Small structural and numerical smoke test for Stage 55."""
from __future__ import annotations
import subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "riemann35_patch/stage55_closure_source/run_source_method.py"
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory)
    for method in ("qmc_refined", "exact_coeff_gaussian_projection", "gaussian_coeff_exact_projection", "gaussian_both", "compact_positive"):
        subprocess.run([sys.executable, str(SCRIPT), "--method", method, "--output", str(output), "--points-per-component", "64", "--base-points-per-component", "32", "--replicates", "1", "--dt", "0.01", "--final-time", "0.02", "--audit-times", "0,0.02"], cwd=ROOT, check=True)
        assert (output / f"stage55_{method}.npz").is_file()
print("Stage 55 smoke test passed")

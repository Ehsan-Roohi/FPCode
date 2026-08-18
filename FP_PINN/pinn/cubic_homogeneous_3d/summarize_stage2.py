#!/usr/bin/env python3
"""Print a compact table and return nonzero only for missing Stage-2 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cubic_operator import CASE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    missing = []
    print(
        f"{'case':<12} {'gate':<6} {'marginal L2':>12} {'mass max':>11} "
        f"{'momentum':>11} {'energy':>11} {'case error':>11}"
    )
    for case in CASE_NAMES:
        path = args.run_root / case / "metrics.json"
        if not path.exists():
            missing.append(str(path))
            print(f"{case:<12} MISSING")
            continue
        data = json.loads(path.read_text())
        print(
            f"{case:<12} {str(data['gate_passed']):<6} "
            f"{data['marginal_relative_l2']:12.4e} {data['max_mass_error']:11.4e} "
            f"{data['max_momentum_norm']:11.4e} {data['max_energy_error']:11.4e} "
            f"{data['case_relaxation_error']:11.4e}"
        )
    if missing:
        raise SystemExit("Missing result files:\n" + "\n".join(missing))


if __name__ == "__main__":
    main()


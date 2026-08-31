#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from riemann35_patch.stage71_harder_unseen.hard_cases import (
    CASE_NAMES,
    hard_case,
    registry_manifest,
)


def main() -> None:
    reg = registry_manifest()
    assert reg["qmc_used_to_define_cases"] is False
    assert reg["closure_parameters_refit"] is False
    for name in CASE_NAMES:
        case = hard_case(name)
        assert case.fingerprint == reg["case_fingerprints"][name]
        assert case.moments.shape == (35,)
        assert np.all(np.isfinite(case.moments))
        assert case.moments[0] > 0.0
        assert max(
            case.audit["mass_error"],
            case.audit["bulk_velocity_error"],
            case.audit["energy_trace_error"],
        ) < 1.0e-10
        assert case.audit["minimum_covariance_eigenvalue"] > 0.0
        print(name, case.fingerprint, case.audit["minimum_covariance_eigenvalue"])
    print("STAGE71_REGISTRY_FINGERPRINT=" + reg["registry_fingerprint"])
    print("STAGE71_PREFLIGHT=PASS")


if __name__ == "__main__":
    main()

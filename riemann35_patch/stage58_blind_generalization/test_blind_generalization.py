#!/usr/bin/env python3
"""Structural tests for the frozen Stage-58 generalization suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from riemann35_patch.stage54_heat_flux_third_order.run_heat_flux_method import (  # noqa: E402
    oblique_heat_flux_state,
)
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import (  # noqa: E402
    initialize_persistent_gaussian_mixture,
    persistent_gaussian_mixture_fp_step,
    persistent_gaussian_mixture_moments,
    stored_scalar_count,
)
from riemann35_patch.stage58_blind_generalization.blind_cases import (  # noqa: E402
    ANCHOR_CASE,
    BLIND_CASES,
    CASE_NAMES,
    blind_case,
    registry_manifest,
)


def test_registry_is_frozen_and_distinct() -> None:
    first = registry_manifest()
    second = registry_manifest()
    assert first == second
    assert first["qmc_used_to_define_cases"] is False
    assert len(first["registry_fingerprint"]) == 64
    assert len(set(first["case_fingerprints"].values())) == len(CASE_NAMES)
    assert tuple(first["blind_cases"]) == BLIND_CASES


def test_anchor_matches_stage57_exactly() -> None:
    anchor = blind_case(ANCHOR_CASE)
    stage57 = oblique_heat_flux_state()
    np.testing.assert_allclose(anchor.moments, stage57["moments"], rtol=2.0e-14, atol=2.0e-14)
    for left, right in zip(anchor.components, stage57["components"]):
        np.testing.assert_allclose(left[0], right[0], rtol=2.0e-14, atol=2.0e-14)
        np.testing.assert_allclose(left[1], right[1], rtol=2.0e-14, atol=2.0e-14)
        np.testing.assert_allclose(left[2], right[2], rtol=2.0e-14, atol=2.0e-14)


def test_case_invariants_and_positive_initialization() -> None:
    for name in CASE_NAMES:
        case = blind_case(name)
        assert case.audit["mass_error"] < 2.0e-13
        assert case.audit["bulk_velocity_error"] < 2.0e-13
        assert case.audit["energy_trace_error"] < 2.0e-13
        assert case.audit["minimum_covariance_eigenvalue"] > 0.0
        assert case.audit["minimum_absolute_third_component"] > 1.0e-8
        state = initialize_persistent_gaussian_mixture(case.components)
        assert stored_scalar_count(state) == 41
        np.testing.assert_allclose(
            persistent_gaussian_mixture_moments(state),
            case.moments,
            rtol=2.0e-13,
            atol=2.0e-13,
        )


def test_one_step_preserves_invariants_and_positivity() -> None:
    for name in CASE_NAMES:
        case = blind_case(name)
        state = initialize_persistent_gaussian_mixture(case.components)
        incoming = persistent_gaussian_mixture_moments(state)
        updated_state, updated, diagnostics = persistent_gaussian_mixture_fp_step(
            state, 3.125e-4, 1.0, quadrature_nodes=5
        )
        assert np.all(updated_state.probabilities > 0.0)
        assert diagnostics.minimum_covariance_eigenvalue > 0.0
        assert diagnostics.heat_flux_projection_fraction == 1.0
        assert diagnostics.heat_flux_projection_residual < 1.0e-8
        np.testing.assert_allclose(updated[0], incoming[0], rtol=0.0, atol=2.0e-13)


def main() -> None:
    test_registry_is_frozen_and_distinct()
    test_anchor_matches_stage57_exactly()
    test_case_invariants_and_positive_initialization()
    test_one_step_preserves_invariants_and_positivity()
    print("Stage 58 structural tests: PASS")


if __name__ == "__main__":
    main()

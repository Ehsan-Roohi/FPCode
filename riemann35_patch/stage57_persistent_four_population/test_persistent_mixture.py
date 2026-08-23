#!/usr/bin/env python3
"""Structural and regression tests for Stage 57."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import macroscopic_state  # noqa: E402
from riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux import (  # noqa: E402
    central_third_components,
    irreducible_decomposition,
    symmetric_tensor,
)
from riemann35_patch.stage54_heat_flux_third_order.run_heat_flux_method import (  # noqa: E402
    oblique_heat_flux_state,
)
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import (  # noqa: E402
    initialize_persistent_gaussian_mixture,
    persistent_gaussian_mixture_fp_step,
    persistent_gaussian_mixture_moments,
    stored_scalar_count,
)
from riemann35_patch.stage57_persistent_four_population.run_persistent_method import (  # noqa: E402
    _initial_source_audit,
)


def test_exact_positive_initialization_and_source() -> None:
    initial = oblique_heat_flux_state()
    state = initialize_persistent_gaussian_mixture(initial["components"])
    moments = persistent_gaussian_mixture_moments(state)
    np.testing.assert_allclose(moments, initial["moments"], rtol=0.0, atol=2.0e-14)
    assert stored_scalar_count(state) == 41
    assert np.min(state.probabilities) > 0.0
    assert min(np.min(np.linalg.eigvalsh(item)) for item in state.covariances) > 0.0
    audit = _initial_source_audit(
        tuple(initial["components"]),
        moments,
        tau=1.0,
        prandtl=2.0 / 3.0,
        quadrature_nodes=5,
    )
    assert audit["initial_moment_relative_residual"] < 1.0e-13
    assert audit["initial_tail_relative_error"] < 1.0e-12
    assert audit["initial_third_source_relative_error"] < 1.0e-12


def test_projection_preserves_lower_and_tracefree_moments() -> None:
    initial = oblique_heat_flux_state()
    base = initialize_persistent_gaussian_mixture(initial["components"])
    _, unprojected, _ = persistent_gaussian_mixture_fp_step(
        base, 0.00125, 1.0, enforce_heat_flux_rate=False
    )
    _, projected, diagnostics = persistent_gaussian_mixture_fp_step(
        base, 0.00125, 1.0, enforce_heat_flux_rate=True
    )
    base_macro = macroscopic_state(initial["moments"])
    unprojected_macro = macroscopic_state(unprojected)
    projected_macro = macroscopic_state(projected)
    np.testing.assert_allclose(
        projected_macro.velocity, unprojected_macro.velocity, rtol=0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        projected_macro.covariance,
        unprojected_macro.covariance,
        rtol=2.0e-12,
        atol=2.0e-13,
    )
    desired_heat_flux = np.exp(-2.0 * (2.0 / 3.0) * 0.00125) * base_macro.heat_flux
    np.testing.assert_allclose(
        projected_macro.heat_flux, desired_heat_flux, rtol=2.0e-12, atol=2.0e-13
    )
    unprojected_components = central_third_components(unprojected[None, :])
    projected_components = central_third_components(projected[None, :])
    unprojected_tracefree = irreducible_decomposition(unprojected_components)[2]
    projected_tracefree = irreducible_decomposition(projected_components)[2]
    np.testing.assert_allclose(
        projected_tracefree, unprojected_tracefree, rtol=2.0e-11, atol=2.0e-13
    )
    assert diagnostics.heat_flux_projection_fraction == 1.0
    assert diagnostics.heat_flux_projection_residual < 1.0e-10
    assert diagnostics.minimum_quadrature_weight > 0.0
    assert diagnostics.minimum_covariance_eigenvalue > 0.0
    assert diagnostics.realizability_margin >= -5.0e-13


def test_rotation_covariance() -> None:
    initial = oblique_heat_flux_state()
    angle = np.deg2rad(23.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_components = tuple(
        (
            weight,
            rotation @ np.asarray(mean),
            rotation @ np.asarray(covariance) @ rotation.T,
        )
        for weight, mean, covariance in initial["components"]
    )
    base_state = initialize_persistent_gaussian_mixture(initial["components"])
    rotated_state = initialize_persistent_gaussian_mixture(rotated_components)
    _, base_moments, _ = persistent_gaussian_mixture_fp_step(
        base_state, 0.00125, 1.0
    )
    _, rotated_moments, _ = persistent_gaussian_mixture_fp_step(
        rotated_state, 0.00125, 1.0
    )
    base_macro = macroscopic_state(base_moments)
    rotated_macro = macroscopic_state(rotated_moments)
    np.testing.assert_allclose(
        rotated_macro.heat_flux,
        rotation @ base_macro.heat_flux,
        rtol=2.0e-11,
        atol=2.0e-13,
    )
    base_components = central_third_components(base_moments[None, :])
    rotated_components_values = central_third_components(rotated_moments[None, :])
    base_tensor = symmetric_tensor(base_components)[0]
    rotated_tensor = symmetric_tensor(rotated_components_values)[0]
    expected = np.einsum(
        "ia,jb,kc,abc->ijk", rotation, rotation, rotation, base_tensor
    )
    np.testing.assert_allclose(rotated_tensor, expected, rtol=3.0e-11, atol=3.0e-13)


def main() -> None:
    test_exact_positive_initialization_and_source()
    test_projection_preserves_lower_and_tracefree_moments()
    test_rotation_covariance()
    print("Stage 57 structural tests: PASS")


if __name__ == "__main__":
    main()

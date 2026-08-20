"""Structural tests for the predeclared Stage-53 boundary audit."""

from __future__ import annotations

import numpy as np

from hyqmom_fp import HYQMOM_35_INDICES, mixture_of_gaussians_moments_35
from riemann35_patch.stage53_boundary_realizability.run_boundary_realizability import (
    METHODS,
    POSITION,
    advance_method,
    configuration,
    configuration_for_epsilon,
    crossing_jet_components,
    esbgk_collision_step,
    field_margins,
    gaussian_diffusion_step,
    global_invariants,
    initialize_crossing_jet_field,
    quadrature_diagnostics,
)


def test_boundary_family_is_positive_and_strictly_approaches_h2_zero() -> None:
    config = configuration("unity")
    h2_values = []
    h4_values = []
    for epsilon in config.epsilons:
        field = initialize_crossing_jet_field(config, epsilon)
        h2, h4 = field_margins(field)
        h2_values.append(h2)
        h4_values.append(h4)
        assert h2 > 0.0
        assert h4 > 0.0
    assert all(right < left for left, right in zip(h2_values, h2_values[1:]))
    assert all(right < left for left, right in zip(h4_values, h4_values[1:]))


def test_exact_diffusion_matches_gaussian_covariance_convolution() -> None:
    components = crossing_jet_components(0.03, 0.37)
    incoming = mixture_of_gaussians_moments_35(components)
    diffusivity = 0.41
    dt = 0.02
    mapped = gaussian_diffusion_step(incoming, dt, diffusivity)
    variance_increment = 2.0 * diffusivity * dt
    expected = mixture_of_gaussians_moments_35(
        tuple(
            (
                weight,
                mean,
                covariance + variance_increment * np.eye(3),
            )
            for weight, mean, covariance in components
        )
    )
    np.testing.assert_allclose(mapped, expected, rtol=2.0e-13, atol=2.0e-13)


def test_esbgk_baseline_is_positive_and_preserves_collision_invariants() -> None:
    components = crossing_jet_components(0.03, 0.37)
    incoming = mixture_of_gaussians_moments_35(components)
    updated = esbgk_collision_step(incoming, 0.0025, 1.0, 2.0 / 3.0)
    field = np.asarray([incoming, updated])
    h2, h4 = field_margins(field)
    assert h2 > 0.0
    assert h4 > 0.0
    invariant_positions = [
        POSITION[(0, 0, 0)],
        POSITION[(1, 0, 0)],
        POSITION[(0, 1, 0)],
        POSITION[(0, 0, 1)],
    ]
    np.testing.assert_allclose(
        updated[invariant_positions], incoming[invariant_positions], atol=2.0e-14
    )
    energy_positions = [
        POSITION[(2, 0, 0)],
        POSITION[(0, 2, 0)],
        POSITION[(0, 0, 2)],
    ]
    np.testing.assert_allclose(
        np.sum(updated[energy_positions]),
        np.sum(incoming[energy_positions]),
        atol=2.0e-14,
    )


def test_every_declared_arm_advances_a_near_boundary_state() -> None:
    config = configuration("smoke")
    initial = initialize_crossing_jet_field(config, 0.03)
    initial_invariants = global_invariants(initial)
    for method in METHODS:
        updated, cfl = advance_method(method, initial, config.fine_dt, config)
        assert updated.shape == (config.cells, len(HYQMOM_35_INDICES))
        assert np.all(np.isfinite(updated))
        assert cfl < 0.8
        h2, h4 = field_margins(updated)
        assert h2 > 0.0
        assert h4 > 0.0
        if method != "diffusion_only":
            np.testing.assert_allclose(
                global_invariants(updated),
                initial_invariants,
                rtol=2.0e-13,
                atol=2.0e-13,
            )
        if method != "esbgk_collision_baseline":
            minimum_weight, residual = quadrature_diagnostics(updated)
            assert minimum_weight > 0.0
            assert residual < 2.0e-2


def test_time_refinement_and_gates_are_predeclared() -> None:
    config = configuration("unity")
    assert config.coarse_dt == 2.0 * config.fine_dt
    assert config.final_time / config.coarse_dt == 100
    assert config.h_tolerance == 2.0e-10
    assert config.refinement_tolerance == 5.0e-2
    assert len(METHODS) == len(set(METHODS)) == 5


def test_epsilon_array_selection_preserves_all_locked_controls() -> None:
    base = configuration("unity")
    for index, epsilon in enumerate(base.epsilons):
        selected = configuration_for_epsilon("unity", index)
        assert selected.epsilons == (epsilon,)
        assert selected.cells == base.cells
        assert selected.final_time == base.final_time
        assert selected.coarse_dt == base.coarse_dt
        assert selected.fine_dt == base.fine_dt
        assert selected.h_tolerance == base.h_tolerance
        assert selected.conservation_tolerance == base.conservation_tolerance
        assert selected.refinement_tolerance == base.refinement_tolerance

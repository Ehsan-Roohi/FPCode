"""Focused structural tests for the isolated Stage-34 audit."""

from __future__ import annotations

import numpy as np

from hyqmom_fp import HYQMOM_35_INDICES, mixture_of_gaussians_moments_35
from riemann35_patch.stage34_two_peak.run_two_peak_audit import (
    COMPONENTS,
    all_indices_through,
    analytic_bgk_esbgk_histories,
    analytic_mixture_moments,
    degree_eight_margin,
    invariant_drift,
    moments_from_nodes,
    run_stage9_mixture,
)


def test_two_peak_definition_and_exact_retained_moments() -> None:
    assert len(COMPONENTS) == 2
    assert COMPONENTS[0][0] == COMPONENTS[1][0] == 0.5
    np.testing.assert_allclose(COMPONENTS[0][1], -np.asarray(COMPONENTS[1][1]))
    np.testing.assert_allclose(COMPONENTS[0][2], COMPONENTS[1][2])
    separation_ratio = abs(COMPONENTS[0][1][0]) / np.sqrt(COMPONENTS[0][2][0, 0])
    assert separation_ratio > 1.0  # genuinely bimodal, not merely two-component

    indices = all_indices_through(8)
    exact = analytic_mixture_moments(COMPONENTS, indices)
    retained = mixture_of_gaussians_moments_35(COMPONENTS)
    lookup = {index: value for index, value in zip(indices, exact)}
    np.testing.assert_allclose(
        retained,
        [lookup[index] for index in HYQMOM_35_INDICES],
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_positive_measure_has_nonnegative_degree_eight_matrix() -> None:
    # A small tensor-product positive rule is sufficient for this structural
    # test; production uses the independently scrambled Sobol rule.
    nodes_1d, weights_1d = np.polynomial.hermite.hermgauss(5)
    variance = float(COMPONENTS[0][2][0, 0])
    nodes_1d *= np.sqrt(2.0 * variance)
    weights_1d /= np.sqrt(np.pi)
    nodes = []
    weights = []
    for component_weight, mean, _ in COMPONENTS:
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    nodes.append(
                        [mean[0] + nodes_1d[i], mean[1] + nodes_1d[j], mean[2] + nodes_1d[k]]
                    )
                    weights.append(
                        component_weight * weights_1d[i] * weights_1d[j] * weights_1d[k]
                    )
    indices = all_indices_through(8)
    moments = moments_from_nodes(np.asarray(nodes), np.asarray(weights), indices)
    margin, eigenvalue = degree_eight_margin(moments, indices)
    assert margin > -1.0e-12
    assert eigenvalue > -1.0e-12


def test_stage9_one_step_is_realizable_and_complete() -> None:
    result = run_stage9_mixture(
        dt=1.0e-3,
        final_time=1.0e-3,
        sample_interval=1.0e-3,
        tau=1.0,
        prandtl=2.0 / 3.0,
    )
    assert result["status"] == "REACHED_FINAL_TIME"
    assert result["completed_steps"] == 1
    assert np.min(result["margin35_history"]) > 0.0
    assert np.min(result["margin8_history"]) > 0.0


def test_analytic_bgk_and_esbgk_preserve_invariants() -> None:
    bgk, esbgk, metadata = analytic_bgk_esbgk_histories(
        [0.0, 0.2, 1.0], tau=1.0, prandtl=2.0 / 3.0
    )
    assert metadata["nu"] == -0.5
    assert metadata["tau_sigma"] == 0.5
    for history in (bgk, esbgk):
        drift = invariant_drift(history)
        assert drift["maximum_mass_drift"] < 2.0e-14
        assert drift["maximum_momentum_drift"] < 2.0e-14
        assert drift["maximum_energy_trace_drift"] < 2.0e-14

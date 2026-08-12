"""Structural qualification tests for the Stage-24D clean-room DVM."""

from __future__ import annotations

import numpy as np

from hyqmom_fp import (
    CubicFPCoefficients,
    DVMGrid,
    DVMState,
    bernoulli_function,
    dvm_cubic_fp_step,
    project_cell_masses_minimum_kl,
    scharfetter_gummel_proposal,
)


def test_bernoulli_function_limits_and_flux_bias() -> None:
    values = bernoulli_function(np.asarray([-50.0, 0.0, 50.0]))
    assert np.isclose(values[1], 1.0)
    assert values[0] > 49.9
    assert values[2] < 1.0e-19


def test_exact_cell_integrated_moments() -> None:
    grid = DVMGrid(lower=(-1.0, -2.0, -3.0), upper=(1.0, 2.0, 3.0), shape=(4, 4, 4))
    masses = np.full(grid.size, 1.0 / grid.size)
    state = DVMState(grid, masses)
    moments = state.moments(((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)))
    assert np.allclose(moments, [1.0, 1.0 / 3.0, 4.0 / 3.0, 3.0], atol=2.0e-15)


def test_minimum_kl_matches_feasible_cell_target() -> None:
    grid = DVMGrid(lower=(-2.0, -2.0, -2.0), upper=(2.0, 2.0, 2.0), shape=(5, 5, 5))
    centers = grid.centers()
    proposal_masses = np.exp(-0.5 * np.einsum("ni,ni->n", centers, centers))
    proposal_masses /= np.sum(proposal_masses)
    proposal = DVMState(grid, proposal_masses)
    target_masses = proposal_masses * np.exp(0.03 * centers[:, 0] ** 3 - 0.02 * centers[:, 1] ** 2)
    target_masses /= np.sum(target_masses)
    target = DVMState(grid, target_masses).moments()
    projected, diagnostics = project_cell_masses_minimum_kl(proposal, target)
    assert diagnostics.relative_moment_residual < 2.0e-9
    assert diagnostics.iterations > 0
    assert np.min(projected.masses) > 0.0


def test_sg_proposal_is_positive_and_mass_conservative() -> None:
    grid = DVMGrid(lower=(-4.0, -4.0, -4.0), upper=(4.0, 4.0, 4.0), shape=(7, 7, 7))
    centers = grid.centers()
    masses = np.exp(-0.5 * np.einsum("ni,ni->n", centers, centers))
    masses /= np.sum(masses)
    state = DVMState(grid, masses)
    coefficients = CubicFPCoefficients.ornstein_uhlenbeck(tau=1.0, theta=1.0)
    proposed = scharfetter_gummel_proposal(state, 0.02, coefficients)
    assert np.min(proposed.masses) > 0.0
    assert abs(np.sum(proposed.masses) - np.sum(state.masses)) < 2.0e-14


def test_guided_step_conserves_collision_invariants() -> None:
    grid = DVMGrid(lower=(-4.0, -4.0, -4.0), upper=(4.0, 4.0, 4.0), shape=(7, 7, 7))
    centers = grid.centers()
    masses = np.exp(-0.5 * np.einsum("ni,ni->n", centers, centers))
    masses *= 1.0 + 0.02 * np.tanh(centers[:, 0])
    masses /= np.sum(masses)
    state = DVMState(grid, masses)
    final, diagnostics = dvm_cubic_fp_step(state, 2.0e-4, 1.0, guided=True)
    assert diagnostics.projection is not None
    assert diagnostics.projection.relative_moment_residual < 2.0e-8
    assert diagnostics.mass_drift < 2.0e-10
    assert diagnostics.momentum_drift < 2.0e-9
    assert diagnostics.energy_drift < 2.0e-9
    assert np.min(final.masses) > 0.0
    phases = (
        diagnostics.initial_source_seconds,
        diagnostics.sg_proposal_seconds,
        diagnostics.proposal_source_seconds,
        diagnostics.target_assembly_seconds,
        diagnostics.projection_seconds,
        diagnostics.final_diagnostics_seconds,
    )
    assert all(value >= 0.0 for value in phases)
    assert diagnostics.total_seconds >= sum(phases)

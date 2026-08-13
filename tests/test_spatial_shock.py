"""Structural tests for the Stage-25A one-dimensional shock transport."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from hyqmom_fp import (
    DVMGrid,
    SpatialDVMState,
    SpatialGrid1D,
    dvm_upwind_transport_step,
    full_dvm_shock_step,
    adaptive_shock_step,
    initialize_adaptive_normal_shock,
    initialize_normal_shock_dvm,
    initialize_normal_shock_moments,
    macro_upwind_transport_step,
    normal_shock_rankine_hugoniot,
    stage25_hysteresis,
)
from hyqmom_fp.mixture_closure import (
    fit_equal_variance_marginal,
    fit_location_scale_marginal,
)
from riemann35_patch.stage25a.run_normal_shock import (
    _restore_checkpoint,
    _write_checkpoint,
    configuration,
)


def _small_grids() -> tuple[SpatialGrid1D, DVMGrid]:
    return (
        SpatialGrid1D(-3.0, 3.0, 6),
        DVMGrid(
            lower=(-12.0, -10.0, -10.0),
            upper=(14.0, 10.0, 10.0),
            shape=(17, 13, 13),
        ),
    )


def test_rankine_hugoniot_fluxes_match() -> None:
    shock = normal_shock_rankine_hugoniot(3.0)
    rho1, u1, t1 = (
        shock.upstream_density,
        shock.upstream_velocity,
        shock.upstream_theta,
    )
    rho2, u2, t2 = (
        shock.downstream_density,
        shock.downstream_velocity,
        shock.downstream_theta,
    )
    assert np.isclose(rho1 * u1, rho2 * u2, rtol=2.0e-14)
    assert np.isclose(
        rho1 * u1**2 + rho1 * t1,
        rho2 * u2**2 + rho2 * t2,
        rtol=2.0e-14,
    )
    h1 = 2.5 * t1 + 0.5 * u1**2
    h2 = 2.5 * t2 + 0.5 * u2**2
    assert np.isclose(h1, h2, rtol=2.0e-14)


def test_uniform_dvm_transport_is_positive_and_stationary() -> None:
    xgrid, vgrid = _small_grids()
    shock = normal_shock_rankine_hugoniot(3.0)
    _, left, _ = initialize_normal_shock_dvm(xgrid, vgrid, shock)
    uniform = SpatialDVMState(
        xgrid, vgrid, np.repeat(left.masses[None, :], xgrid.cells, axis=0)
    )
    final, diagnostics = dvm_upwind_transport_step(uniform, 0.02, left, left)
    assert np.allclose(final.masses, uniform.masses, rtol=0.0, atol=3.0e-17)
    assert diagnostics.minimum_mass > 0.0
    assert diagnostics.mass_balance_residual < 2.0e-13
    assert diagnostics.momentum_balance_residual < 2.0e-13
    assert diagnostics.energy_balance_residual < 2.0e-13


def test_uniform_macro_transport_is_stationary() -> None:
    xgrid, _ = _small_grids()
    shock = normal_shock_rankine_hugoniot(3.0)
    upstream = shock.upstream_moments
    field = np.repeat(upstream[None, :], xgrid.cells, axis=0)
    final, diagnostics = macro_upwind_transport_step(
        field, xgrid, 0.01, upstream, upstream
    )
    assert np.allclose(final, field, rtol=2.0e-13, atol=2.0e-13)
    assert diagnostics.mass_balance_residual < 2.0e-12


def test_adaptive_initial_birth_is_known_and_local() -> None:
    xgrid, vgrid = _small_grids()
    shock = normal_shock_rankine_hugoniot(3.0)
    adaptive, _, _ = initialize_adaptive_normal_shock(
        xgrid, vgrid, shock, initial_active_half_width=1
    )
    assert np.sum(adaptive.active) == 3
    assert adaptive.transition_count == 3
    assert adaptive.blocked_births == 0
    assert np.min(adaptive.micro_masses[adaptive.active]) > 0.0
    expected = initialize_normal_shock_moments(xgrid, shock)
    assert np.allclose(adaptive.moments, expected)
    policy = stage25_hysteresis()
    assert policy.tail_on == 0.40
    assert policy.source_on == 0.10124


def test_one_coupled_shock_step_is_positive_causal_and_conservative() -> None:
    xgrid, vgrid = _small_grids()
    shock = normal_shock_rankine_hugoniot(3.0)
    full, left, right = initialize_normal_shock_dvm(xgrid, vgrid, shock)
    full_next, full_diagnostics = full_dvm_shock_step(
        full, 0.005, 1.0, left, right
    )
    assert np.min(full_next.masses) > 0.0
    assert full_diagnostics.transport.mass_balance_residual < 2.0e-12
    assert full_diagnostics.transport.momentum_balance_residual < 2.0e-12
    assert full_diagnostics.transport.energy_balance_residual < 2.0e-12
    assert full_diagnostics.maximum_projection_residual < 2.0e-9

    adaptive, left, right = initialize_adaptive_normal_shock(
        xgrid, vgrid, shock, initial_active_half_width=1
    )
    adaptive_next, adaptive_diagnostics = adaptive_shock_step(
        adaptive, 0.005, 1.0, left, right
    )
    assert adaptive_diagnostics.blocked_births == 0
    assert adaptive_diagnostics.maximum_micro_macro_residual < 2.0e-8
    assert adaptive_diagnostics.transport.mass_balance_residual < 2.0e-10
    assert adaptive_diagnostics.transport.momentum_balance_residual < 2.0e-10
    assert adaptive_diagnostics.transport.energy_balance_residual < 2.0e-10
    assert np.min(adaptive_next.micro_masses[adaptive_next.active]) > 0.0


def test_narrow_marginal_branch_is_scale_invariant() -> None:
    """Regression for the Stage-25A qualification failure on Unity.

    The old absolute tolerance sent this narrow, weakly leptokurtic state to
    the equal-variance Pearson branch, where its component weight became
    degenerate.  Scaling the velocity must not change the selected physical
    branch or its normalized reconstruction accuracy.
    """

    second = 8.211656829056058e-6
    third = 5.430200735561627e-13
    fourth = 2.0278273191294818e-10
    base = fit_location_scale_marginal(second, third, fourth)
    velocity_scale = 1.0e3
    scaled = fit_location_scale_marginal(
        second * velocity_scale**2,
        third * velocity_scale**3,
        fourth * velocity_scale**4,
    )
    assert base.branch == scaled.branch
    assert np.allclose(base.weights, scaled.weights, rtol=2.0e-11, atol=2.0e-13)
    assert np.allclose(
        base.means * velocity_scale, scaled.means, rtol=2.0e-10, atol=2.0e-12
    )
    assert np.allclose(
        base.component_variances * velocity_scale**2,
        scaled.component_variances,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    assert base.reconstruction_error < 2.0e-10
    assert scaled.reconstruction_error < 2.0e-10


def test_singular_pearson_carrier_uses_bounded_residual_fallback() -> None:
    marginal = fit_equal_variance_marginal(
        8.211656829056058e-6,
        5.430200735561627e-13,
        2.0278273191294818e-10,
    )
    assert marginal.branch == "degenerate-residual-fallback"
    assert np.array_equal(marginal.weights, np.asarray([1.0]))
    assert np.array_equal(marginal.means, np.asarray([0.0]))
    assert marginal.component_variances[0] > 0.0
    assert np.isfinite(marginal.reconstruction_error)


def test_stage25a_checkpoint_round_trip_is_exact() -> None:
    config = configuration("smoke", None)
    xgrid, vgrid = _small_grids()
    # Keep the checkpoint test small while exercising the same state types.
    config.update(
        {
            "x_lower": xgrid.lower,
            "x_upper": xgrid.upper,
            "spatial_cells": xgrid.cells,
            "v_lower": vgrid.lower,
            "v_upper": vgrid.upper,
            "velocity_shape": vgrid.shape,
            "initial_active_half_width": 1,
        }
    )
    shock = normal_shock_rankine_hugoniot(3.0)
    reference, _, _ = initialize_normal_shock_dvm(xgrid, vgrid, shock)
    macro = initialize_normal_shock_moments(xgrid, shock)
    adaptive, _, _ = initialize_adaptive_normal_shock(
        xgrid, vgrid, shock, initial_active_half_width=1
    )
    dt = float(config["cfl"]) * xgrid.width / np.max(
        np.abs(vgrid.centers()[:, 0])
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "restart.npz"
        _write_checkpoint(
            path,
            config=config,
            dt=dt,
            completed_steps=0,
            reference=reference,
            macro=macro,
            adaptive=adaptive,
            reference_history=[reference.moments()],
            macro_history=[macro.copy()],
            adaptive_history=[adaptive.moments.copy()],
            active_history=[adaptive.active.copy()],
            reference_diagnostics=[],
            macro_diagnostics=[],
            adaptive_diagnostics=[],
            reference_seconds=0.0,
            macro_seconds=0.0,
            adaptive_seconds=0.0,
        )
        restored = _restore_checkpoint(
            path, config=config, dt=dt, xgrid=xgrid, vgrid=vgrid
        )
    assert restored[0] == 0
    assert np.array_equal(restored[1].masses, reference.masses)
    assert np.array_equal(restored[2], macro)
    assert np.array_equal(restored[3].moments, adaptive.moments)
    assert np.array_equal(restored[3].micro_masses, adaptive.micro_masses)
    assert np.array_equal(restored[3].active, adaptive.active)

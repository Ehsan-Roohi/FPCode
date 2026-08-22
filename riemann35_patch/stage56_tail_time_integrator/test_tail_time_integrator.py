#!/usr/bin/env python3
"""Structural and regression tests for Stage 56."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import DynamicHighOrderState, HYQMOM_35_INDICES, realizability_margin_35
from riemann35_patch.stage54_heat_flux_third_order.run_heat_flux_method import (
    oblique_heat_flux_state,
)
from riemann35_patch.stage55_closure_source_audit.run_closure_method import exact_initial_tail
from riemann35_patch.stage56_tail_time_integrator.integrator import (
    exact_tail_relaxation,
    legacy_lie_projected_step,
    positive_tail_target,
    strang_exact_projected_step,
)


def initial_state() -> DynamicHighOrderState:
    source = oblique_heat_flux_state()
    return DynamicHighOrderState(
        moments=np.asarray(source["moments"], dtype=float),
        tail_moments=exact_initial_tail(tuple(source["components"])),
        maximum_order=6,
    )


def test_positive_target() -> None:
    state = initial_state()
    tail, minimum_weight, negative_mass, residual = positive_tail_target(state.moments)
    assert tail.shape == (49,)
    assert np.all(np.isfinite(tail))
    assert minimum_weight > 0.0
    assert negative_mass == 0.0
    assert np.isfinite(residual)


def test_exact_relaxation_semigroup() -> None:
    state = initial_state()
    full, full_diagnostics = exact_tail_relaxation(state, 0.0025, 0.01)
    first, _ = exact_tail_relaxation(state, 0.00125, 0.01)
    halves, half_diagnostics = exact_tail_relaxation(first, 0.00125, 0.01)
    np.testing.assert_allclose(halves.moments, full.moments, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(halves.tail_moments, full.tail_moments, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(
        half_diagnostics.relative_distance_after,
        full_diagnostics.relative_distance_after,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_projected_steps_preserve_invariants_and_positivity() -> None:
    state = initial_state()
    position = {index: slot for slot, index in enumerate(HYQMOM_35_INDICES)}
    invariant_positions = tuple(
        position[index]
        for index in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    )
    energy_positions = tuple(
        position[index] for index in ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    )
    for stepper in (legacy_lie_projected_step, strang_exact_projected_step):
        updated, diagnostics = stepper(state, 0.00125, 1.0)
        np.testing.assert_allclose(
            updated.moments[list(invariant_positions)],
            state.moments[list(invariant_positions)],
            rtol=0.0,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            np.sum(updated.moments[list(energy_positions)]),
            np.sum(state.moments[list(energy_positions)]),
            rtol=0.0,
            atol=2.0e-14,
        )
        assert diagnostics.minimum_weight > 0.0
        assert diagnostics.maximum_negative_mass_fraction == 0.0
        assert diagnostics.limiter_fraction > 0.0
        assert realizability_margin_35(updated.moments) >= -5.0e-13


def main() -> None:
    test_positive_target()
    test_exact_relaxation_semigroup()
    test_projected_steps_preserve_invariants_and_positivity()
    print("Stage 56 structural tests: PASS")


if __name__ == "__main__":
    main()

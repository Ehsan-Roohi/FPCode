#!/usr/bin/env python3
"""Dependency-light structural and mathematical tests for Stage 56."""

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
from riemann35_patch.stage55_closure_source_audit.run_closure_method import (  # noqa: E402
    exact_initial_tail,
)
from riemann35_patch.stage56_time_consistent_closure.time_consistent import (  # noqa: E402
    MOMENT_MATRIX_BASIS_3,
    PACKED_INDICES,
    exact_ou_degree_six_map,
    h3_margin,
    pack_degree_six,
    positive_projection_target,
    time_consistent_degree_six_step,
)


def initial_degree_six() -> np.ndarray:
    initial = oblique_heat_flux_state()
    return pack_degree_six(
        initial["moments"],
        exact_initial_tail(tuple(initial["components"])),
    )


def test_layout_and_h3() -> None:
    packed = initial_degree_six()
    assert len(PACKED_INDICES) == 84
    assert len(set(PACKED_INDICES)) == 84
    assert len(MOMENT_MATRIX_BASIS_3) == 20
    assert h3_margin(packed) > 0.0


def test_exact_ou_semigroup_and_invariants() -> None:
    packed = initial_degree_six()
    first = exact_ou_degree_six_map(packed, 0.91)
    second = exact_ou_degree_six_map(first, 0.83)
    direct = exact_ou_degree_six_map(packed, 0.91 * 0.83)
    np.testing.assert_allclose(second, direct, rtol=3.0e-13, atol=3.0e-13)
    invariant_positions = (0, 1, 5, 15)
    np.testing.assert_allclose(first[list(invariant_positions)], packed[list(invariant_positions)], atol=2.0e-14)
    energy_positions = (2, 9, 19)
    np.testing.assert_allclose(
        np.sum(first[list(energy_positions)]),
        np.sum(packed[list(energy_positions)]),
        atol=3.0e-14,
    )
    assert h3_margin(first) >= -2.0e-13


def test_projection_honors_node_count() -> None:
    packed = initial_degree_six()
    target4, diagnostics4 = positive_projection_target(packed, quadrature_nodes=4)
    target5, diagnostics5 = positive_projection_target(packed, quadrature_nodes=5)
    assert target4.shape == target5.shape == (49,)
    assert diagnostics4["quadrature_nodes"] == 4
    assert diagnostics5["quadrature_nodes"] == 5
    assert diagnostics4["node_count"] != diagnostics5["node_count"]
    assert min(diagnostics4["minimum_weight"], diagnostics5["minimum_weight"]) > 0.0


def test_one_time_consistent_step() -> None:
    packed = initial_degree_six()
    updated, diagnostics = time_consistent_degree_six_step(
        packed,
        dt=2.5e-4,
        tau=1.0,
        tail_relaxation_time=0.01,
        quadrature_nodes=5,
    )
    assert updated.shape == (84,)
    assert np.all(np.isfinite(updated))
    assert diagnostics.minimum_projection_weight > 0.0
    assert diagnostics.minimum_h3_margin >= -1.1e-12
    np.testing.assert_allclose(updated[0], packed[0], atol=2.0e-13)


def main() -> None:
    test_layout_and_h3()
    test_exact_ou_semigroup_and_invariants()
    test_projection_honors_node_count()
    test_one_time_consistent_step()
    print("Stage 56 structural and mathematical tests: PASS")


if __name__ == "__main__":
    main()

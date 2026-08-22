#!/usr/bin/env python3
"""Dependency-light structural tests for Stage 55."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    HYQMOM_35_INDICES,
    coefficients_from_moments,
    moments_35_from_qmc,
    projected_fp_collision_source,
    reconstruct_two_population_quadrature,
)
from hyqmom_fp.grad_hyqmom import WeightedNodeTailClosure  # noqa: E402
from riemann35_patch.stage54_heat_flux_third_order.run_heat_flux_method import (  # noqa: E402
    oblique_heat_flux_state,
)
from riemann35_patch.stage55_closure_source_audit.run_closure_method import (  # noqa: E402
    TAIL_INDICES,
    _direct_node_source,
    central_source_components,
    exact_initial_tail,
)


def test_tail_layout_and_exact_initialization() -> None:
    state = oblique_heat_flux_state()
    tail = exact_initial_tail(tuple(state["components"]))
    assert len(TAIL_INDICES) == 49
    assert sum(sum(index) == 5 for index in TAIL_INDICES) == 21
    assert sum(sum(index) == 6 for index in TAIL_INDICES) == 28
    assert tail.shape == (49,)
    assert np.all(np.isfinite(tail))


def test_positive_source_quadrature() -> None:
    moments = np.asarray(oblique_heat_flux_state()["moments"])
    quadrature = reconstruct_two_population_quadrature(
        moments,
        quadrature_nodes=4,
        minimum_skewness_norm=0.05,
        residual_correction=False,
    )
    assert np.min(quadrature.weights) > 0.0
    assert quadrature.negative_mass_fraction == 0.0
    node_moments = moments_35_from_qmc(quadrature.nodes, quadrature.weights)
    closure = WeightedNodeTailClosure(quadrature.nodes, quadrature.weights, maximum_order=6)
    coefficients = coefficients_from_moments(node_moments, tau=1.0, closure=closure)
    source = projected_fp_collision_source(node_moments, coefficients, closure=closure)
    direct = _direct_node_source(
        node_moments,
        quadrature.nodes,
        quadrature.weights,
        tau=1.0,
        prandtl=2.0 / 3.0,
    )
    np.testing.assert_allclose(direct, source, rtol=2.0e-11, atol=2.0e-11)
    central = central_source_components(moments, source)
    assert source.shape == (len(HYQMOM_35_INDICES),)
    assert central.shape == (10,)
    assert np.all(np.isfinite(central))


def main() -> None:
    test_tail_layout_and_exact_initialization()
    test_positive_source_quadrature()
    print("Stage 55 structural tests: PASS")


if __name__ == "__main__":
    main()

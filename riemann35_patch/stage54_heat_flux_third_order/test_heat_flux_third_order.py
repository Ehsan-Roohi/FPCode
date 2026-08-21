#!/usr/bin/env python3
"""Dependency-light structural tests for Stage 54."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import macroscopic_state
from riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux import (
    central_third_components,
    irreducible_decomposition,
    symmetric_tensor,
)
from riemann35_patch.stage54_heat_flux_third_order.run_heat_flux_method import (
    THIRD_INDICES,
    oblique_heat_flux_state,
)


def test_initial_state() -> None:
    state = oblique_heat_flux_state()
    assert state["mass_error"] < 1.0e-12
    assert state["momentum_norm"] < 1.0e-12
    assert state["energy_trace_error"] < 1.0e-12
    assert np.linalg.norm(state["heat_flux"]) > 0.10
    components = np.asarray(state["third_components"])
    tensor_norm = np.linalg.norm(symmetric_tensor(components))
    assert len(THIRD_INDICES) == 10
    assert np.all(np.abs(components) > 5.0e-3 * tensor_norm)


def test_irreducible_decomposition() -> None:
    state = oblique_heat_flux_state()
    moments = np.asarray(state["moments"])[None, :]
    components = central_third_components(moments)
    heat_flux, carrying, trace_free = irreducible_decomposition(components)
    np.testing.assert_allclose(heat_flux[0], macroscopic_state(moments[0]).heat_flux, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(
        np.einsum("...ijj->...i", trace_free),
        0.0,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        symmetric_tensor(components),
        carrying + trace_free,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def main() -> None:
    test_initial_state()
    test_irreducible_decomposition()
    print("Stage 54 structural tests: PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyqmom_fp import HYQMOM_35_INDICES, macroscopic_state
from hyqmom_fp.two_population import _gauss_hermite_mixture_nodes
from riemann35_patch.stage55_closure_source_audit.run_closure_method import _direct_node_source
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import (
    initialize_persistent_gaussian_mixture,
    persistent_gaussian_mixture_fp_step,
    persistent_gaussian_mixture_moments,
)
from riemann35_patch.stage58_blind_generalization.blind_cases import CASE_NAMES, blind_case

ORDERS = (2, 3, 4)


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-14))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=CASE_NAMES, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--prandtl', type=float, default=2.0/3.0)
    p.add_argument('--quadrature-nodes', type=int, default=5)
    p.add_argument('--dt', type=float, nargs='+', default=[6.25e-4, 3.125e-4, 1.5625e-4])
    a = p.parse_args()

    case = blind_case(a.case)
    state0 = initialize_persistent_gaussian_mixture(case.components)
    m0 = persistent_gaussian_mixture_moments(state0)
    weights, nodes = _gauss_hermite_mixture_nodes(
        state0.probabilities, state0.means, state0.covariances, state0.rho,
        a.quadrature_nodes,
    )
    exact = _direct_node_source(m0, nodes, weights, tau=a.tau, prandtl=a.prandtl)

    by_dt = []
    for dt in sorted(a.dt, reverse=True):
        state1, m1, diag = persistent_gaussian_mixture_fp_step(
            state0, dt, a.tau, prandtl=a.prandtl,
            quadrature_nodes=a.quadrature_nodes,
        )
        numerical = (m1 - m0) / dt
        order_errors = {}
        for order in ORDERS:
            idx = [i for i, alpha in enumerate(HYQMOM_35_INDICES) if sum(alpha) == order]
            order_errors[str(order)] = relerr(numerical[idx], exact[idx])
        by_dt.append({
            'dt_over_tau': dt/a.tau,
            'relative_source_error_all35': relerr(numerical, exact),
            'relative_source_error_by_order': order_errors,
            'heat_flux_projection_fraction': float(diag.heat_flux_projection_fraction),
            'heat_flux_projection_residual': float(diag.heat_flux_projection_residual),
        })

    finest = by_dt[-1]
    out = {
        'schema': 'riemann35-stage63-initial-source-audit-v1',
        'case': a.case,
        'role': case.role,
        'fingerprint': case.fingerprint,
        'quadrature_nodes_per_population': a.quadrature_nodes,
        'exact_source_norm': float(np.linalg.norm(exact)),
        'finest_dt_over_tau': finest['dt_over_tau'],
        'finest_relative_source_error_all35': finest['relative_source_error_all35'],
        'finest_relative_source_error_by_order': finest['relative_source_error_by_order'],
        'dt_refinement': by_dt,
        'interpretation': (
            'Compares the continuous cubic-FP generator at the exact initial four-Gaussian state '
            'with the derivative implied by one persistent Stage-57 finite-time step. '
            'A blind-only O(1) error that survives dt refinement identifies the source/map itself '
            'as the immediate generalization bottleneck.'
        ),
    }
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / f'stage63_{a.case}_summary.json'
    path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

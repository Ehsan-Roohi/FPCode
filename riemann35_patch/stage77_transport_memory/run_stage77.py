#!/usr/bin/env python3
"""Stage 77: isolate spatial transport-memory loss from collision physics.

The frozen Stage-76 left/right states and spatial CFL are reused.  Only one
free-transport step is taken; there is no collision.  Three paths are compared:

1. fine positive DVM upwind transport (independent kinetic reference),
2. a conservative 35-moment transport oracle built from the exact half-range
   Gaussian fluxes of the labelled populations, without recompression, and
3. the Stage-76 four-Gaussian candidate after transporting only population
   density/momentum/second moments and recompressing each population to a
   Gaussian.

If (2) agrees with (1) while (3) loses third-order/heat-flux information, the
Stage-76 failure is localized to Gaussian recompression during transport rather
than the promoted collision core.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyqmom_fp import HYQMOM_35_INDICES, SpatialGrid1D, dvm_upwind_transport_step
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import (
    initialize_persistent_gaussian_mixture,
)
from riemann35_patch.stage71_harder_unseen.hard_cases import hard_case
from riemann35_patch.stage76_spatial_kinetic_dvm.run_stage76 import (
    LEFT_CASE,
    RIGHT_CASE,
    candidate_moments,
    common_grid,
    component_tuples,
    derive,
    gaussian_nodes,
    initialize_spatial_dvm,
    low_errors,
    rel,
    transport_candidate,
)

EXPECTED_LEFT = "11a27e07eac086371a0df7c6fd85f5ba717e6ff73841005a44cb78fdc6798a31"
EXPECTED_RIGHT = "011d6ae4b1e1ca76eca468881c58f26a66f3988f02f974d5a6e4784dff0684d7"


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--spatial-cells", type=int, default=10)
    p.add_argument("--cfl", type=float, default=0.10)
    p.add_argument("--vcells", type=int, default=49)
    p.add_argument("--margin-sigma", type=float, default=7.0)
    p.add_argument("--flux-gh", type=int, default=11)
    return p.parse_args()


def jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def monomial_matrix(nodes: np.ndarray) -> np.ndarray:
    out = np.ones((nodes.shape[0], len(HYQMOM_35_INDICES)), dtype=float)
    for j, powers in enumerate(HYQMOM_35_INDICES):
        for axis, power in enumerate(powers):
            if power:
                out[:, j] *= nodes[:, axis] ** power
    return out


def half_flux_35(component, sign: int, order: int) -> np.ndarray:
    rho, mean, covariance = component
    nodes, probs = gaussian_nodes(np.asarray(mean), np.asarray(covariance), order)
    vx = nodes[:, 0]
    mask = vx > 0.0 if sign > 0 else vx < 0.0
    weighted = float(rho) * probs[mask] * vx[mask]
    return weighted @ monomial_matrix(nodes[mask])


def transport_oracle_35(states, dt: float, dx: float, left_state, right_state, order: int):
    initial = candidate_moments(states)
    nx = len(states)
    left_components = component_tuples(left_state)
    right_components = component_tuples(right_state)
    cells = [component_tuples(s) for s in states]
    face = np.zeros((nx + 1, len(HYQMOM_35_INDICES)), dtype=float)

    face[0] = sum((half_flux_35(c, +1, order) for c in left_components), np.zeros(len(HYQMOM_35_INDICES)))
    face[0] += sum((half_flux_35(c, -1, order) for c in cells[0]), np.zeros(len(HYQMOM_35_INDICES)))
    for f in range(1, nx):
        face[f] = sum((half_flux_35(c, +1, order) for c in cells[f - 1]), np.zeros(len(HYQMOM_35_INDICES)))
        face[f] += sum((half_flux_35(c, -1, order) for c in cells[f]), np.zeros(len(HYQMOM_35_INDICES)))
    face[-1] = sum((half_flux_35(c, +1, order) for c in cells[-1]), np.zeros(len(HYQMOM_35_INDICES)))
    face[-1] += sum((half_flux_35(c, -1, order) for c in right_components), np.zeros(len(HYQMOM_35_INDICES)))

    updated = initial - (dt / dx) * (face[1:] - face[:-1])
    residual = dx * np.sum(updated - initial, axis=0) + dt * (face[-1] - face[0])
    scale = max(dx * float(np.sum(np.abs(initial))), 1.0e-30)
    return updated, float(np.linalg.norm(residual) / scale)


def tensor_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    cq, ct, ctf = derive(candidate)
    rq, rt, rtf = derive(reference)
    return {
        "full_third": rel(ct, rt),
        "tracefree": rel(ctf, rtf),
        "heat_flux": rel(cq, rq),
    }


def vector_alignment(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.ravel(np.asarray(a, float))
    bb = np.ravel(np.asarray(b, float))
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / den) if den > 0.0 else 1.0


def main() -> None:
    a = arguments()
    a.output.mkdir(parents=True, exist_ok=True)

    left = hard_case(LEFT_CASE)
    right = hard_case(RIGHT_CASE)
    if left.fingerprint != EXPECTED_LEFT or right.fingerprint != EXPECTED_RIGHT:
        raise RuntimeError("Stage-77 frozen case fingerprint changed")

    xgrid = SpatialGrid1D(-1.0, 1.0, a.spatial_cells)
    vgrid = common_grid(left.components, right.components, a.vcells, a.margin_sigma)
    vmax = float(np.max(np.abs(vgrid.centers()[:, 0])))
    dt = a.cfl * xgrid.width / vmax

    dvm, dvm_left, dvm_right, lp, rp = initialize_spatial_dvm(
        xgrid, vgrid, left.components, right.components
    )
    dvm_initial = dvm.moments()
    dvm_after, dvm_diag = dvm_upwind_transport_step(dvm, dt, dvm_left, dvm_right)
    dvm_moments = dvm_after.moments()

    left_state = initialize_persistent_gaussian_mixture(left.components)
    right_state = initialize_persistent_gaussian_mixture(right.components)
    states = [left_state if x < 0.0 else right_state for x in xgrid.centers]
    candidate_initial = candidate_moments(states)

    oracle, oracle_balance = transport_oracle_35(
        states, dt, xgrid.width, left_state, right_state, a.flux_gh
    )
    recompressed_states, candidate_balance = transport_candidate(
        states, dt, xgrid.width, left_state, right_state, a.flux_gh
    )
    recompressed = candidate_moments(recompressed_states)

    initial_error = rel(candidate_initial, dvm_initial)
    oracle_vs_dvm = tensor_metrics(oracle, dvm_moments)
    recompressed_vs_dvm = tensor_metrics(recompressed, dvm_moments)
    recompression_defect = tensor_metrics(recompressed, oracle)

    od_rho, od_mom, od_energy = low_errors(oracle, dvm_moments)
    rd_rho, rd_mom, rd_energy = low_errors(recompressed, dvm_moments)
    ro_rho, ro_mom, ro_energy = low_errors(recompressed, oracle)

    oq, ot, _ = derive(oracle)
    dq, dtensor, _ = derive(dvm_moments)
    rq, rtensor, _ = derive(recompressed)
    q_alignment = vector_alignment(rq - oq, rq - dq)
    third_alignment = vector_alignment(rtensor - ot, rtensor - dtensor)
    q_magnitude_ratio = float(
        np.linalg.norm(rq - oq) / max(np.linalg.norm(rq - dq), 1.0e-30)
    )
    third_magnitude_ratio = float(
        np.linalg.norm(rtensor - ot) / max(np.linalg.norm(rtensor - dtensor), 1.0e-30)
    )

    interface = np.argsort(np.abs(xgrid.centers))[:2]
    local = {
        "interface_cells": interface,
        "x": xgrid.centers[interface],
        "dvm_heat_flux": dq[interface],
        "oracle_heat_flux": oq[interface],
        "recompressed_heat_flux": rq[interface],
    }

    diagnosis_gates = {
        "initial_representation_matches_dvm": initial_error < 5.0e-8,
        "oracle_transport_conservative": oracle_balance < 1.0e-10,
        "candidate_low_transport_conservative": candidate_balance < 1.0e-10,
        "oracle_vs_dvm_full_third": oracle_vs_dvm["full_third"] < 0.01,
        "oracle_vs_dvm_heat_flux": oracle_vs_dvm["heat_flux"] < 0.01,
        "recompression_preserves_density": ro_rho < 1.0e-10,
        "recompression_preserves_momentum": ro_mom < 1.0e-10,
        "recompression_preserves_energy": ro_energy < 1.0e-10,
        "recompression_creates_third_defect": recompression_defect["full_third"] > 0.02,
        "recompression_creates_heat_flux_defect": recompression_defect["heat_flux"] > 0.02,
        "third_error_vector_aligned_with_recompression": third_alignment > 0.90,
        "heat_flux_error_vector_aligned_with_recompression": q_alignment > 0.90,
    }
    confirmed = all(diagnosis_gates.values())
    diagnosis = (
        "GAUSSIAN_RECOMPRESSION_THIRD_MEMORY_LOSS_CONFIRMED"
        if confirmed
        else "TRANSPORT_MEMORY_DIAGNOSIS_INCONCLUSIVE"
    )

    summary = {
        "schema": "riemann35-stage77-transport-memory-audit-v1",
        "diagnosis": diagnosis,
        "scope": "one-step collision-free transport audit on frozen Stage-76 heterogeneous interface",
        "controls": {
            "qmc_used": False,
            "collision_used": False,
            "left_case": LEFT_CASE,
            "left_fingerprint": left.fingerprint,
            "right_case": RIGHT_CASE,
            "right_fingerprint": right.fingerprint,
            "spatial_cells": a.spatial_cells,
            "velocity_shape": list(vgrid.shape),
            "velocity_lower": list(vgrid.lower),
            "velocity_upper": list(vgrid.upper),
            "cfl": a.cfl,
            "dt": dt,
            "flux_gauss_hermite_order": a.flux_gh,
            "diagnosis_thresholds_frozen_before_run": True,
        },
        "dvm_initial_projection": {
            "left": None if lp is None else lp.relative_moment_residual,
            "right": None if rp is None else rp.relative_moment_residual,
        },
        "dvm_transport": {
            "minimum_mass": dvm_diag.minimum_mass,
            "mass_balance_residual": dvm_diag.mass_balance_residual,
            "momentum_balance_residual": dvm_diag.momentum_balance_residual,
            "energy_balance_residual": dvm_diag.energy_balance_residual,
        },
        "metrics": {
            "initial_candidate_vs_dvm_all35": initial_error,
            "oracle_balance": oracle_balance,
            "candidate_low_balance": candidate_balance,
            "oracle_vs_dvm": {
                **oracle_vs_dvm,
                "density": od_rho,
                "momentum": od_mom,
                "energy": od_energy,
            },
            "recompressed_vs_dvm": {
                **recompressed_vs_dvm,
                "density": rd_rho,
                "momentum": rd_mom,
                "energy": rd_energy,
            },
            "recompression_defect_vs_uncompressed_oracle": {
                **recompression_defect,
                "density": ro_rho,
                "momentum": ro_mom,
                "energy": ro_energy,
            },
            "error_vector_attribution": {
                "third_alignment": third_alignment,
                "heat_flux_alignment": q_alignment,
                "third_recompression_to_total_error_magnitude_ratio": third_magnitude_ratio,
                "heat_flux_recompression_to_total_error_magnitude_ratio": q_magnitude_ratio,
            },
        },
        "interface_diagnostics": local,
        "diagnosis_gates": diagnosis_gates,
        "diagnosis_confirmed": confirmed,
    }

    np.savez_compressed(
        a.output / "stage77_transport_memory.npz",
        x=xgrid.centers,
        dt=np.asarray(dt),
        dvm_initial=dvm_initial,
        candidate_initial=candidate_initial,
        dvm_after=dvm_moments,
        oracle_after=oracle,
        recompressed_after=recompressed,
    )
    (a.output / "stage77_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage 77 — transport-memory audit",
        "",
        f"Diagnosis: **{diagnosis}**",
        "",
        "One collision-free spatial transport step; no QMC and no collision operator are used.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| oracle vs DVM full-third | {100*oracle_vs_dvm['full_third']:.4f}% |",
        f"| oracle vs DVM heat flux | {100*oracle_vs_dvm['heat_flux']:.4f}% |",
        f"| recompressed vs DVM full-third | {100*recompressed_vs_dvm['full_third']:.4f}% |",
        f"| recompressed vs DVM heat flux | {100*recompressed_vs_dvm['heat_flux']:.4f}% |",
        f"| recompression-only full-third defect | {100*recompression_defect['full_third']:.4f}% |",
        f"| recompression-only heat-flux defect | {100*recompression_defect['heat_flux']:.4f}% |",
        f"| recompression density defect | {100*ro_rho:.6e}% |",
        f"| recompression momentum defect | {100*ro_mom:.6e}% |",
        f"| recompression energy defect | {100*ro_energy:.6e}% |",
        f"| third error-vector alignment | {third_alignment:.6f} |",
        f"| heat-flux error-vector alignment | {q_alignment:.6f} |",
        f"| oracle transport balance | {oracle_balance:.3e} |",
        f"| candidate low-order balance | {candidate_balance:.3e} |",
    ]
    (a.output / "STAGE77_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"STAGE77_DIAGNOSIS={diagnosis}", flush=True)


if __name__ == "__main__":
    main()

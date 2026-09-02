#!/usr/bin/env python3
"""Stage 76: spatial kinetic-flux validation of the promoted persistent closure.

Reference: positive full-DVM upwind transport + independent cubic-FP DVM collision.
Candidate: four labelled Gaussian populations transported conservatively with
half-range kinetic Gaussian fluxes for each population's density/momentum/
second moments, compressed back to full-covariance Gaussians, then advanced by
the promoted persistent collision closure.

The left state is the frozen Stage-71 dense_hot_extreme case (rho=2.2) and the
right state is the frozen dilute_broad case (rho=0.42).  This deliberately
stresses the density-consistent projector in a spatially heterogeneous setting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numpy.polynomial.hermite import hermgauss

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyqmom_fp import DVMGrid, SpatialDVMState, SpatialGrid1D, full_dvm_shock_step
from hyqmom_fp.dvm_reference import initialize_diagonal_gaussian_mixture
from riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux import (
    central_third_components,
    irreducible_decomposition,
    relative_history_error,
    symmetric_tensor,
)
from riemann35_patch.stage57_persistent_four_population.persistent_mixture import (
    PersistentGaussianMixtureState,
    initialize_persistent_gaussian_mixture,
    persistent_gaussian_mixture_fp_step,
    persistent_gaussian_mixture_moments,
)
from riemann35_patch.stage71_harder_unseen.hard_cases import hard_case

LEFT_CASE = "dense_hot_extreme"
RIGHT_CASE = "dilute_broad"
PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--spatial-cells", type=int, default=10)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--cfl", type=float, default=0.10)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    p.add_argument("--coarse-vcells", type=int, default=19)
    p.add_argument("--fine-vcells", type=int, default=25)
    p.add_argument("--margin-sigma", type=float, default=7.0)
    p.add_argument("--flux-gh", type=int, default=11)
    return p.parse_args()


def jsonable(v):
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, dict): return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jsonable(x) for x in v]
    return v


def common_grid(left_components, right_components, cells: int, margin_sigma: float) -> DVMGrid:
    comps = tuple(left_components) + tuple(right_components)
    means = np.asarray([c[1] for c in comps], float)
    covs = np.asarray([c[2] for c in comps], float)
    diagonal = np.asarray([np.diag(np.diag(c)) for c in covs])
    if not np.allclose(covs, diagonal, atol=1e-12):
        raise ValueError("Stage76 DVM initializer requires diagonal boundary covariances")
    sig = np.sqrt(np.diagonal(covs, axis1=1, axis2=2))
    lo = np.min(means - margin_sigma * sig, axis=0)
    hi = np.max(means + margin_sigma * sig, axis=0)
    pad = 0.02 * np.maximum(hi - lo, 1.0)
    return DVMGrid(tuple(lo - pad), tuple(hi + pad), (cells, cells, cells))


def initialize_spatial_dvm(xgrid, vgrid, left_components, right_components):
    left, lp = initialize_diagonal_gaussian_mixture(vgrid, left_components, match_exact_moments=True)
    right, rp = initialize_diagonal_gaussian_mixture(vgrid, right_components, match_exact_moments=True)
    mask = xgrid.centers < 0.0
    masses = np.where(mask[:, None], left.masses[None, :], right.masses[None, :])
    return SpatialDVMState(xgrid, vgrid, masses), left, right, lp, rp


def component_low_vector(weight: float, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    out = np.empty(10)
    out[0] = weight
    out[1:4] = weight * mean
    second = covariance + np.outer(mean, mean)
    out[4:] = [weight * second[i, j] for i, j in PAIRS]
    return out


def component_from_low(u: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    rho = float(u[0])
    if not np.isfinite(rho) or rho <= 1e-14:
        raise FloatingPointError(f"nonpositive transported component density {rho}")
    mean = np.asarray(u[1:4]) / rho
    second = np.zeros((3, 3))
    for value, (i, j) in zip(u[4:] / rho, PAIRS):
        second[i, j] = second[j, i] = value
    cov = 0.5 * ((second - np.outer(mean, mean)) + (second - np.outer(mean, mean)).T)
    mineig = float(np.min(np.linalg.eigvalsh(cov)))
    if mineig <= 1e-12:
        cov += (1e-12 - mineig + 1e-13) * np.eye(3)
    return rho, mean, cov


def state_low(state: PersistentGaussianMixtureState) -> np.ndarray:
    return np.asarray([
        component_low_vector(state.rho * float(p), m, c)
        for p, m, c in zip(state.probabilities, state.means, state.covariances)
    ])


def state_from_low(values: np.ndarray) -> PersistentGaussianMixtureState:
    components = tuple(component_from_low(values[k]) for k in range(values.shape[0]))
    return initialize_persistent_gaussian_mixture(components)


def gaussian_nodes(mean: np.ndarray, cov: np.ndarray, order: int):
    x, w = hermgauss(order)
    mesh = np.meshgrid(x, x, x, indexing="ij")
    z = np.column_stack([m.ravel() for m in mesh]) * np.sqrt(2.0)
    wm = np.meshgrid(w, w, w, indexing="ij")
    weights = (wm[0] * wm[1] * wm[2]).ravel() / (np.pi ** 1.5)
    L = np.linalg.cholesky(cov)
    nodes = mean[None, :] + z @ L.T
    return nodes, weights


def half_flux_component(component, sign: int, order: int) -> np.ndarray:
    rho, mean, cov = component
    nodes, probs = gaussian_nodes(np.asarray(mean), np.asarray(cov), order)
    vx = nodes[:, 0]
    mask = vx > 0.0 if sign > 0 else vx < 0.0
    q = rho * probs[mask] * vx[mask]
    v = nodes[mask]
    out = np.empty(10)
    out[0] = np.sum(q)
    out[1:4] = q @ v
    out[4:] = [np.sum(q * v[:, i] * v[:, j]) for i, j in PAIRS]
    return out


def component_tuples(state: PersistentGaussianMixtureState):
    return tuple(
        (state.rho * float(p), np.asarray(m), np.asarray(c))
        for p, m, c in zip(state.probabilities, state.means, state.covariances)
    )


def transport_candidate(states, dt: float, dx: float, left_state, right_state, flux_order: int):
    nx = len(states); ncomp = states[0].probabilities.size
    lows = np.asarray([state_low(s) for s in states])
    face = np.zeros((nx + 1, ncomp, 10))
    left_components = component_tuples(left_state); right_components = component_tuples(right_state)
    all_components = [component_tuples(s) for s in states]
    for k in range(ncomp):
        face[0, k] = half_flux_component(left_components[k], +1, flux_order) + half_flux_component(all_components[0][k], -1, flux_order)
        for f in range(1, nx):
            face[f, k] = half_flux_component(all_components[f-1][k], +1, flux_order) + half_flux_component(all_components[f][k], -1, flux_order)
        face[-1, k] = half_flux_component(all_components[-1][k], +1, flux_order) + half_flux_component(right_components[k], -1, flux_order)
    updated = lows - (dt / dx) * (face[1:] - face[:-1])
    new_states = [state_from_low(updated[i]) for i in range(nx)]
    # conservative transport balance in the component low variables
    residual = dx * np.sum(updated - lows, axis=(0, 1)) + dt * np.sum(face[-1] - face[0], axis=0)
    scale = max(dx * float(np.sum(np.abs(lows))), 1e-30)
    return new_states, float(np.linalg.norm(residual) / scale)


def collide_candidate(states, dt, tau, prandtl):
    out=[]; minfrac=1.0; maxres=0.0; mineig=float("inf")
    for s in states:
        ns, _, d = persistent_gaussian_mixture_fp_step(s, dt, tau, prandtl=prandtl, quadrature_nodes=5, enforce_heat_flux_rate=True)
        out.append(ns)
        minfrac=min(minfrac, d.heat_flux_projection_fraction)
        maxres=max(maxres, d.heat_flux_projection_residual)
        mineig=min(mineig, d.minimum_covariance_eigenvalue)
    return out, minfrac, maxres, mineig


def candidate_moments(states):
    return np.asarray([persistent_gaussian_mixture_moments(s) for s in states])


def run_reference(xgrid, vgrid, left_components, right_components, dt, steps, tau, prandtl):
    state, left, right, lp, rp = initialize_spatial_dvm(xgrid, vgrid, left_components, right_components)
    hist=[state.moments()]; maxproj=0.0; maxcoll=0.0; maxbal=0.0; minmass=float(np.min(state.masses)); start=time.perf_counter()
    for step in range(1, steps+1):
        state, d = full_dvm_shock_step(state, dt, tau, left, right, prandtl=prandtl, guided=True)
        hist.append(state.moments())
        maxproj=max(maxproj, d.maximum_projection_residual); maxcoll=max(maxcoll,d.maximum_collision_invariant_drift)
        maxbal=max(maxbal,d.transport.mass_balance_residual,d.transport.momentum_balance_residual,d.transport.energy_balance_residual)
        minmass=min(minmass,float(np.min(state.masses)))
        print(f"[stage76] DVM n={vgrid.shape[0]} step={step}/{steps}", flush=True)
    return np.asarray(hist), {
        "grid": {"lower":list(vgrid.lower),"upper":list(vgrid.upper),"shape":list(vgrid.shape)},
        "initial_left_projection": lp.relative_moment_residual if lp else None,
        "initial_right_projection": rp.relative_moment_residual if rp else None,
        "maximum_projection_residual":maxproj,
        "maximum_collision_invariant_drift":maxcoll,
        "maximum_transport_balance_residual":maxbal,
        "minimum_mass":minmass,
        "elapsed_seconds":time.perf_counter()-start,
    }


def derive(hist):
    flat=hist.reshape(-1,35)
    c=central_third_components(flat)
    q,_,tf=irreducible_decomposition(c)
    return q.reshape(hist.shape[:-1]+(3,)), symmetric_tensor(c).reshape(hist.shape[:-1]+(3,3,3)), tf.reshape(hist.shape[:-1]+(3,3,3))


def rel(a,b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))


def low_errors(candidate, reference):
    rho=rel(candidate[...,0],reference[...,0])
    mom=rel(candidate[...,1:4],reference[...,1:4])
    # raw trace indices for repository order: use macroscopic-safe explicit lookup
    from hyqmom_fp import HYQMOM_35_INDICES
    pos={idx:i for i,idx in enumerate(HYQMOM_35_INDICES)}
    energy=lambda h: h[...,pos[(2,0,0)]]+h[...,pos[(0,2,0)]]+h[...,pos[(0,0,2)]]
    return rho,mom,rel(energy(candidate),energy(reference))


def main():
    a=args(); a.output.mkdir(parents=True,exist_ok=True)
    left=hard_case(LEFT_CASE); right=hard_case(RIGHT_CASE)
    if left.fingerprint != "11a27e07eac086371a0df7c6fd85f5ba717e6ff73841005a44cb78fdc6798a31": raise RuntimeError("left fingerprint changed")
    if right.fingerprint != "011d6ae4b1e1ca76eca468881c58f26a66f3988f02f974d5a6e4784dff0684d7": raise RuntimeError("right fingerprint changed")
    xgrid=SpatialGrid1D(-1.0,1.0,a.spatial_cells)
    coarse_grid=common_grid(left.components,right.components,a.coarse_vcells,a.margin_sigma)
    fine_grid=common_grid(left.components,right.components,a.fine_vcells,a.margin_sigma)
    vmax=max(np.max(np.abs(coarse_grid.centers()[:,0])),np.max(np.abs(fine_grid.centers()[:,0])))
    dt=a.cfl*xgrid.width/vmax
    coarse,cd=run_reference(xgrid,coarse_grid,left.components,right.components,dt,a.steps,a.tau,a.prandtl)
    fine,fd=run_reference(xgrid,fine_grid,left.components,right.components,dt,a.steps,a.tau,a.prandtl)

    left_state=initialize_persistent_gaussian_mixture(left.components); right_state=initialize_persistent_gaussian_mixture(right.components)
    states=[left_state if x<0 else right_state for x in xgrid.centers]
    ch=[candidate_moments(states)]; maxbal=0.0; minfrac=1.0; maxproj=0.0; mineig=float("inf"); start=time.perf_counter()
    for step in range(1,a.steps+1):
        states,b=transport_candidate(states,dt,xgrid.width,left_state,right_state,a.flux_gh)
        states,pf,pr,me=collide_candidate(states,dt,a.tau,a.prandtl)
        maxbal=max(maxbal,b); minfrac=min(minfrac,pf); maxproj=max(maxproj,pr); mineig=min(mineig,me)
        ch.append(candidate_moments(states))
        print(f"[stage76] candidate step={step}/{a.steps} balance={b:.3e} projection={pf:.6f}",flush=True)
    cand=np.asarray(ch)

    cq,ct,ctf=derive(cand); fq,ft,ftf=derive(fine); qq,qt,qtf=derive(coarse)
    ref_third=relative_history_error(qt,ft); ref_tf=relative_history_error(qtf,ftf); ref_q=relative_history_error(qq,fq)
    cand_third=relative_history_error(ct,ft); cand_tf=relative_history_error(ctf,ftf); cand_q=relative_history_error(cq,fq)
    rhoerr,momerr,enerr=low_errors(cand,fine)
    gates={
      "dvm_refinement_third":ref_third<0.015,
      "dvm_refinement_tracefree":ref_tf<0.02,
      "candidate_density":rhoerr<0.02,
      "candidate_momentum":momerr<0.03,
      "candidate_energy":enerr<0.02,
      "candidate_third":cand_third<0.03,
      "candidate_tracefree":cand_tf<0.05,
      "candidate_heat_flux":cand_q<0.03,
      "candidate_transport_balance":maxbal<1e-9,
      "candidate_full_projection":minfrac>=1.0-2e-13,
      "candidate_projection_residual":maxproj<1e-8,
      "candidate_positive_covariance":mineig>0.0,
      "dvm_positive":min(cd["minimum_mass"],fd["minimum_mass"])>0.0,
      "dvm_projection":max(cd["maximum_projection_residual"],fd["maximum_projection_residual"])<2e-8,
    }
    passed=all(gates.values())
    summary={
      "schema":"riemann35-stage76-spatial-kinetic-dvm-v1",
      "qualification_pass":passed,
      "scope":"spatial heterogeneous four-population transport/compression/persistent-collision candidate vs full positive DVM transport+collision reference",
      "left_case":LEFT_CASE,"left_fingerprint":left.fingerprint,"right_case":RIGHT_CASE,"right_fingerprint":right.fingerprint,
      "controls":{"qmc_used":False,"prescribed_common_carrier":False,"spatial_cells":a.spatial_cells,"steps":a.steps,"dt":dt,"cfl":a.cfl,"flux_gauss_hermite_order":a.flux_gh,"thresholds_frozen_before_run":True,"closure_parameter_refit":False},
      "dvm_coarse":cd,"dvm_fine":fd,
      "metrics":{"dvm_refinement_third":ref_third,"dvm_refinement_tracefree":ref_tf,"dvm_refinement_heat_flux":ref_q,"candidate_density":rhoerr,"candidate_momentum":momerr,"candidate_energy":enerr,"candidate_third":cand_third,"candidate_tracefree":cand_tf,"candidate_heat_flux":cand_q,"candidate_transport_balance":maxbal,"candidate_min_projection_fraction":minfrac,"candidate_max_projection_residual":maxproj,"candidate_min_covariance_eigenvalue":mineig,"candidate_elapsed_seconds":time.perf_counter()-start},
      "gates":gates,
    }
    np.savez_compressed(a.output/"stage76_histories.npz",x=xgrid.centers,dt=np.asarray(dt),dvm_coarse=coarse,dvm_fine=fine,candidate=cand)
    (a.output/"stage76_summary.json").write_text(json.dumps(jsonable(summary),indent=2)+"\n")
    m=summary["metrics"]
    lines=["# Stage 76 — spatial kinetic/DVM gate","",f"Qualification objective: **{'PASS' if passed else 'FAIL'}**","","No QMC reference and no prescribed common-carrier advection are used.","", "| metric | value | gate |","|---|---:|---:|",f"| DVM refine full-third | {100*ref_third:.3f}% | <1.5% |",f"| candidate density | {100*rhoerr:.3f}% | <2% |",f"| candidate momentum | {100*momerr:.3f}% | <3% |",f"| candidate energy | {100*enerr:.3f}% | <2% |",f"| candidate full-third | {100*cand_third:.3f}% | <3% |",f"| candidate trace-free | {100*cand_tf:.3f}% | <5% |",f"| candidate heat flux | {100*cand_q:.3f}% | <3% |",f"| transport balance | {maxbal:.3e} | <1e-9 |",f"| min projection fraction | {minfrac:.6f} | 1 |",f"| max projection residual | {maxproj:.3e} | <1e-8 |"]
    (a.output/"STAGE76_RESULTS.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    if not passed: print("STAGE76_QUALIFICATION=FAIL",flush=True)
    else: print("STAGE76_QUALIFICATION=PASS",flush=True)

if __name__=="__main__": main()

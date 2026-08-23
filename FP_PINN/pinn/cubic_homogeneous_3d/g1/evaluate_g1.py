#!/usr/bin/env python3
"""Deterministic evaluation and gate for the heat-flux G1 stage.

Every number produced here is a deterministic quadrature of the model density.
Nothing is sampled, so two evaluations of the same weights give the same
metrics to round-off, and the quadrature error is *measured* by evaluating on
two different grids (the training grid and a finer, wider held-out grid).
The audit of the G0 evaluator showed that its 131072/524288-sample Monte-Carlo
protocol carried a one-sigma uncertainty of 3.6-4.5 percentage points on the
analytic-Qx L2 error, which is larger than every gate it was asked to decide.

Usage
-----
    python evaluate_g1.py --config config.json --weights epoch-010000.weights.h5 \
        --reference reference_fv/reference.npz --output eval/epoch-010000
    python evaluate_g1.py --config config.json --sweep-dir checkpoints_h5 \
        --reference reference_fv/reference.npz --output sweep

The reference is the deterministic axisymmetric finite-volume solution of the
same cubic Fokker--Planck equation (axisym_fp_reference.py); the primary gate
still uses the closed-form Qx(t) = Qx(0) exp(-4/3 nu t) of heat_flux_g0.py.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for path in (str(PARENT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from heat_flux_g0 import (  # noqa: E402
    HEAT_FLUX_DECAY_RATE_FACTOR,
    HEAT_FLUX_PRIMARY_L2,
    HEAT_FLUX_PUBLICATION_L2,
    analytic_heat_flux_history,
    effective_prandtl_number,
    fit_decay_rate,
    heat_flux_gate_summary,
    relative_l2,
)
from train_stage2 import Config as BaseConfig, tf_initial_logpdf  # noqa: E402
from axisym_quadrature import build_quadrature, trapezoid_nodes, gauss_legendre_nodes  # noqa: E402
from structure_model import (  # noqa: E402
    G1Config,
    StructuredDensityModel,
    assemble_slices,
    axisym_moment_tensors_tf,
    log_residual,
    quadrature_tensors,
)


# --------------------------------------------------------------------------
# Gates (frozen for G1; equality passes)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class G1Gates:
    qx_analytic_l2_primary: float = HEAT_FLUX_PRIMARY_L2          # 0.02
    qx_analytic_l2_publication: float = HEAT_FLUX_PUBLICATION_L2  # 0.01
    qx_quadrature_uncertainty_pp: float = 0.5    # |err(fine) - err(train)| in percentage points
    mass_drift: float = 0.005                     # tilted model (structural)
    momentum_drift: float = 0.001
    energy_drift: float = 0.005
    raw_mass_drift: float = 0.05                  # pre-tilt network density (diagnostic)
    raw_energy_drift: float = 0.05
    marginal_relative_l2: float = 0.03            # vs deterministic FV reference
    stress_anisotropy_abs: float = 0.02           # max |(pxx - pyy) - (pxx - pyy)_ref|
    transverse_heat_flux_relative: float = 1.0e-6
    initial_condition_linf: float = 2.0e-6
    portable_reload_linf: float = 1.0e-7
    decay_rate_relative: float = 0.05             # |rate - 4/3| / (4/3)


GATES = G1Gates()


# --------------------------------------------------------------------------
# Model construction / loading
# --------------------------------------------------------------------------
def split_config(config: dict[str, Any]) -> tuple[BaseConfig, G1Config]:
    """Rebuild (BaseConfig, G1Config) from a G1 config.json dictionary."""
    base_fields = BaseConfig.__dataclass_fields__
    g1_fields = G1Config.__dataclass_fields__
    base_kwargs = {k: v for k, v in config.items() if k in base_fields}
    for required in ("case", "output_dir", "reference"):
        base_kwargs.setdefault(required, config.get(required, "heat_flux" if required == "case" else ""))
    base_kwargs["case"] = "heat_flux"
    base = BaseConfig(**base_kwargs)
    g1 = G1Config(**{k: v for k, v in config.items() if k in g1_fields})
    return base, g1


def build_model(config: dict[str, Any], weights: str | Path | None = None) -> StructuredDensityModel:
    base, g1 = split_config(config)
    model = StructuredDensityModel(base, g1)
    model.build_model()
    if weights is not None:
        model.load_weights(str(weights))
    return model


# --------------------------------------------------------------------------
# Deterministic moments without derivative tapes (fast path)
# --------------------------------------------------------------------------
def tilted_slices(model: StructuredDensityModel, times: np.ndarray, quad) -> dict[str, np.ndarray]:
    """Raw and tilted moments, beta and log f at the quadrature nodes for each time."""
    nodes32 = quad.nodes32
    nv = int(nodes32.shape[0])
    out: dict[str, list] = {k: [] for k in ("beta", "raw", "tilted", "log_f", "min_log_f_raw")}
    for chunk in np.array_split(np.asarray(times, dtype=np.float64), max(1, len(times) // 8)):
        nt = len(chunk)
        flat_t = tf.reshape(tf.repeat(tf.constant(chunk, tf.float32)[:, None], nv, axis=1), [-1, 1])
        flat_c = tf.reshape(tf.tile(nodes32[None, :, :], [nt, 1, 1]), [-1, 3])
        log_raw = tf.cast(tf.reshape(model.raw_log_density(flat_t, flat_c), [nt, nv]), tf.float64)
        beta, _, log_f = model.solve_tilt(log_raw, quad.psi64, quad.weights64)
        wf = quad.weights64[None, :] * tf.exp(log_f)
        raw_wf = quad.weights64[None, :] * tf.exp(log_raw)
        out["beta"].append(beta.numpy())
        out["raw"].append({k: v.numpy() for k, v in axisym_moment_tensors_tf(quad.cx64, quad.rho64, raw_wf).items()})
        out["tilted"].append({k: v.numpy() for k, v in axisym_moment_tensors_tf(quad.cx64, quad.rho64, wf).items()})
        out["log_f"].append(log_f.numpy())
        out["min_log_f_raw"].append(np.min(log_raw.numpy(), axis=1))
    result = {
        "beta": np.concatenate(out["beta"], axis=0),
        "log_f": np.concatenate(out["log_f"], axis=0),
        "min_log_f_raw": np.concatenate(out["min_log_f_raw"], axis=0),
    }
    for name in ("raw", "tilted"):
        merged: dict[str, np.ndarray] = {}
        for key in out[name][0]:
            merged[key] = np.concatenate([d[key] for d in out[name]], axis=0)
        result[name] = merged
    return result


def residual_diagnostics(model: StructuredDensityModel, times: np.ndarray, quad, nu: float) -> np.ndarray:
    """f-weighted RMS of the log residual per time slice (with derivative tapes)."""
    rms = []
    for chunk in np.array_split(np.asarray(times, dtype=np.float64), max(1, len(times) // 3)):
        state = assemble_slices(model, tf.constant(chunk, tf.float32), quad)
        residual = log_residual(state, quad, nu).numpy().astype(np.float64)
        wf = state.wf.numpy()
        rms.append(np.sqrt(np.sum(wf * residual**2, axis=1) / np.sum(wf, axis=1)))
    return np.concatenate(rms)


PREDICT_BATCH = 131_072


def portable_reload_check(
    model: StructuredDensityModel, config: dict[str, Any], weights: str | Path,
) -> float:
    """Compare an in-memory model with a clean model loaded from ``weights``."""
    rng = np.random.default_rng(int(config.get("seed", 0)) + 1)
    points = np.stack([
        rng.normal(size=2048) * 1.5,
        np.abs(rng.normal(size=2048)) * 1.5,
        np.zeros(2048),
    ], axis=1)
    times = tf.fill([points.shape[0], 1], 0.731)
    points_tf = tf.constant(points, tf.float32)
    before = model.raw_log_density(times, points_tf).numpy()
    reloaded = build_model(config, weights)
    after = reloaded.raw_log_density(times, points_tf).numpy()
    return float(np.max(np.abs(before - after)))


def raw_log_density_batched(model: StructuredDensityModel, t: float, c: np.ndarray) -> np.ndarray:
    """model.raw_log_density evaluated in batches (bounded memory)."""
    pieces = []
    for start in range(0, c.shape[0], PREDICT_BATCH):
        block = tf.constant(c[start:start + PREDICT_BATCH], tf.float32)
        pieces.append(model.raw_log_density(tf.fill([block.shape[0], 1], float(t)), block).numpy().ravel())
    return np.concatenate(pieces).astype(np.float64)


def density_on_points(
    model: StructuredDensityModel, t: float, cx: np.ndarray, rho: np.ndarray, beta: np.ndarray,
) -> np.ndarray:
    """Tilted density at arbitrary (cx, rho) points for one time (beta given)."""
    c = np.stack([cx.ravel(), rho.ravel(), np.zeros(cx.size)], axis=1)
    log_raw = raw_log_density_batched(model, t, c)
    r2 = cx.ravel() ** 2 + rho.ravel() ** 2
    tilt = beta[0] + beta[1] * cx.ravel() + beta[2] * (r2 - 3.0)
    return np.exp(log_raw + tilt).reshape(cx.shape)


def marginal_x_at(model, t, centers, beta, rho_max=9.0, n_rho=64) -> np.ndarray:
    rho, wr = gauss_legendre_nodes(rho_max, n_rho)
    CX, RHO = np.meshgrid(np.asarray(centers, dtype=np.float64), rho, indexing="ij")
    f = density_on_points(model, t, CX, RHO, beta)
    return f @ (2.0 * np.pi * rho * wr)


def marginal_y_at(model, t, centers, beta, half_width=9.0, n_grid=145) -> np.ndarray:
    """Transverse marginal int int f(cx, sqrt(cy^2 + cz^2)) dcx dcz on a trapezoid grid.

    The 2-D trapezoid rule is spectrally accurate here because the integrand
    is smooth in (cx, cz) and negligible at the ends (spacing 0.125).
    """
    grid, w = trapezoid_nodes(half_width, n_grid)
    centers = np.asarray(centers, dtype=np.float64)
    result = np.empty(centers.shape[0])
    for start in range(0, centers.shape[0], 8):
        block = centers[start:start + 8]
        CY, CX, CZ = np.meshgrid(block, grid, grid, indexing="ij")
        RHO = np.sqrt(CY**2 + CZ**2)
        f = density_on_points(model, t, CX, RHO, beta)        # [nb, nx, nz]
        result[start:start + 8] = np.einsum("cxz,x,z->c", f, w, w)
    return result


# --------------------------------------------------------------------------
# Full evaluation of one weight file
# --------------------------------------------------------------------------
def evaluate_model(
    model: StructuredDensityModel,
    config: dict[str, Any],
    reference_path: str | Path,
    output: str | Path,
    *,
    with_residual: bool = True,
    with_fields: bool = True,
    portable_reload_linf: float | None = None,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reference = np.load(reference_path)
    times = np.asarray(reference["time"], dtype=np.float64)
    nu = float(config.get("nu", 1.0))

    quad_train = quadrature_tensors(build_quadrature(
        cx_half_width=float(config.get("cx_half_width", 8.0)), n_cx=int(config.get("n_cx", 129)),
        rho_max=float(config.get("rho_max", 8.0)), n_rho=int(config.get("n_rho", 32)),
    ))
    quad_fine = quadrature_tensors(build_quadrature(
        cx_half_width=float(config.get("eval_cx_half_width", 9.0)), n_cx=int(config.get("eval_n_cx", 257)),
        rho_max=float(config.get("eval_rho_max", 9.0)), n_rho=int(config.get("eval_n_rho", 64)),
    ))

    fine = tilted_slices(model, times, quad_fine)
    train = tilted_slices(model, times, quad_train)
    tm, rm = fine["tilted"], fine["raw"]
    analytic_qx = analytic_heat_flux_history(times, nu=nu)

    qx_fine = tm["q"][:, 0]
    qx_train = train["tilted"]["q"][:, 0]
    qx_err_fine = relative_l2(qx_fine, analytic_qx)
    qx_err_train = relative_l2(qx_train, analytic_qx)
    quadrature_uncertainty_pp = abs(qx_err_fine - qx_err_train) * 100.0
    decay_rate = fit_decay_rate(times, qx_fine, qx0=float(qx_fine[0]))
    raw_qx_err = relative_l2(rm["q"][:, 0], analytic_qx)

    ref_q = reference["q"]
    ref_pij = reference["pij"]
    ref_dm2 = reference["dm2"]
    qx_vs_reference = relative_l2(qx_fine, ref_q[:, 0])
    aniso_model = tm["pij"][:, 0] - tm["pij"][:, 3]
    aniso_ref = ref_pij[:, 0] - ref_pij[:, 3]
    stress_anisotropy = float(np.max(np.abs(aniso_model - aniso_ref)))

    # Marginals against the FV reference at the three G0 time indices.
    centers = np.asarray(reference["histogram_centers"], dtype=np.float64)
    selected = np.unique([0, len(times) // 2, len(times) - 1])
    pred_marginals, ref_marginals = [], []
    marginal_x_all = []
    for index in range(len(times)):
        marginal_x_all.append(marginal_x_at(model, float(times[index]), centers, fine["beta"][index]))
    marginal_x_all = np.asarray(marginal_x_all)
    for index in selected:
        my = marginal_y_at(model, float(times[index]), centers, fine["beta"][index])
        pred_marginals.extend([marginal_x_all[index], my, my])
        ref_marginals.extend([reference["marginal_x"][index], reference["marginal_y"][index], reference["marginal_z"][index]])
    pred_marginals = np.asarray(pred_marginals)
    ref_marginals = np.asarray(ref_marginals)
    marginal_error = relative_l2(pred_marginals, ref_marginals)
    marginal_x_history_error = relative_l2(marginal_x_all, reference["marginal_x"])

    # Full field against the FV reference on its own grid (diagnostic).
    field_errors = np.full(len(times), np.nan)
    if with_fields and "fields" in reference.files:
        gx = np.asarray(reference["grid_x"], dtype=np.float64)
        gr = np.asarray(reference["grid_rho"], dtype=np.float64)
        GX, GR = np.meshgrid(gx, gr, indexing="ij")
        for index in range(len(times)):
            f_model = density_on_points(model, float(times[index]), GX, GR, fine["beta"][index])
            f_ref = reference["fields"][index]
            field_errors[index] = np.linalg.norm(f_model - f_ref) / max(np.linalg.norm(f_ref), 1.0e-300)

    residual_rms = residual_diagnostics(model, times, quad_train, nu) if with_residual else np.full(len(times), np.nan)

    # Structural checks.
    ic_nodes = quad_fine.nodes64.numpy()
    log_f0 = tf_initial_logpdf("heat_flux", tf.constant(ic_nodes, tf.float64)).numpy().ravel()
    log_raw0 = model.raw_log_density(tf.zeros([ic_nodes.shape[0], 1]), tf.constant(ic_nodes, tf.float32)).numpy().ravel()
    initial_linf = float(np.max(np.abs(np.exp(log_raw0.astype(np.float64)) - np.exp(log_f0))))
    # Exact axisymmetry: the network must not see the transverse angle.
    probe = ic_nodes[::97]
    rotated = np.stack([probe[:, 0], probe[:, 1] * np.cos(0.7), probe[:, 1] * np.sin(0.7)], axis=1)
    t_probe = tf.fill([probe.shape[0], 1], 0.37)
    axisym_linf = float(np.max(np.abs(
        model.raw_log_density(t_probe, tf.constant(probe, tf.float32)).numpy()
        - model.raw_log_density(t_probe, tf.constant(rotated, tf.float32)).numpy()
    )))

    max_mass = float(np.max(np.abs(tm["mass"] - 1.0)))
    max_momentum = float(np.max(np.abs(tm["mean"][:, 0])))
    max_energy = float(np.max(np.abs(tm["dm2"] - 3.0)))
    raw_mass = float(np.max(np.abs(rm["mass"] - 1.0)))
    raw_energy = float(np.max(np.abs(rm["dm2"] - 3.0)))
    transverse = float(np.max(np.linalg.norm(tm["q"][:, 1:3], axis=1)) / max(np.max(np.abs(ref_q[:, 0])), 1.0e-12))
    log_f_fine = np.asarray(fine["log_f"], dtype=np.float64)
    all_log_f_finite = bool(np.all(np.isfinite(log_f_fine)))
    min_log_f = float(np.min(log_f_fine)) if all_log_f_finite else float("nan")
    decay_rate_relative = float(abs(decay_rate - HEAT_FLUX_DECAY_RATE_FACTOR * nu) / (HEAT_FLUX_DECAY_RATE_FACTOR * nu))

    gates = heat_flux_gate_summary(qx_err_fine)
    checks = {
        "qx_analytic_l2_primary": bool(qx_err_fine <= GATES.qx_analytic_l2_primary),
        "qx_quadrature_uncertainty": bool(quadrature_uncertainty_pp <= GATES.qx_quadrature_uncertainty_pp),
        "mass_drift": bool(max_mass <= GATES.mass_drift),
        "momentum_drift": bool(max_momentum <= GATES.momentum_drift),
        "energy_drift": bool(max_energy <= GATES.energy_drift),
        "raw_mass_drift": bool(raw_mass <= GATES.raw_mass_drift),
        "raw_energy_drift": bool(raw_energy <= GATES.raw_energy_drift),
        "marginal_relative_l2": bool(marginal_error <= GATES.marginal_relative_l2),
        "stress_anisotropy": bool(stress_anisotropy <= GATES.stress_anisotropy_abs),
        "transverse_heat_flux": bool(transverse <= GATES.transverse_heat_flux_relative),
        # The exponential ansatz is strictly positive when its logarithm is
        # finite. Check the whole held-out grid: checking only min(log f)
        # would miss an isolated +inf whenever another entry was finite.
        "positive_density": all_log_f_finite,
        "initial_condition_linf": bool(initial_linf <= GATES.initial_condition_linf),
        "exact_axisymmetry": bool(axisym_linf <= 1.0e-4),
        "decay_rate": bool(np.isfinite(decay_rate) and decay_rate_relative <= GATES.decay_rate_relative),
    }
    if portable_reload_linf is not None:
        checks["portable_reload"] = bool(portable_reload_linf <= GATES.portable_reload_linf)
    # Raw (pre-tilt) drift is diagnostic: it measures how much of the solution
    # the three tilt parameters carry, not whether f solves the equation.
    diagnostic = ("raw_mass_drift", "raw_energy_drift")
    structural_pass = bool(all(v for k, v in checks.items() if not k.startswith("qx_") and k not in diagnostic))
    primary_pass = bool(structural_pass and checks["qx_analytic_l2_primary"] and checks["qx_quadrature_uncertainty"])
    publication_pass = bool(primary_pass and qx_err_fine <= GATES.qx_analytic_l2_publication)
    status = "PASS" if primary_pass else "NO_GO"

    # Selection score (lower is better), used by the checkpoint sweep.
    score = float(
        qx_err_fine + 0.25 * marginal_error
        + 10.0 * max(0.0, raw_mass - GATES.raw_mass_drift)
        + 10.0 * max(0.0, raw_energy - GATES.raw_energy_drift)
        + 10.0 * max(0.0, stress_anisotropy - GATES.stress_anisotropy_abs)
    )
    if not np.isfinite(score):
        score = 1.0e300

    metrics: dict[str, Any] = {
        "status": status,
        "primary_pass": primary_pass,
        "publication_pass": publication_pass,
        "structural_pass": structural_pass,
        "gate_level": gates["level"],
        "gate_checks": checks,
        "diagnostic_checks": list(diagnostic),
        "gates": asdict(GATES),
        "selection_score": score,
        "qx_analytic_l2_fine": float(qx_err_fine),
        "qx_analytic_l2_train_quadrature": float(qx_err_train),
        "qx_quadrature_uncertainty_pp": float(quadrature_uncertainty_pp),
        "qx_analytic_l2_raw_pre_tilt": float(raw_qx_err),
        "qx_vs_fv_reference_l2": float(qx_vs_reference),
        "fv_reference_qx_analytic_l2": float(relative_l2(ref_q[:, 0], analytic_qx)),
        "decay_rate": float(decay_rate),
        "decay_rate_exact": float(HEAT_FLUX_DECAY_RATE_FACTOR * nu),
        "decay_rate_relative_error": decay_rate_relative,
        "effective_prandtl": float(effective_prandtl_number(decay_rate, nu=nu)),
        "max_mass_error": max_mass,
        "max_momentum_norm": max_momentum,
        "max_energy_error": max_energy,
        "raw_max_mass_error": raw_mass,
        "raw_max_energy_error": raw_energy,
        "max_abs_beta": float(np.max(np.abs(fine["beta"]))),
        "marginal_relative_l2": float(marginal_error),
        "marginal_x_history_relative_l2": float(marginal_x_history_error),
        "field_relative_l2_max": float(np.nanmax(field_errors)) if np.any(np.isfinite(field_errors)) else float("nan"),
        "field_relative_l2_by_time": [float(v) for v in field_errors],
        "stress_anisotropy_max_abs_error": stress_anisotropy,
        "stress_anisotropy_model_at_tmax": float(aniso_model[-1]),
        "stress_anisotropy_reference_at_tmax": float(aniso_ref[-1]),
        "max_transverse_heat_flux_relative": transverse,
        "min_log_density_on_fine_quadrature": min_log_f,
        "initial_condition_linf": initial_linf,
        "exact_axisymmetry_linf": axisym_linf,
        "residual_rms_by_time": [float(v) for v in residual_rms],
        "residual_rms_max": float(np.nanmax(residual_rms)) if np.any(np.isfinite(residual_rms)) else float("nan"),
        "quadrature_train": {"n_cx": int(quad_train.size // int(config.get("n_rho", 32))), "n_rho": int(config.get("n_rho", 32))},
        "quadrature_fine": {"n_cx": int(config.get("eval_n_cx", 257)), "n_rho": int(config.get("eval_n_rho", 64))},
        "reference_file": str(reference_path),
    }
    if portable_reload_linf is not None:
        metrics["portable_reload_linf"] = float(portable_reload_linf)

    np.savez_compressed(
        output / "validation.npz", time=times, analytic_qx=analytic_qx,
        model_qx_fine=qx_fine, model_qx_train=qx_train, raw_qx=rm["q"][:, 0],
        model_mass=tm["mass"], model_mean=tm["mean"], model_dm2=tm["dm2"], model_pij=tm["pij"], model_q=tm["q"],
        raw_mass=rm["mass"], raw_dm2=rm["dm2"], beta=fine["beta"],
        reference_q=ref_q, reference_pij=ref_pij, reference_dm2=ref_dm2,
        histogram_centers=centers, selected_time_indices=selected,
        predicted_marginals=pred_marginals, reference_marginals=ref_marginals,
        marginal_x_history=marginal_x_all, reference_marginal_x=reference["marginal_x"],
        field_relative_l2=field_errors, residual_rms=residual_rms,
    )
    with (output / "moments_by_time.csv").open("w") as stream:
        stream.write("time,mass,mean_x,dm2,p_xx,p_yy,q_x,q_x_train_quadrature,raw_q_x,analytic_q_x,reference_q_x,"
                     "q_x_error_vs_analytic,beta_0,beta_1,beta_2,raw_mass,raw_dm2,residual_rms,field_rel_l2\n")
        for i, t in enumerate(times):
            stream.write(
                f"{t:.6f},{tm['mass'][i]:.12f},{tm['mean'][i,0]:.3e},{tm['dm2'][i]:.12f},{tm['pij'][i,0]:.8f},"
                f"{tm['pij'][i,3]:.8f},{qx_fine[i]:.8f},{qx_train[i]:.8f},{rm['q'][i,0]:.8f},{analytic_qx[i]:.8f},"
                f"{ref_q[i,0]:.8f},{qx_fine[i]-analytic_qx[i]:+.3e},{fine['beta'][i,0]:+.6e},{fine['beta'][i,1]:+.6e},"
                f"{fine['beta'][i,2]:+.6e},{rm['mass'][i]:.8f},{rm['dm2'][i]:.8f},{residual_rms[i]:.4e},{field_errors[i]:.4e}\n"
            )
    try:
        make_plot(output, times, analytic_qx, qx_fine, rm["q"][:, 0], ref_q[:, 0], centers, selected,
                  pred_marginals, ref_marginals, residual_rms, aniso_model, aniso_ref, rm, tm)
    except Exception as error:  # plotting must never fail the gate
        metrics["plot_error"] = repr(error)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def make_plot(output, times, analytic_qx, qx, raw_qx, ref_qx, centers, selected, pred_m, ref_m,
              residual_rms, aniso_model, aniso_ref, rm, tm) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    ax.plot(times, analytic_qx, "k-", label="analytic 0.25 e^{-4t/3}")
    ax.plot(times, ref_qx, "g--", label="FV reference")
    ax.plot(times, qx, "r.-", label="G1 model (tilted)")
    ax.plot(times, raw_qx, "r:", alpha=0.6, label="pre-tilt")
    ax.set_xlabel("t"); ax.set_ylabel("Q_x"); ax.legend(fontsize=8); ax.set_title("heat flux")
    ax = axes[0, 1]
    ax.plot(times, (qx - analytic_qx) / np.max(analytic_qx), "r.-", label="(Q_x - exact)/Q_x(0)")
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("t"); ax.legend(fontsize=8); ax.set_title("heat-flux error")
    ax = axes[0, 2]
    ax.plot(times, rm["mass"] - 1.0, "b.-", label="pre-tilt mass - 1")
    ax.plot(times, rm["dm2"] - 3.0, "m.-", label="pre-tilt energy - 3")
    ax.plot(times, tm["mass"] - 1.0, "b--", label="tilted mass - 1")
    ax.plot(times, tm["dm2"] - 3.0, "m--", label="tilted energy - 3")
    ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=8); ax.set_title("invariants")
    ax = axes[1, 0]
    for k, index in enumerate(selected):
        ax.plot(centers, ref_m[3 * k], "k-", lw=0.8)
        ax.plot(centers, pred_m[3 * k], "--", label=f"t={times[index]:.2f}")
    ax.set_xlabel("c_x"); ax.set_title("marginal_x: model (dashed) vs FV (black)"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(times, residual_rms, "k.-"); ax.set_yscale("log"); ax.set_xlabel("t")
    ax.set_title("f-weighted RMS log residual (train quadrature)")
    ax = axes[1, 2]
    ax.plot(times, aniso_model, "r.-", label="model p_xx - p_yy")
    ax.plot(times, aniso_ref, "k-", label="FV reference")
    ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=8); ax.set_title("stress anisotropy")
    fig.tight_layout()
    fig.savefig(output / "g1_validation.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# Checkpoint sweep
# --------------------------------------------------------------------------
def sweep_checkpoints(config: dict[str, Any], sweep_dir: Path, reference: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(sweep_dir.glob("epoch-*.weights.h5"))
    if not files:
        raise SystemExit(f"No checkpoints found in {sweep_dir}")
    rows = []
    best = None
    for weights in files:
        model = build_model(config, weights)
        metrics = evaluate_model(model, config, reference, output / weights.stem.replace(".weights", ""),
                                 with_residual=False, with_fields=False)
        row = {
            "checkpoint": weights.name, "status": metrics["status"], "selection_score": metrics["selection_score"],
            "qx_analytic_l2_fine": metrics["qx_analytic_l2_fine"],
            "qx_quadrature_uncertainty_pp": metrics["qx_quadrature_uncertainty_pp"],
            "marginal_relative_l2": metrics["marginal_relative_l2"],
            "raw_max_mass_error": metrics["raw_max_mass_error"], "raw_max_energy_error": metrics["raw_max_energy_error"],
            "stress_anisotropy_max_abs_error": metrics["stress_anisotropy_max_abs_error"],
            "structural_pass": metrics["structural_pass"], "primary_pass": metrics["primary_pass"],
            "publication_pass": metrics["publication_pass"],
        }
        rows.append(row)
        print("sweep " + json.dumps(row), flush=True)
        # Structurally admissible checkpoints are preferred; among them the
        # lowest deterministic selection score wins.
        key = (0 if metrics["structural_pass"] else 1, metrics["selection_score"])
        if best is None or key < best[0]:
            best = (key, weights, metrics)
    assert best is not None
    _, weights, _ = best
    selected_weights = output / "selected.weights.h5"
    shutil.copy2(weights, selected_weights)
    # Candidate sweeps skip the expensive residual and full-field diagnostics.
    # Re-evaluate the selected file completely so the published/aggregated
    # record contains those diagnostics and the blocking reload gate.
    selected_model = build_model(config, weights)
    reload_linf = portable_reload_check(selected_model, config, selected_weights)
    metrics = evaluate_model(
        selected_model,
        config,
        reference,
        output / "selected_evaluation",
        with_residual=True,
        with_fields=True,
        portable_reload_linf=reload_linf,
    )
    summary = {
        "selected_checkpoint": weights.name,
        "selected_status": metrics["status"],
        "selected_metrics": metrics,
        "checkpoints": rows,
    }
    (output / "checkpoint_sweep.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"SELECTED {weights.name} status={metrics['status']} qx_l2={metrics['qx_analytic_l2_fine']:.4f}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--weights")
    group.add_argument("--sweep-dir")
    parser.add_argument("--strict-gate", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.weights:
        model = build_model(config, args.weights)
        metrics = evaluate_model(model, config, args.reference, args.output)
        print("G1_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
        status = metrics["status"]
    else:
        summary = sweep_checkpoints(config, Path(args.sweep_dir), Path(args.reference), Path(args.output))
        status = summary["selected_status"]
    print(f"G1_STATUS {status}", flush=True)
    if args.strict_gate and status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

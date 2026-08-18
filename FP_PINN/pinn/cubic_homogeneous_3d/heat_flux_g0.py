"""Analytic heat-flux relaxation target and predeclared Stage-2 G0 gates.

For the homogeneous cubic Fokker--Planck closure used in this directory,
the active third central moment obeys

    dQ_x/dt = -(4/3) * nu * Q_x,

with the exact initial value Q_x(0)=0.25 for the positive skew-mixture case.
The analytic history, rather than a noisy particle history, is therefore the
primary quantitative reference for the heat-flux PINN gate.  Particle results
remain an independent implementation cross-check.
"""

from __future__ import annotations

from typing import Any

import numpy as np


HEAT_FLUX_INITIAL_QX = 0.25
HEAT_FLUX_DECAY_RATE_FACTOR = 4.0 / 3.0
STRESS_DECAY_RATE_FACTOR = 2.0

# These thresholds were frozen before the G0 rerun.  Equality passes.
HEAT_FLUX_CONTINUATION_L2 = 0.05
HEAT_FLUX_PRIMARY_L2 = 0.02
HEAT_FLUX_PUBLICATION_L2 = 0.01


def analytic_heat_flux_history(
    times: np.ndarray,
    *,
    qx0: float = HEAT_FLUX_INITIAL_QX,
    nu: float = 1.0,
) -> np.ndarray:
    """Return Qx(t)=Qx(0) exp[-(4/3) nu t] at the requested times."""
    values = np.asarray(times, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("times must be finite")
    if np.any(values < 0.0):
        raise ValueError("times must be nonnegative")
    if not np.isfinite(qx0):
        raise ValueError("qx0 must be finite")
    if not np.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu must be finite and positive")
    return float(qx0) * np.exp(-HEAT_FLUX_DECAY_RATE_FACTOR * float(nu) * values)


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Return a finite relative L2 error with an explicit zero-reference guard."""
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have the same shape")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(reference)):
        return float("inf")
    denominator = float(np.linalg.norm(reference))
    if denominator <= 0.0:
        raise ValueError("reference norm must be positive")
    return float(np.linalg.norm(prediction - reference) / denominator)


def fit_decay_rate(
    times: np.ndarray,
    qx: np.ndarray,
    *,
    qx0: float = HEAT_FLUX_INITIAL_QX,
) -> float:
    """Fit Qx=Qx0 exp(-rate*t) with Qx0 fixed by the exact initial ansatz."""
    times = np.asarray(times, dtype=np.float64)
    qx = np.asarray(qx, dtype=np.float64)
    if times.shape != qx.shape:
        raise ValueError("times and qx must have the same shape")
    same_sign = qx * float(qx0) > 0.0
    mask = (
        np.isfinite(times)
        & np.isfinite(qx)
        & (times > 0.0)
        & same_sign
        & (np.abs(qx) > abs(float(qx0)) * 1.0e-10)
    )
    if np.count_nonzero(mask) < 2:
        return float("nan")
    selected_t = times[mask]
    log_ratio = np.log(np.abs(qx[mask] / float(qx0)))
    denominator = float(np.dot(selected_t, selected_t))
    if denominator <= 0.0:
        return float("nan")
    return float(-np.dot(selected_t, log_ratio) / denominator)


def effective_prandtl_number(decay_rate: float, *, nu: float = 1.0) -> float:
    """Infer Pr_eff = heat-relaxation rate / stress-relaxation rate."""
    if not np.isfinite(decay_rate) or not np.isfinite(nu) or nu <= 0.0:
        return float("nan")
    return float(decay_rate / (STRESS_DECAY_RATE_FACTOR * nu))


def heat_flux_gate_summary(error: float) -> dict[str, Any]:
    """Classify one analytic-history L2 error at all three frozen levels."""
    finite = bool(np.isfinite(error))
    continuation = bool(finite and error <= HEAT_FLUX_CONTINUATION_L2)
    primary = bool(finite and error <= HEAT_FLUX_PRIMARY_L2)
    publication = bool(finite and error <= HEAT_FLUX_PUBLICATION_L2)
    if publication:
        level = "PUBLICATION_PASS"
    elif primary:
        level = "PRIMARY_PASS"
    elif continuation:
        level = "CONTINUATION_PASS"
    else:
        level = "NO_GO"
    return {
        "analytic_history_relative_l2": float(error),
        "continuation_threshold": HEAT_FLUX_CONTINUATION_L2,
        "primary_threshold": HEAT_FLUX_PRIMARY_L2,
        "publication_threshold": HEAT_FLUX_PUBLICATION_L2,
        "continuation_pass": continuation,
        "primary_pass": primary,
        "publication_pass": publication,
        "level": level,
    }


__all__ = [
    "HEAT_FLUX_CONTINUATION_L2",
    "HEAT_FLUX_DECAY_RATE_FACTOR",
    "HEAT_FLUX_INITIAL_QX",
    "HEAT_FLUX_PRIMARY_L2",
    "HEAT_FLUX_PUBLICATION_L2",
    "analytic_heat_flux_history",
    "effective_prandtl_number",
    "fit_decay_rate",
    "heat_flux_gate_summary",
    "relative_l2",
]

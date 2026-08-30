"""Closed-form second-moment history reserved for G2 validation.

The cubic closure enforces d(P_ij-theta delta_ij)/dt = -2 nu
(P_ij-theta delta_ij).  The qualification trainer never imports this module.
"""

from __future__ import annotations

import numpy as np


STRESS_PXX0 = 1.6
STRESS_PPERP0 = 0.7
STRESS_DELTA0 = STRESS_PXX0 - STRESS_PPERP0
STRESS_DECAY_RATE_FACTOR = 2.0


def analytic_stress_delta(times: np.ndarray, *, nu: float = 1.0) -> np.ndarray:
    values = np.asarray(times, dtype=np.float64)
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("times must be finite and nonnegative")
    if not np.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu must be finite and positive")
    return STRESS_DELTA0 * np.exp(-STRESS_DECAY_RATE_FACTOR * float(nu) * values)


def analytic_stress_components(times: np.ndarray, *, nu: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    amplitude = np.exp(-STRESS_DECAY_RATE_FACTOR * float(nu) * np.asarray(times, dtype=np.float64))
    return 1.0 + 0.6 * amplitude, 1.0 - 0.3 * amplitude


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
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


def fit_decay_rate(times: np.ndarray, delta: np.ndarray, *, delta0: float = STRESS_DELTA0) -> float:
    times = np.asarray(times, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    mask = np.isfinite(times) & np.isfinite(delta) & (times > 0.0) & (delta * delta0 > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    t = times[mask]
    log_ratio = np.log(np.abs(delta[mask] / float(delta0)))
    denominator = float(np.dot(t, t))
    return float(-np.dot(t, log_ratio) / denominator) if denominator > 0.0 else float("nan")


__all__ = [
    "STRESS_DECAY_RATE_FACTOR",
    "STRESS_DELTA0",
    "STRESS_PPERP0",
    "STRESS_PXX0",
    "analytic_stress_components",
    "analytic_stress_delta",
    "fit_decay_rate",
    "relative_l2",
]

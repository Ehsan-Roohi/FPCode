"""Analytic reference solution for the dimensionless 1-D OU equation.

The initial density is an equally weighted mixture of two Gaussians.  Under

    f_t = (v f)_v + f_vv,

each component remains Gaussian, with mean mu(t) = mu(0) exp(-t) and
variance s2(t) = 1 + (s2(0) - 1) exp(-2t).

This module intentionally depends only on NumPy so the reference can be
tested before TensorFlow or a GPU is available.
"""

from __future__ import annotations

import numpy as np


DEFAULT_MU0 = 2.0
DEFAULT_SIGMA0 = 0.5


def trapezoidal_integral(
    values: np.ndarray, coordinates: np.ndarray, axis: int = -1
) -> np.ndarray:
    """NumPy 1.26/2.x compatible trapezoidal integration."""
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return implementation(values, coordinates, axis=axis)


def gaussian_pdf(v: np.ndarray, mean: float, variance: float) -> np.ndarray:
    """Return a normalized Gaussian density evaluated at ``v``."""
    v = np.asarray(v, dtype=np.float64)
    return np.exp(-0.5 * (v - mean) ** 2 / variance) / np.sqrt(
        2.0 * np.pi * variance
    )


def initial_density(
    v: np.ndarray,
    mu0: float = DEFAULT_MU0,
    sigma0: float = DEFAULT_SIGMA0,
) -> np.ndarray:
    """Symmetric, normalized bimodal initial density."""
    variance0 = sigma0**2
    return 0.5 * (
        gaussian_pdf(v, -mu0, variance0)
        + gaussian_pdf(v, mu0, variance0)
    )


def exact_density(
    t: np.ndarray | float,
    v: np.ndarray,
    mu0: float = DEFAULT_MU0,
    sigma0: float = DEFAULT_SIGMA0,
) -> np.ndarray:
    """Exact OU solution; ``t`` and ``v`` follow NumPy broadcasting rules."""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    decay = np.exp(-t)
    mean = mu0 * decay
    variance = 1.0 + (sigma0**2 - 1.0) * decay**2
    return 0.5 * (
        gaussian_pdf(v, -mean, variance)
        + gaussian_pdf(v, mean, variance)
    )


def exact_second_moment(
    t: np.ndarray | float,
    mu0: float = DEFAULT_MU0,
    sigma0: float = DEFAULT_SIGMA0,
) -> np.ndarray:
    """Return E[V^2](t) for the symmetric mixture."""
    t = np.asarray(t, dtype=np.float64)
    initial_m2 = mu0**2 + sigma0**2
    return 1.0 + (initial_m2 - 1.0) * np.exp(-2.0 * t)

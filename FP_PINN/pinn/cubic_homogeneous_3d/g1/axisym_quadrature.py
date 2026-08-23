"""Deterministic velocity quadrature for the axisymmetric heat-flux case.

The heat-flux initial state, the homogeneous cubic Fokker--Planck operator and
the axisymmetric PINN ansatz are all invariant under rotations about the x
axis.  Every velocity moment of such a density is therefore an exact
two-dimensional integral,

    <g> = int dcx int_0^inf 2*pi*rho drho  g(cx, rho) f(t, cx, rho),

with rho^2 = cy^2 + cz^2.  This module builds tensor-product quadrature grids
for that integral:

* cx: composite trapezoid on [-L, L].  For integrands that decay to zero with
  all derivatives at the ends (Gaussian-like densities) the trapezoid rule is
  spectrally accurate, so a spacing of 0.125 already integrates a unit-variance
  Gaussian to machine precision.
* rho: Gauss--Legendre on [0, L_rho].  The integrand 2*pi*rho*f(rho) has a
  nonzero first derivative at rho=0, which makes the trapezoid rule only
  second-order accurate there; Gauss--Legendre has no such end-point term and
  converges geometrically for smooth integrands.

Every node is returned as a 3-D velocity (cx, rho, 0).  Because the PINN is
axisymmetric, the 3-D velocity-space Laplacian evaluated by automatic
differentiation at (cx, rho, 0) equals the axisymmetric Laplacian
d_xx + d_rho,rho + (1/rho) d_rho, so no change to the differentiation code is
required.

Nothing here depends on TensorFlow; the arrays are consumed by the training
and evaluation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisymQuadrature:
    """Tensor-product (cx, rho) quadrature written as 3-D nodes and weights."""

    nodes: np.ndarray        # [n, 3] float64, nodes (cx, rho, 0)
    weights: np.ndarray      # [n] float64, includes the 2*pi*rho Jacobian
    cx_nodes: np.ndarray     # [nx]
    rho_nodes: np.ndarray    # [nr]
    cx_half_width: float
    rho_max: float
    cx_shift: float

    @property
    def size(self) -> int:
        return int(self.nodes.shape[0])

    def integrate(self, values: np.ndarray) -> np.ndarray:
        """Integrate values sampled at the nodes (last axis = node axis)."""
        return np.tensordot(np.asarray(values, dtype=np.float64), self.weights, axes=([-1], [0]))


def trapezoid_nodes(half_width: float, n_nodes: int, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Uniform nodes on [-L, L] with composite trapezoid weights.

    ``shift`` translates the whole grid by a fraction of one cell.  This is the
    deterministic analogue of an independent quadrature panel: the integration
    error of a shifted trapezoid rule is the same, but the node set differs, so
    a network cannot fit a single node set.
    """
    if n_nodes < 3:
        raise ValueError("n_nodes must be at least three")
    spacing = 2.0 * half_width / (n_nodes - 1)
    nodes = -half_width + spacing * (np.arange(n_nodes) + shift)
    weights = np.full(n_nodes, spacing)
    # The density is negligible at the ends, so the half-weights only matter
    # formally; keep them for exactness of the rule.
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return nodes, weights


def gauss_legendre_nodes(upper: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre nodes and weights on [0, upper]."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least two")
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    return 0.5 * upper * (x + 1.0), 0.5 * upper * w


def build_quadrature(
    cx_half_width: float = 8.0,
    n_cx: int = 129,
    rho_max: float = 8.0,
    n_rho: int = 32,
    cx_shift: float = 0.0,
) -> AxisymQuadrature:
    """Build the tensor-product (cx, rho) quadrature as 3-D nodes."""
    cx, wx = trapezoid_nodes(cx_half_width, n_cx, cx_shift)
    rho, wr = gauss_legendre_nodes(rho_max, n_rho)
    CX, RHO = np.meshgrid(cx, rho, indexing="ij")
    WX, WR = np.meshgrid(wx, wr, indexing="ij")
    weights = (WX * WR * 2.0 * np.pi * RHO).ravel()
    nodes = np.stack([CX.ravel(), RHO.ravel(), np.zeros(CX.size)], axis=1)
    return AxisymQuadrature(
        nodes=nodes, weights=weights, cx_nodes=cx, rho_nodes=rho,
        cx_half_width=float(cx_half_width), rho_max=float(rho_max), cx_shift=float(cx_shift),
    )


def build_panels(
    n_panels: int,
    rng: np.random.Generator,
    **kwargs,
) -> list[AxisymQuadrature]:
    """Independent deterministic panels that differ by a random cx shift.

    The first panel is unshifted; later panels are shifted by a uniformly
    random fraction of one cx cell.  All panels have identical accuracy.
    """
    if n_panels < 1:
        raise ValueError("n_panels must be at least one")
    panels = [build_quadrature(cx_shift=0.0, **kwargs)]
    for _ in range(n_panels - 1):
        panels.append(build_quadrature(cx_shift=float(rng.uniform(0.0, 1.0)), **kwargs))
    return panels


def invariant_features(nodes: np.ndarray) -> np.ndarray:
    """psi(c) = (1, cx, |c|^2): the collision invariants of the operator."""
    nodes = np.asarray(nodes, dtype=np.float64)
    r2 = np.sum(nodes * nodes, axis=1)
    return np.stack([np.ones(nodes.shape[0]), nodes[:, 0], r2], axis=1)


def heat_flux_mode(nodes: np.ndarray, theta: float = 1.0) -> np.ndarray:
    """Third Hermite mode  phi3 = cx (|c|^2 - 5 theta).

    phi3 is orthogonal, in the Maxwellian-weighted inner product with
    temperature theta, to 1, cx, and |c|^2.  Its Maxwellian projection onto
    cx|c|^2 is 10 theta^3, so for a small amplitude b the heat flux is
    Q_x ~ 10 theta^3 b.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    r2 = np.sum(nodes * nodes, axis=1)
    return nodes[:, 0] * (r2 - 5.0 * theta)


def axisymmetric_moments(quad: AxisymQuadrature, density: np.ndarray) -> dict[str, float]:
    """Deterministic moments of an axisymmetric density sampled on the quadrature.

    Returns raw mass, mean_x, and the *central* moments used by the closure:
    dm2, pxx, pyy (= pzz), Qx, m3x, dm4.  Transverse odd moments vanish by
    symmetry and are not returned.
    """
    density = np.asarray(density, dtype=np.float64)
    w = quad.weights * density
    mass = float(w.sum())
    cx = quad.nodes[:, 0]
    rho2 = quad.nodes[:, 1] ** 2
    mean_x = float((w * cx).sum() / mass)
    vx = cx - mean_x
    r2 = vx * vx + rho2

    def avg(values: np.ndarray) -> float:
        return float((w * values).sum() / mass)

    return {
        "mass": mass,
        "mean_x": mean_x,
        "dm2": avg(r2),
        "pxx": avg(vx * vx),
        "pyy": 0.5 * avg(rho2),
        "qx": avg(vx * r2),
        "m3x": avg(vx ** 3),
        "dm4": avg(r2 * r2),
    }


__all__ = [
    "AxisymQuadrature",
    "axisymmetric_moments",
    "build_panels",
    "build_quadrature",
    "gauss_legendre_nodes",
    "heat_flux_mode",
    "invariant_features",
    "trapezoid_nodes",
]

"""Discrete conservative axisymmetric Dougherty--Fokker--Planck tools for F1."""
from pathlib import Path

import numpy as np
import tensorflow as tf


F1_GATES = {
    "maximum_flux_relative_spread": 1.0e-2,
    "relative_residual_rms": 2.0e-1,
    "boundary_relative_error": 5.0e-3,
    "collision_invariant_relative_rms": 5.0e-6,
    "collision_projection_relative_rms": 5.0e-2,
}


def first_derivative_matrix(nodes):
    """Second-order three-point derivative matrix on arbitrary distinct nodes."""
    x = np.asarray(nodes, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or np.any(np.diff(x) <= 0.0):
        raise ValueError("nodes must be a strictly increasing 1-D array of length >= 3")
    matrix = np.zeros((len(x), len(x)), dtype=np.float64)
    for i in range(len(x)):
        if i == 0:
            index = np.arange(3)
        elif i == len(x) - 1:
            index = np.arange(len(x) - 3, len(x))
        else:
            index = np.arange(i - 1, i + 2)
        dx = x[index] - x[i]
        vandermonde = np.vstack((np.ones(3), dx, dx * dx))
        matrix[i, index] = np.linalg.solve(vandermonde, np.array([0.0, 1.0, 0.0]))
    return matrix


def structured_axisymmetric_quadrature(reference_path, decimals=7):
    """Compress a tensor 3-V quadrature to the complete ``(vx, r^2)`` product."""
    with np.load(Path(reference_path), allow_pickle=False) as data:
        velocity = np.asarray(data["v"], dtype=np.float64)
        weights = np.asarray(data["w"], dtype=np.float64)
    vx = np.unique(np.round(velocity[:, 0], decimals))
    s = np.unique(np.round(velocity[:, 1] ** 2 + velocity[:, 2] ** 2, decimals))
    ix = np.searchsorted(vx, np.round(velocity[:, 0], decimals))
    ir = np.searchsorted(s, np.round(velocity[:, 1] ** 2 + velocity[:, 2] ** 2, decimals))
    weight_grid = np.zeros((len(vx), len(s)), dtype=np.float64)
    np.add.at(weight_grid, (ix, ir), weights)
    if np.any(weight_grid <= 0.0):
        raise ValueError("registered quadrature is not a complete (vx, r^2) product")
    return vx, s, weight_grid


def raw_dougherty_collision_numpy(f, vx, s, density, velocity, temperature):
    """Axisymmetric Dougherty operator in divergence form on a structured grid.

    With ``s=r^2``, the radial divergence is
    ``2 d/ds [s (f + 2 T df/ds)]``.  This avoids the removable singularity at
    ``r=0`` and preserves differentiability of the TensorFlow implementation.
    """
    f = np.asarray(f, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if f.shape[-2:] != (len(vx), len(s)):
        raise ValueError("f must end in (len(vx), len(s))")
    dv = first_derivative_matrix(vx)
    ds = first_derivative_matrix(s)
    dfdv = np.einsum("ij,...jk->...ik", dv, f)
    dfds = np.einsum("ij,...kj->...ki", ds, f)
    u = np.asarray(velocity, dtype=np.float64)[..., None, None]
    t = np.asarray(temperature, dtype=np.float64)[..., None, None]
    jv = (vx[None, :, None] - u) * f + t * dfdv
    js = s[None, None, :] * (f + 2.0 * t * dfds)
    return (np.einsum("ij,...jk->...ik", dv, jv)
            + 2.0 * np.einsum("ij,...kj->...ki", ds, js))


def conservative_collision_numpy(raw, f, vx, s, weights):
    """Project a discrete collision term onto the null space of 1, vx, |v|^2."""
    raw = np.asarray(raw, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    phi = np.stack(np.broadcast_arrays(
        np.ones((len(vx), len(s))), vx[:, None], vx[:, None] ** 2 + s[None, :]),
        axis=-1)
    rhs = np.einsum("...ij,ij,ijk->...k", raw, weights, phi)
    gram = np.einsum("...ij,ij,ijk,ijl->...kl", f, weights, phi, phi)
    lam = np.linalg.solve(gram + 1.0e-12 * np.eye(3), rhs[..., None])[..., 0]
    corrected = raw - f * np.einsum("...k,ijk->...ij", lam, phi)
    return corrected


def collision_invariants_numpy(collision, vx, s, weights):
    """Return discrete mass, momentum, and twice-energy production."""
    phi = np.stack(np.broadcast_arrays(
        np.ones((len(vx), len(s))), np.asarray(vx)[:, None],
        np.asarray(vx)[:, None] ** 2 + np.asarray(s)[None, :]), axis=-1)
    return np.einsum("...ij,ij,ijk->...k", collision, weights, phi)


def conservative_dougherty_collision_tf(f, values, vx_1d, s_1d, weights,
                                         derivative_v, derivative_s):
    """Differentiable float32 F1 collision operator and its two audit errors."""
    dtype = f.dtype
    vx_1d = tf.cast(vx_1d, dtype)
    s_1d = tf.cast(s_1d, dtype)
    weights = tf.cast(weights, dtype)
    derivative_v = tf.cast(derivative_v, dtype)
    derivative_s = tf.cast(derivative_s, dtype)
    nvx, ns = tf.shape(vx_1d)[0], tf.shape(s_1d)[0]
    grid = tf.reshape(f, (-1, nvx, ns))
    dfdv = tf.einsum("ij,xjk->xik", derivative_v, grid)
    dfds = tf.einsum("ij,xkj->xki", derivative_s, grid)
    u = values["u"][:, None, None]
    temperature = values["temperature"][:, None, None]
    velocity_flux = ((vx_1d[None, :, None]-u)*grid+temperature*dfdv)
    radial_flux = s_1d[None, None, :]*(grid+2.0*temperature*dfds)
    raw_grid = (tf.einsum("ij,xjk->xik", derivative_v, velocity_flux)
                +2.0*tf.einsum("ij,xkj->xki", derivative_s, radial_flux))
    raw = tf.reshape(raw_grid, tf.shape(f))

    vx = tf.repeat(vx_1d, tf.shape(s_1d)[0])
    s = tf.tile(s_1d, [tf.shape(vx_1d)[0]])
    phi = tf.stack((tf.ones_like(vx), vx, vx*vx+s), axis=1)
    weighted_f = f*weights[None, :]
    rhs = tf.einsum("xv,vi->xi", raw*weights[None, :], phi)
    gram = tf.einsum("xv,vi,vj->xij", weighted_f, phi, phi)
    lam = tf.linalg.solve(gram+tf.cast(1.0e-8, dtype)*tf.eye(
        3, dtype=dtype)[None, :, :], rhs[..., None])[..., 0]
    corrected = raw-f*tf.einsum("xi,vi->xv", lam, phi)
    invariant = tf.einsum("xv,vi->xi", corrected*weights[None, :], phi)
    denominator = (tf.einsum("xv,vi->xi", tf.abs(raw)*weights[None, :],
                             tf.abs(phi))+tf.cast(1.0e-12, dtype))
    invariant_relative = tf.sqrt(tf.reduce_mean(tf.square(invariant/denominator)))
    correction_relative = (tf.sqrt(tf.reduce_mean(tf.square(corrected-raw)))
                           /(tf.sqrt(tf.reduce_mean(tf.square(raw)))
                             +tf.cast(1.0e-12, dtype)))
    return corrected, invariant_relative, correction_relative


def analytic_dougherty_collision_tf(log_builder, f, values, vx, s, weights):
    """Evaluate Dougherty--FP from velocity derivatives of ``log(f)``.

    ``log_builder(vx_matrix, s_matrix)`` must evaluate the represented log
    distribution elementwise while holding its x-dependent coefficients fixed.
    In ``s=r^2`` coordinates the radial Laplacian is
    ``4(s log(f)_ss + log(f)_s)``.
    """
    dtype = f.dtype
    vx = tf.cast(vx, dtype)
    s = tf.cast(s, dtype)
    weights = tf.cast(weights, dtype)
    vx_matrix = tf.broadcast_to(vx[None, :], tf.shape(f))
    s_matrix = tf.broadcast_to(s[None, :], tf.shape(f))
    with tf.GradientTape(persistent=True) as second:
        second.watch((vx_matrix, s_matrix))
        with tf.GradientTape(persistent=True) as first:
            first.watch((vx_matrix, s_matrix))
            log_f = log_builder(vx_matrix, s_matrix)
        log_v = first.gradient(
            log_f, vx_matrix, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        log_s = first.gradient(
            log_f, s_matrix, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    log_vv = second.gradient(
        log_v, vx_matrix, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    log_ss = second.gradient(
        log_s, s_matrix, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    del first, second
    u = values["u"][:, None]
    temperature = values["temperature"][:, None]
    factor = (3.0+(vx_matrix-u)*log_v+2.0*s_matrix*log_s
              +temperature*(log_vv+4.0*s_matrix*log_ss+4.0*log_s
                             +log_v*log_v+4.0*s_matrix*log_s*log_s))
    raw = f*factor

    phi = tf.stack((tf.ones_like(vx), vx, vx*vx+s), axis=1)
    weighted_f = f*weights[None, :]
    rhs = tf.einsum("xv,vi->xi", raw*weights[None, :], phi)
    gram = tf.einsum("xv,vi,vj->xij", weighted_f, phi, phi)
    lam = tf.linalg.solve(gram+tf.cast(1.0e-8, dtype)*tf.eye(
        3, dtype=dtype)[None, :, :], rhs[..., None])[..., 0]
    corrected = raw-f*tf.einsum("xi,vi->xv", lam, phi)
    invariant = tf.einsum("xv,vi->xi", corrected*weights[None, :], phi)
    denominator = (tf.einsum("xv,vi->xi", tf.abs(raw)*weights[None, :],
                             tf.abs(phi))+tf.cast(1.0e-12, dtype))
    invariant_relative = tf.sqrt(tf.reduce_mean(tf.square(invariant/denominator)))
    correction_relative = (tf.sqrt(tf.reduce_mean(tf.square(corrected-raw)))
                           /(tf.sqrt(tf.reduce_mean(tf.square(raw)))
                             +tf.cast(1.0e-12, dtype)))
    return corrected, invariant_relative, correction_relative

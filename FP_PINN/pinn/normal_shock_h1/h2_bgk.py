"""Numerical building blocks for the H2 stationary BGK shock solver."""
from pathlib import Path

import numpy as np


PSI_MAX = 0.65
KNUDSEN_EFFECTIVE = 1.0 / 80.0
PROJECTION_MOMENTS = ("mass", "momentum", "energy", "normal_stress", "energy_flux")
H2_GATES = {
    "rho_core_relative_l2": 2.0e-2,
    "u_core_relative_l2": 2.0e-2,
    "temperature_core_relative_l2": 2.0e-2,
    "qx_core_relative_l2": 2.0e-1,
    "sigma_xx_core_relative_l2": 2.0e-1,
    "maximum_flux_relative_spread": 1.0e-2,
    "relative_residual_rms": 2.0e-1,
    "boundary_relative_error": 5.0e-3,
}


def compact_quadrature_arrays(reference_path, decimals=7):
    """Return flattened (vx, transverse-speed-squared, weight) arrays."""
    with np.load(Path(reference_path), allow_pickle=False) as data:
        velocity = np.asarray(data["v"], dtype=np.float64)
        weights = np.asarray(data["w"], dtype=np.float64)
    vx_values, ix = np.unique(np.round(velocity[:, 0], decimals), return_inverse=True)
    r2_values, ir = np.unique(
        np.round(velocity[:, 1] ** 2 + velocity[:, 2] ** 2, decimals),
        return_inverse=True,
    )
    compact = np.zeros((len(vx_values), len(r2_values)), dtype=np.float64)
    np.add.at(compact, (ix, ir), weights)
    iv, jr = np.nonzero(compact > 0.0)
    return vx_values[iv], r2_values[jr], compact[iv, jr]


def moments_numpy(distribution, vx, r2, weights):
    """Integrate the five H2 validation moments and three invariant fluxes."""
    f = np.asarray(distribution, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    wf = f * weights
    rho = np.sum(wf, axis=-1)
    momentum = np.sum(wf * vx, axis=-1)
    u = momentum / rho
    cx = vx[None, :] - u[..., None]
    c2 = cx * cx + r2
    temperature = np.sum(wf * c2, axis=-1) / (3.0 * rho)
    qx = 0.5 * np.sum(wf * cx * c2, axis=-1)
    sigma = np.sum(wf * (cx * cx - c2 / 3.0), axis=-1)
    return {
        "rho": rho,
        "u": u,
        "temperature": temperature,
        "qx": qx,
        "sigma_xx": sigma,
        "mass_flux": momentum,
        "momentum_flux": np.sum(wf * vx * vx, axis=-1),
        "energy_flux": 0.5 * np.sum(wf * vx * (vx * vx + r2), axis=-1),
    }


def hermite_modes(vx, r2, u, temperature):
    """Heat-flux and normal-stress modes used by the positive ansatz."""
    cx = (np.asarray(vx) - u) / np.sqrt(temperature)
    c2 = cx * cx + np.asarray(r2) / temperature
    phi_q = ((0.5 * c2 - 2.5) * cx) / 8.0
    phi_sigma = (cx * cx - c2 / 3.0) / 4.0
    return phi_q, phi_sigma


def conservative_fields_numpy(rho, temperature, invariant_fluxes):
    """Recover u, normal stress, and heat flux from the three steady fluxes.

    The construction is the macroscopic part of H2R2.  It makes the flux
    identities algebraic rather than trainable penalties.
    """
    rho = np.asarray(rho, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    mass, momentum, energy = np.asarray(invariant_fluxes, dtype=np.float64)
    u = mass / rho
    sigma = momentum - rho * u * u - rho * temperature
    qx = energy - u * (
        0.5 * rho * u * u + 2.5 * rho * temperature + sigma
    )
    return {"rho": rho, "u": u, "temperature": temperature,
            "qx": qx, "sigma_xx": sigma}


def raw_projection_targets(fields, invariant_fluxes, velocity_scale):
    """Return scaled raw moments constrained by the H2R2 exponential tilt."""
    rho = np.asarray(fields["rho"], dtype=np.float64)
    u = np.asarray(fields["u"], dtype=np.float64)
    temperature = np.asarray(fields["temperature"], dtype=np.float64)
    sigma = np.asarray(fields["sigma_xx"], dtype=np.float64)
    energy = float(np.asarray(invariant_fluxes, dtype=np.float64)[2])
    scale = float(velocity_scale)
    return np.stack(
        (rho,
         rho * u / scale,
         rho * (u * u + 3.0 * temperature) / scale**2,
         (rho * u * u + rho * temperature + sigma) / scale**2,
         np.full_like(rho, energy / scale**3)),
        axis=-1,
    )

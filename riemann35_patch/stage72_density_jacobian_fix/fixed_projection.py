from __future__ import annotations
import numpy as np
from hyqmom_fp import macroscopic_state
import riemann35_patch.stage57_persistent_four_population.persistent_mixture as pm


def density_consistent_heat_flux_projection(populations, desired_heat_flux):
    """Stage57 projector with the missing rho factor restored in third-moment Jacobian rows."""
    baseline_moments = pm.persistent_gaussian_mixture_moments(populations)
    baseline_macro = macroscopic_state(baseline_moments)
    baseline_tensor = pm._symmetric_third_tensor(pm._third_components(baseline_moments))
    baseline_heat_flux = 0.5 * np.einsum("ijj->i", baseline_tensor)
    target_tensor = baseline_tensor - pm._trace_carrying_tensor(baseline_heat_flux) + pm._trace_carrying_tensor(desired_heat_flux)
    target_third = pm._unique_third_components(target_tensor)
    target_covariance = baseline_macro.covariance
    target_mean = baseline_macro.velocity
    count = populations.probabilities.size
    projected = populations
    minimum_fraction = 1.0

    for _ in range(20):
        current_moments = pm.persistent_gaussian_mixture_moments(projected)
        current_macro = macroscopic_state(current_moments)
        current_third = pm._third_components(current_moments)
        mean_delta = target_mean - current_macro.velocity
        covariance_delta_target = target_covariance - current_macro.covariance
        covariance_right = np.asarray([covariance_delta_target[i, j] for i, j in pm.SYMMETRIC_PAIRS])
        third_right = target_third - current_third
        rhs = np.concatenate([mean_delta, covariance_right, third_right])
        if np.linalg.norm(rhs) <= 2.0e-13:
            break

        matrix = np.zeros((19, count * 9))
        offsets = projected.means - current_macro.velocity
        for component, (probability, offset, covariance) in enumerate(zip(projected.probabilities, offsets, projected.covariances)):
            for direction in range(3):
                column = component * 9 + direction
                basis_mean = np.zeros(3); basis_mean[direction] = 1.0
                matrix[direction, column] = probability
                covariance_derivative = probability * (np.outer(offset, basis_mean) + np.outer(basis_mean, offset))
                for p, (i, j) in enumerate(pm.SYMMETRIC_PAIRS):
                    matrix[3 + p, column] = covariance_derivative[i, j]
                for p, powers in enumerate(pm.THIRD_INDICES):
                    dirs = tuple(axis for axis, power in enumerate(powers) for _ in range(power))
                    i, j, k = dirs
                    matrix[9 + p, column] = populations.rho * probability * (
                        basis_mean[i] * (offset[j] * offset[k] + covariance[j, k])
                        + basis_mean[j] * (offset[i] * offset[k] + covariance[i, k])
                        + basis_mean[k] * (offset[i] * offset[j] + covariance[i, j])
                    )
            for p, (left, right) in enumerate(pm.SYMMETRIC_PAIRS):
                column = component * 9 + 3 + p
                basis_cov = np.zeros((3, 3)); basis_cov[left, right] = 1.0; basis_cov[right, left] = 1.0
                for q, (i, j) in enumerate(pm.SYMMETRIC_PAIRS):
                    matrix[3 + q, column] = probability * basis_cov[i, j]
                for q, powers in enumerate(pm.THIRD_INDICES):
                    dirs = tuple(axis for axis, power in enumerate(powers) for _ in range(power))
                    i, j, k = dirs
                    matrix[9 + q, column] = populations.rho * probability * (
                        offset[i] * basis_cov[j, k] + offset[j] * basis_cov[i, k] + offset[k] * basis_cov[i, j]
                    )

        row_scale = np.maximum(np.linalg.norm(matrix, axis=1), 1.0e-12)
        correction = np.linalg.lstsq(matrix / row_scale[:, None], rhs / row_scale, rcond=1.0e-11)[0]
        mean_correction = correction.reshape(count, 9)[:, :3]
        covariance_correction = np.zeros_like(projected.covariances)
        for component in range(count):
            for p, (i, j) in enumerate(pm.SYMMETRIC_PAIRS):
                value = correction[component * 9 + 3 + p]
                covariance_correction[component, i, j] = value
                covariance_correction[component, j, i] = value
        fraction = 1.0
        for _ in range(60):
            trial_covariances = projected.covariances + fraction * covariance_correction
            if min(np.min(np.linalg.eigvalsh(item)) for item in trial_covariances) > 1.0e-12:
                break
            fraction *= 0.5
        else:
            raise FloatingPointError("density-consistent projection could not preserve SPD")
        minimum_fraction = min(minimum_fraction, fraction)
        projected = pm.PersistentGaussianMixtureState(
            rho=projected.rho,
            probabilities=projected.probabilities.copy(),
            means=projected.means + fraction * mean_correction,
            covariances=projected.covariances + fraction * covariance_correction,
        )
        pm._validate_state(projected)

    final_moments = pm.persistent_gaussian_mixture_moments(projected)
    final_macro = macroscopic_state(final_moments)
    final_tensor = pm._symmetric_third_tensor(pm._third_components(final_moments))
    lower_scale = max(np.linalg.norm(target_covariance), 1.0e-14)
    third_scale = max(np.linalg.norm(target_tensor), populations.rho * max(baseline_macro.theta, 1.0e-14) ** 1.5, 1.0e-14)
    residual = max(
        float(np.linalg.norm(final_macro.velocity - target_mean)),
        float(np.linalg.norm(final_macro.covariance - target_covariance) / lower_scale),
        float(np.linalg.norm(final_tensor - target_tensor) / third_scale),
    )
    return projected, minimum_fraction, residual


def install_density_consistent_projection():
    pm._heat_flux_trace_projection = density_consistent_heat_flux_projection

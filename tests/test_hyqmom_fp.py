"""Regression tests for the first HyQMOM-FP coupling milestone."""

from __future__ import annotations

import unittest

import numpy as np

from hyqmom_fp import (
    CubicFPCoefficients,
    GaussianTailClosure,
    HYQMOM_35_INDICES,
    HYQMOM_35_NAMES,
    coefficients_from_moments,
    coefficients_from_particles,
    macroscopic_state,
    maxwellian_moments_35,
    mixture_of_gaussians_moments_35,
    moments_35_from_particles,
    particle_cubic_fp_step,
    particle_macroscopic_state,
    projected_fp_collision_source,
    sample_gaussian_mixture,
)


class HyQMOMFPCollisionTests(unittest.TestCase):
    def setUp(self) -> None:
        covariance = np.asarray(
            [
                [0.72, 0.04, 0.01],
                [0.04, 0.48, -0.02],
                [0.01, -0.02, 0.39],
            ]
        )
        self.nonequilibrium = mixture_of_gaussians_moments_35(
            [
                (0.55, (-0.35, 0.08, 0.02), covariance),
                (0.45, (0.55, -0.04, -0.01), covariance),
            ]
        )

    def test_ordering_matches_hyqmom_35_contract(self) -> None:
        self.assertEqual(len(HYQMOM_35_INDICES), 35)
        self.assertEqual(len(set(HYQMOM_35_INDICES)), 35)
        self.assertEqual(HYQMOM_35_NAMES[0], "M000")
        self.assertEqual(HYQMOM_35_NAMES[-1], "M022")
        self.assertTrue(all(sum(index) <= 4 for index in HYQMOM_35_INDICES))

    def test_ou_operator_keeps_a_maxwellian_stationary(self) -> None:
        moments = maxwellian_moments_35(
            rho=1.7, velocity=(0.3, -0.2, 0.1), theta=0.8
        )
        coefficients = CubicFPCoefficients.ornstein_uhlenbeck(
            tau=0.6, theta=0.8
        )
        source = projected_fp_collision_source(moments, coefficients)
        self.assertLess(np.max(np.abs(source)), 2.0e-13)

    def test_cubic_source_conserves_collision_invariants(self) -> None:
        coefficients = coefficients_from_moments(
            self.nonequilibrium, tau=0.9, prandtl=2.0 / 3.0
        )
        source = projected_fp_collision_source(
            self.nonequilibrium, coefficients, GaussianTailClosure()
        )
        position = {index: i for i, index in enumerate(HYQMOM_35_INDICES)}
        self.assertEqual(source[position[(0, 0, 0)]], 0.0)
        self.assertEqual(source[position[(1, 0, 0)]], 0.0)
        self.assertEqual(source[position[(0, 1, 0)]], 0.0)
        self.assertEqual(source[position[(0, 0, 1)]], 0.0)
        energy_source = (
            source[position[(2, 0, 0)]]
            + source[position[(0, 2, 0)]]
            + source[position[(0, 0, 2)]]
        )
        self.assertLess(abs(energy_source), 2.0e-14)

    def test_explicit_step_preserves_mass_momentum_and_energy(self) -> None:
        moments = self.nonequilibrium.copy()
        coefficients = coefficients_from_moments(moments, tau=1.1)
        updated = moments + 1.0e-4 * projected_fp_collision_source(
            moments, coefficients
        )
        before = macroscopic_state(moments)
        after = macroscopic_state(updated)
        np.testing.assert_allclose(after.rho, before.rho, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(
            after.velocity, before.velocity, rtol=0.0, atol=2e-14
        )
        np.testing.assert_allclose(after.theta, before.theta, rtol=0.0, atol=2e-14)

    def test_cubic_projection_requests_sixth_order_tail(self) -> None:
        requested_orders: set[int] = set()
        gaussian = GaussianTailClosure()

        def recording_closure(index, moments, state):
            requested_orders.add(sum(index))
            return gaussian(index, moments, state)

        coefficients = coefficients_from_moments(self.nonequilibrium, tau=1.0)
        projected_fp_collision_source(
            self.nonequilibrium, coefficients, recording_closure
        )
        self.assertIn(5, requested_orders)
        self.assertIn(6, requested_orders)

    def test_particle_and_moment_coefficient_maps_agree(self) -> None:
        particles = sample_gaussian_mixture(
            [
                (0.55, (-0.35, 0.08, 0.02), 0.5 * np.eye(3)),
                (0.45, (0.55, -0.04, -0.01), 0.35 * np.eye(3)),
            ],
            particles=4_000,
            seed=17,
        )
        moments = moments_35_from_particles(particles)
        moment_coefficients = coefficients_from_moments(moments, tau=0.9)
        particle_coefficients = coefficients_from_particles(particles, tau=0.9)
        np.testing.assert_allclose(
            particle_coefficients.C,
            moment_coefficients.C,
            rtol=2e-13,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            particle_coefficients.gamma,
            moment_coefficients.gamma,
            rtol=2e-13,
            atol=2e-14,
        )
        self.assertAlmostEqual(
            particle_coefficients.beta, moment_coefficients.beta, places=13
        )

    def test_particle_step_is_reproducible_and_preserves_invariants(self) -> None:
        particles = sample_gaussian_mixture(
            [
                (0.5, (-0.55, 0.0, 0.0), 0.45 * np.eye(3)),
                (0.5, (0.55, 0.0, 0.0), 0.45 * np.eye(3)),
            ],
            particles=2_000,
            seed=23,
        )
        before = particle_macroscopic_state(particles)
        first, diagnostics = particle_cubic_fp_step(
            particles,
            dt=2.5e-4,
            tau=1.0,
            rng=np.random.default_rng(24),
        )
        second, _ = particle_cubic_fp_step(
            particles,
            dt=2.5e-4,
            tau=1.0,
            rng=np.random.default_rng(24),
        )
        np.testing.assert_array_equal(first, second)
        after = particle_macroscopic_state(first)
        np.testing.assert_allclose(after.velocity, before.velocity, atol=5e-15)
        self.assertAlmostEqual(after.theta, before.theta, places=14)
        self.assertLess(abs(diagnostics.energy_drift), 3e-14)
        self.assertLess(diagnostics.momentum_drift, 5e-15)


if __name__ == "__main__":
    unittest.main()

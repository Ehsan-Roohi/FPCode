"""Regression tests for the first HyQMOM-FP coupling milestone."""

from __future__ import annotations

import unittest

import numpy as np

from hyqmom_fp import (
    CubicFPCoefficients,
    GaussianTailClosure,
    WeightedNodeTailClosure,
    HYQMOM_35_INDICES,
    HYQMOM_35_NAMES,
    coefficients_from_moments,
    coefficients_from_particles,
    coefficients_from_weighted_nodes,
    finite_gaussian_mixture_fp_step,
    fit_equal_variance_marginal,
    fit_location_scale_marginal,
    fp_collision_moment_source,
    gaussian_gqmom_marginal,
    grad_hyqmom_fp_step,
    HermiteGalerkinTailClosure,
    first_35_from_hermite_state,
    initialize_hermite_moment_state,
    macroscopic_state,
    maximum_entropy_fp_step,
    maxwellian_moments_35,
    mixture_of_gaussians_moments_35,
    moments_35_from_particles,
    particle_cubic_fp_step,
    particle_macroscopic_state,
    projected_fp_collision_source,
    qmc_cubic_fp_step,
    realizability_margin_35,
    reconstruct_gaussian_mixture_quadrature,
    reconstruct_grad_hyqmom_quadrature,
    reconstruct_maximum_entropy_quadrature,
    reconstruct_two_population_quadrature,
    sample_gaussian_mixture,
    sample_gaussian_mixture_qmc,
    two_population_fp_step,
)
from hyqmom_fp.moments import multivariate_gaussian_raw_moment


POSITION = {index: position for position, index in enumerate(HYQMOM_35_INDICES)}


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

    def test_single_moment_generator_matches_projected_source_before_projection(self) -> None:
        closure = GaussianTailClosure()
        coefficients = coefficients_from_moments(
            self.nonequilibrium, tau=0.9, prandtl=2.0 / 3.0, closure=closure
        )
        source = projected_fp_collision_source(
            self.nonequilibrium,
            coefficients,
            closure,
            enforce_invariants=False,
        )
        for position, index in enumerate(HYQMOM_35_INDICES):
            self.assertAlmostEqual(
                fp_collision_moment_source(
                    index,
                    self.nonequilibrium,
                    coefficients,
                    closure=closure,
                ),
                source[position],
                places=12,
            )

    def test_physical_coefficients_enforce_stress_and_heat_flux_rates(self) -> None:
        quadrature = reconstruct_gaussian_mixture_quadrature(self.nonequilibrium)
        probabilities = quadrature.weights / np.sum(quadrature.weights)
        nodes = quadrature.nodes
        mean = np.sum(probabilities[:, None] * nodes, axis=0)
        peculiar = nodes - mean
        c2 = np.einsum("ni,ni->n", peculiar, peculiar)
        covariance = np.einsum(
            "n,ni,nj->ij", probabilities, peculiar, peculiar
        )
        theta = float(np.trace(covariance) / 3.0)
        heat_contraction = np.einsum(
            "n,ni,n->i", probabilities, peculiar, c2
        )
        coefficients = coefficients_from_weighted_nodes(
            nodes, quadrature.weights, tau=1.0, prandtl=2.0 / 3.0
        )
        drift = -peculiar + peculiar @ coefficients.C.T
        drift += (c2 - 3.0 * theta)[:, None] * coefficients.gamma
        drift += coefficients.beta * (
            c2[:, None] * peculiar - heat_contraction[None, :]
        )
        covariance_rate = (
            np.einsum("n,ni,nj->ij", probabilities, drift, peculiar)
            + np.einsum("n,ni,nj->ij", probabilities, peculiar, drift)
            + 2.0 * theta * np.eye(3)
        )
        heat_rate = np.einsum(
            "n,ni,n->i", probabilities, drift, c2
        ) + 2.0 * np.einsum(
            "n,ni,n->i",
            probabilities,
            peculiar,
            np.einsum("ni,ni->n", peculiar, drift),
        )
        np.testing.assert_allclose(
            covariance_rate,
            -2.0 * (covariance - theta * np.eye(3)),
            atol=8.0e-12,
            rtol=3.0e-11,
        )
        np.testing.assert_allclose(
            heat_rate,
            -(4.0 / 3.0) * heat_contraction,
            atol=8.0e-12,
            rtol=3.0e-11,
        )
        self.assertLessEqual(coefficients.beta, 0.0)

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

        def empirical_tail(index, retained, state):
            del retained, state
            return float(np.mean(np.prod(particles ** np.asarray(index), axis=1)))

        moment_coefficients = coefficients_from_moments(
            moments, tau=0.9, closure=empirical_tail
        )
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

    def test_equal_variance_fit_reconstructs_symmetric_bimodal_marginal(self) -> None:
        second = 0.45 + 0.55**2
        fourth = 3.0 * 0.45**2 + 6.0 * 0.45 * 0.55**2 + 0.55**4
        fit = fit_equal_variance_marginal(second, 0.0, fourth)
        self.assertEqual(fit.weights.size, 2)
        np.testing.assert_allclose(fit.weights, [0.5, 0.5], atol=2e-14)
        np.testing.assert_allclose(np.sort(fit.means), [-0.55, 0.55], atol=2e-14)
        self.assertAlmostEqual(fit.variance, 0.45, places=13)
        self.assertLess(fit.reconstruction_error, 2e-14)

    @staticmethod
    def _mixture_moment(fit, order: int) -> float:
        nodes, weights = np.polynomial.hermite.hermgauss(max(4, (order + 2) // 2))
        value = 0.0
        for probability, mean, variance in zip(
            fit.weights, fit.means, fit.component_variances
        ):
            samples = mean + np.sqrt(2.0 * variance) * nodes
            value += probability * np.dot(weights / np.sqrt(np.pi), samples**order)
        return float(value)

    def test_location_scale_fit_matches_symmetric_leptokurtic_state(self) -> None:
        fit = fit_location_scale_marginal(1.0, 0.0, 4.5)
        self.assertEqual(fit.branch, "symmetric-location-scale")
        np.testing.assert_allclose(fit.means, [0.0, 0.0], atol=0.0)
        self.assertGreater(np.ptp(fit.component_variances), 1.0)
        self.assertAlmostEqual(self._mixture_moment(fit, 2), 1.0, places=13)
        self.assertAlmostEqual(self._mixture_moment(fit, 3), 0.0, places=13)
        self.assertAlmostEqual(self._mixture_moment(fit, 4), 4.5, places=12)
        self.assertTrue(np.isfinite(self._mixture_moment(fit, 6)))

    def test_gaussian_gqmom_matches_hyqmom_fifth_moment(self) -> None:
        for third, fourth in ((0.0, 3.0), (0.4, 2.2), (-0.7, 3.4)):
            marginal = gaussian_gqmom_marginal(third, fourth)
            numerical = np.asarray(
                [
                    np.dot(marginal.weights, marginal.nodes**order)
                    for order in range(7)
                ]
            )
            expected_fifth = 0.5 * third * (
                5.0 * fourth - 3.0 * third**2 - 1.0
            )
            np.testing.assert_allclose(
                numerical[:5], [1.0, 0.0, 1.0, third, fourth], atol=2.0e-13
            )
            self.assertAlmostEqual(numerical[5], expected_fifth, places=12)
            self.assertGreater(numerical[6], 0.0)

    def test_grad_hyqmom_reconstructs_all_35_moments(self) -> None:
        quadrature = reconstruct_grad_hyqmom_quadrature(self.nonequilibrium)
        np.testing.assert_allclose(
            quadrature.reconstructed_moments,
            self.nonequilibrium,
            rtol=3.0e-12,
            atol=3.0e-12,
        )
        self.assertLess(quadrature.relative_moment_residual, 3.0e-12)
        closure = WeightedNodeTailClosure(quadrature.nodes, quadrature.weights)
        self.assertTrue(
            np.isfinite(closure((6, 0, 0), self.nonequilibrium, None))
        )

    def test_grad_hyqmom_split_step_is_conservative_and_realizable(self) -> None:
        updated, diagnostics = grad_hyqmom_fp_step(
            self.nonequilibrium, 2.5e-3, 1.0
        )
        before = macroscopic_state(self.nonequilibrium)
        after = macroscopic_state(updated)
        self.assertAlmostEqual(after.rho, before.rho, places=14)
        np.testing.assert_allclose(after.velocity, before.velocity, atol=2.0e-14)
        self.assertAlmostEqual(after.theta, before.theta, places=13)
        self.assertGreaterEqual(diagnostics.limiter_fraction, 0.0)
        self.assertLessEqual(diagnostics.limiter_fraction, 1.0)
        self.assertGreater(diagnostics.realizability_margin, -5.0e-13)

    def test_leptokurtic_sixth_moment_is_bounded_across_zero_skewness(self) -> None:
        exact = fit_location_scale_marginal(1.0, 0.0, 4.5)
        target = self._mixture_moment(exact, 6)
        for epsilon in (1.0e-3, 1.0e-6, 1.0e-9):
            positive = fit_location_scale_marginal(1.0, epsilon, 4.5)
            negative = fit_location_scale_marginal(1.0, -epsilon, 4.5)
            self.assertLess(positive.reconstruction_error, 2.0e-12)
            self.assertLess(negative.reconstruction_error, 2.0e-12)
            sixth_positive = self._mixture_moment(positive, 6)
            sixth_negative = self._mixture_moment(negative, 6)
            self.assertTrue(np.isfinite(sixth_positive))
            self.assertAlmostEqual(sixth_positive, sixth_negative, places=9)
            self.assertLess(abs(sixth_positive - target), 20.0 * epsilon)

    def test_location_scale_fit_covers_high_symmetric_kurtosis(self) -> None:
        fit = fit_location_scale_marginal(1.0, 0.0, 18.5)
        self.assertEqual(fit.branch, "symmetric-location-scale-high-kurtosis")
        self.assertGreater(np.min(fit.component_variances), 0.0)
        self.assertAlmostEqual(self._mixture_moment(fit, 2), 1.0, places=12)
        self.assertAlmostEqual(self._mixture_moment(fit, 4), 18.5, places=10)

    def test_unequal_weight_fit_covers_skewed_high_kurtosis(self) -> None:
        probability = 0.05
        complement = 1.0 - probability
        separation = 0.5
        means = np.asarray([complement * separation, -probability * separation])
        variances = np.asarray([10.0, 0.5])
        weights = np.asarray([probability, complement])
        second = float(np.dot(weights, variances + means**2))
        third = float(np.dot(weights, means**3 + 3.0 * means * variances))
        fourth = float(
            np.dot(weights, means**4 + 6.0 * means**2 * variances + 3.0 * variances**2)
        )
        fit = fit_location_scale_marginal(second, third, fourth)
        self.assertEqual(fit.branch, "unequal-weight-location-scale")
        self.assertGreater(np.min(fit.weights), 1.0e-3)
        self.assertGreaterEqual(np.min(fit.component_variances), 0.0)
        self.assertAlmostEqual(self._mixture_moment(fit, 2), second, places=11)
        self.assertAlmostEqual(self._mixture_moment(fit, 3), third, places=10)
        self.assertAlmostEqual(self._mixture_moment(fit, 4), fourth, places=9)

    def test_quadrature_reconstructs_axis_aligned_leptokurtic_35_state(self) -> None:
        half_difference = np.sqrt(1.5 / 3.0)
        moments = mixture_of_gaussians_moments_35(
            [
                (0.5, (0.0, 0.0, 0.0), np.diag([1.0 + half_difference, 1.0, 1.0])),
                (0.5, (0.0, 0.0, 0.0), np.diag([1.0 - half_difference, 1.0, 1.0])),
            ]
        )
        quadrature = reconstruct_gaussian_mixture_quadrature(moments)
        np.testing.assert_allclose(
            quadrature.reconstructed_moments, moments, rtol=3e-12, atol=3e-12
        )
        self.assertTrue(any("location-scale" in item.branch for item in quadrature.marginals))

    def test_mixture_quadrature_reconstructs_the_symmetric_35_state(self) -> None:
        covariance = 0.45 * np.eye(3)
        moments = mixture_of_gaussians_moments_35(
            [
                (0.5, (-0.55, 0.0, 0.0), covariance),
                (0.5, (0.55, 0.0, 0.0), covariance),
            ]
        )
        quadrature = reconstruct_gaussian_mixture_quadrature(moments)
        self.assertEqual(quadrature.weights.size, 128)
        np.testing.assert_allclose(
            quadrature.reconstructed_moments, moments, rtol=2e-13, atol=2e-13
        )
        self.assertLess(quadrature.relative_moment_residual, 2e-13)

    def test_finite_mixture_step_is_conservative_realizable_and_continuous(self) -> None:
        covariance = 0.45 * np.eye(3)
        moments = mixture_of_gaussians_moments_35(
            [
                (0.5, (-0.55, 0.0, 0.0), covariance),
                (0.5, (0.55, 0.0, 0.0), covariance),
            ]
        )
        step, diagnostics = finite_gaussian_mixture_fp_step(
            moments, 2.5e-4, 1.0
        )
        before = macroscopic_state(moments)
        after = macroscopic_state(step)
        self.assertAlmostEqual(after.rho, before.rho, places=14)
        np.testing.assert_allclose(after.velocity, before.velocity, atol=2e-14)
        self.assertAlmostEqual(after.theta, before.theta, places=13)
        self.assertGreater(realizability_margin_35(step), -2e-13)
        self.assertEqual(diagnostics.quadrature_nodes, 128)
        half, _ = finite_gaussian_mixture_fp_step(moments, 1.25e-4, 1.0)
        ratio = np.linalg.norm(step - moments) / np.linalg.norm(half - moments)
        self.assertTrue(1.95 <= ratio <= 2.05)

    def test_asymmetric_two_gaussian_state_exercises_odd_moments(self) -> None:
        covariance = 0.38 * np.eye(3)
        moments = mixture_of_gaussians_moments_35(
            [
                (0.35, (-0.8, 0.0, 0.0), covariance),
                (0.65, (0.25, 0.0, 0.0), covariance),
            ]
        )
        quadrature = reconstruct_gaussian_mixture_quadrature(moments)
        np.testing.assert_allclose(
            quadrature.reconstructed_moments, moments, rtol=3e-13, atol=3e-13
        )
        self.assertGreater(abs(moments[POSITION[(3, 0, 0)]]), 1e-3)
        updated, diagnostics = finite_gaussian_mixture_fp_step(
            moments, 2.5e-4, 1.0
        )
        self.assertTrue(np.all(np.isfinite(updated)))
        self.assertGreater(diagnostics.realizability_margin, -2e-13)

    def test_two_population_tail_is_exact_for_a_rare_equal_covariance_beam(self) -> None:
        beam_weight = 0.08
        background_weight = 1.0 - beam_weight
        beam_mean = np.asarray([1.0, 0.35, -0.15])
        background_mean = -(beam_weight / background_weight) * beam_mean
        covariance = 2.5e-3 * np.eye(3)
        components = (
            (beam_weight, beam_mean, covariance),
            (background_weight, background_mean, covariance),
        )
        moments = mixture_of_gaussians_moments_35(components)
        quadrature = reconstruct_two_population_quadrature(
            moments, minimum_skewness_norm=0.1
        )
        np.testing.assert_allclose(
            np.sort(quadrature.component_weights),
            np.sort([beam_weight, background_weight]),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            quadrature.reconstructed_moments,
            moments,
            rtol=3.0e-12,
            atol=3.0e-12,
        )
        closure = WeightedNodeTailClosure(quadrature.nodes, quadrature.weights)
        state = macroscopic_state(moments)
        for index in ((5, 0, 0), (4, 1, 0), (3, 2, 1), (6, 0, 0)):
            exact = sum(
                weight * multivariate_gaussian_raw_moment(index, mean, local_covariance)
                for weight, mean, local_covariance in components
            )
            self.assertAlmostEqual(
                closure(index, moments, state), exact, delta=2.0e-10 * max(abs(exact), 1.0)
            )
        self.assertLess(quadrature.negative_mass_fraction, 1.0e-12)

    def test_two_population_step_is_conservative_and_realizable(self) -> None:
        beam_weight = 0.08
        beam_mean = np.asarray([1.0, 0.35, -0.15])
        background_mean = -(beam_weight / (1.0 - beam_weight)) * beam_mean
        covariance = 2.5e-3 * np.eye(3)
        moments = mixture_of_gaussians_moments_35(
            [
                (beam_weight, beam_mean, covariance),
                (1.0 - beam_weight, background_mean, covariance),
            ]
        )
        updated, diagnostics = two_population_fp_step(
            moments, 2.5e-3, 1.0, minimum_skewness_norm=0.1
        )
        before = macroscopic_state(moments)
        after = macroscopic_state(updated)
        self.assertAlmostEqual(after.rho, before.rho, places=14)
        np.testing.assert_allclose(after.velocity, before.velocity, atol=2.0e-14)
        self.assertAlmostEqual(after.theta, before.theta, places=13)
        self.assertGreaterEqual(diagnostics.limiter_fraction, 0.0)
        self.assertLessEqual(diagnostics.limiter_fraction, 1.0)
        self.assertGreater(diagnostics.realizability_margin, -5.0e-13)

    def test_hermite_tail_is_exact_for_a_shifted_maxwellian(self) -> None:
        mean = np.asarray([0.2, -0.1, 0.3])
        covariance = 0.7 * np.eye(3)
        dynamic = initialize_hermite_moment_state(
            [(1.0, mean, covariance)], maximum_order=6
        )
        moments = first_35_from_hermite_state(dynamic)
        closure = HermiteGalerkinTailClosure(dynamic, moments)
        for index in ((5, 0, 0), (4, 2, 0), (3, 2, 1), (8, 0, 0)):
            exact = multivariate_gaussian_raw_moment(index, mean, covariance)
            self.assertAlmostEqual(
                closure(index, moments),
                exact,
                delta=3.0e-13 * max(abs(exact), 1.0),
            )

    def test_qmc_gaussian_mixture_is_positive_and_matches_low_moments(self) -> None:
        components = [
            (0.08, np.asarray([1.0, 0.35, -0.15]), 0.02 * np.eye(3)),
            (
                0.92,
                np.asarray([-0.08 / 0.92, -0.028 / 0.92, 0.012 / 0.92]),
                0.02 * np.eye(3),
            ),
        ]
        nodes, weights = sample_gaussian_mixture_qmc(
            components, points_per_component=256, seed=17
        )
        self.assertTrue(np.all(weights > 0.0))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=14)
        qmc_moments = moments_35_from_particles(nodes, weights)
        exact = mixture_of_gaussians_moments_35(components)
        low_order = [position for position, index in enumerate(HYQMOM_35_INDICES) if sum(index) <= 2]
        np.testing.assert_allclose(
            qmc_moments[low_order], exact[low_order], rtol=2.0e-13, atol=2.0e-13
        )

    def test_qmc_step_is_reproducible_and_preserves_invariants(self) -> None:
        components = [
            (0.2, np.zeros(3), np.diag([1.0, 0.6, 1.4])),
            (0.8, np.zeros(3), 0.1 * np.eye(3)),
        ]
        nodes, weights = sample_gaussian_mixture_qmc(
            components, points_per_component=128, seed=23
        )
        before = particle_macroscopic_state(nodes, weights)
        first, diagnostics = qmc_cubic_fp_step(
            nodes, weights, dt=2.5e-3, tau=1.0, seed=29
        )
        second, _ = qmc_cubic_fp_step(
            nodes, weights, dt=2.5e-3, tau=1.0, seed=29
        )
        np.testing.assert_array_equal(first, second)
        after = particle_macroscopic_state(first, weights)
        np.testing.assert_allclose(after.velocity, before.velocity, atol=2.0e-14)
        self.assertAlmostEqual(after.theta, before.theta, places=13)
        self.assertLess(abs(diagnostics.momentum_drift), 2.0e-14)
        self.assertLess(abs(diagnostics.energy_drift), 2.0e-13)

    def test_maximum_entropy_rare_beam_is_positive_and_moment_consistent(self) -> None:
        beam_weight = 0.08
        beam_mean = np.asarray([1.0, 0.35, -0.15])
        background_mean = -(beam_weight / (1.0 - beam_weight)) * beam_mean
        covariance = 2.5e-3 * np.eye(3)
        moments = mixture_of_gaussians_moments_35(
            [
                (beam_weight, beam_mean, covariance),
                (1.0 - beam_weight, background_mean, covariance),
            ]
        )
        quadrature = reconstruct_maximum_entropy_quadrature(
            moments, nodes_per_dimension=4
        )
        self.assertTrue(np.all(quadrature.weights > 0.0))
        np.testing.assert_allclose(
            quadrature.reconstructed_moments,
            moments,
            rtol=3.0e-11,
            atol=3.0e-11,
        )
        updated, diagnostics = maximum_entropy_fp_step(
            moments, 2.5e-3, 1.0, nodes_per_dimension=4
        )
        before = macroscopic_state(moments)
        after = macroscopic_state(updated)
        self.assertAlmostEqual(after.rho, before.rho, places=14)
        np.testing.assert_allclose(after.velocity, before.velocity, atol=2.0e-14)
        self.assertAlmostEqual(after.theta, before.theta, places=13)
        self.assertGreater(diagnostics.realizability_margin, 0.0)


if __name__ == "__main__":
    unittest.main()

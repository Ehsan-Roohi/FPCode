from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from cubic_operator import (  # noqa: E402
    CASE_NAMES,
    analytic_initial_summary,
    build_closure_system,
    cubic_lambda,
    maxwellian_log_residual,
    moments_from_samples,
    nonlinear_drift,
    ou_cubic_step,
    sample_initial,
    solve_closure,
)


class CubicOperatorTests(unittest.TestCase):
    def test_analytic_initial_summaries_share_invariants(self) -> None:
        for case in CASE_NAMES:
            summary = analytic_initial_summary(case)
            self.assertAlmostEqual(float(summary["mass"]), 1.0)
            np.testing.assert_allclose(summary["mean"], 0.0, atol=0.0)
            self.assertAlmostEqual(float(summary["dm2"]), 3.0)
        self.assertAlmostEqual(
            float(analytic_initial_summary("heat_flux")["q"][0]), 0.25
        )

    def test_monte_carlo_initial_moments(self) -> None:
        rng = np.random.default_rng(481516)
        tolerances = {"equilibrium": 0.025, "stress": 0.03, "heat_flux": 0.035}
        for case in CASE_NAMES:
            values = sample_initial(case, 500_000, rng)
            moments = moments_from_samples(values)
            exact = analytic_initial_summary(case)
            np.testing.assert_allclose(moments.mean, exact["mean"], atol=0.008)
            self.assertAlmostEqual(moments.dm2, float(exact["dm2"]), delta=0.02)
            np.testing.assert_allclose(moments.pij, exact["pij"], atol=0.015)
            np.testing.assert_allclose(moments.q, exact["q"], atol=tolerances[case])

    def test_equilibrium_closure_is_zero(self) -> None:
        # Antithetic sampling makes every odd moment exactly zero.
        rng = np.random.default_rng(73)
        half = rng.normal(size=(200_000, 3))
        values = np.concatenate([half, -half], axis=0)
        moments = moments_from_samples(values)
        # Sampling leaves a small deviatoric stress, so isolate the exact
        # equilibrium limit with a high tolerance appropriate to N=4e5.
        closure = solve_closure(moments)
        self.assertLess(abs(closure.cubic_lambda), 2.0e-6)
        self.assertLess(np.linalg.norm(closure.vector), 5.0e-4)

    def test_maxwellian_exactly_satisfies_ou_equation(self) -> None:
        rng = np.random.default_rng(101)
        points = rng.normal(size=(1000, 3))
        residual = maxwellian_log_residual(points)
        self.assertLess(float(np.max(np.abs(residual))), 2.0e-14)

    def test_m4_yz_contains_r_squared_factor(self) -> None:
        values = np.array(
            [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0],
             [2.0, -1.0, 0.5], [-2.0, 1.0, -0.5]]
        )
        moments = moments_from_samples(values)
        centered = values - values.mean(axis=0)
        expected = np.mean(
            centered[:, 1] * centered[:, 2] * np.sum(centered**2, axis=1)
        )
        plain_yz = np.mean(centered[:, 1] * centered[:, 2])
        self.assertAlmostEqual(moments.m4[4], expected)
        self.assertNotAlmostEqual(moments.m4[4], plain_yz)

    def test_cubic_term_is_active_for_anisotropic_stress(self) -> None:
        rng = np.random.default_rng(22)
        values = sample_initial("stress", 300_000, rng)
        moments = moments_from_samples(values)
        lam = cubic_lambda(moments)
        self.assertLess(lam, -1.0e-4)
        closure = solve_closure(moments)
        self.assertAlmostEqual(closure.cubic_lambda, lam)
        nonlinear = nonlinear_drift(values[:100], moments, closure)
        without_cubic = (
            (values[:100] - moments.mean) @ closure.matrix.T
            + (np.sum((values[:100]-moments.mean)**2, axis=1)-moments.dm2)[:, None]
            * closure.gamma
        )
        self.assertGreater(np.linalg.norm(nonlinear - without_cubic), 1.0e-3)

    def test_closure_linear_system_residual(self) -> None:
        rng = np.random.default_rng(11)
        for case in ("stress", "heat_flux"):
            moments = moments_from_samples(sample_initial(case, 250_000, rng))
            lhs, rhs, _ = build_closure_system(moments)
            closure = solve_closure(moments)
            relative = np.linalg.norm(lhs @ closure.vector - rhs) / max(
                np.linalg.norm(rhs), 1.0e-14
            )
            self.assertLess(relative, 1.0e-8)

    def test_closure_enforces_exact_third_moment_decay_rate(self) -> None:
        """Check the C/Gamma/lambda signs against dQ/dt=-(4/3) nu Q."""
        rng = np.random.default_rng(20260810)
        nu = 1.7
        moments = moments_from_samples(sample_initial("heat_flux", 250_000, rng))
        lhs, _, lam = build_closure_system(moments, nu=nu)
        closure = solve_closure(moments, nu=nu)
        p, q = moments.pij, moments.q
        lambda_basis = np.array([
            3.0*moments.m5[0]-moments.dm2*q[0]
            -2.0*(p[0]*q[0]+p[1]*q[1]+p[2]*q[2]),
            3.0*moments.m5[1]-moments.dm2*q[1]
            -2.0*(p[1]*q[0]+p[3]*q[1]+p[4]*q[2]),
            3.0*moments.m5[2]-moments.dm2*q[2]
            -2.0*(p[2]*q[0]+p[4]*q[1]+p[5]*q[2]),
        ])
        induced_rate = (
            -3.0*nu*q + lhs[6:9]@closure.vector + lam*lambda_basis
        )
        np.testing.assert_allclose(
            induced_rate, -(4.0/3.0)*nu*q, rtol=2.0e-8, atol=2.0e-10
        )

    def test_particle_projection_preserves_momentum_and_energy(self) -> None:
        rng = np.random.default_rng(67)
        values = sample_initial("heat_flux", 100_000, rng)
        values -= values.mean(axis=0)
        values *= np.sqrt(3.0 / np.mean(np.sum(values * values, axis=1)))
        moments = moments_from_samples(values)
        updated = ou_cubic_step(
            values, moments, solve_closure(moments), 0.005, rng,
            target_dm2=3.0,
        )
        after = moments_from_samples(updated)
        np.testing.assert_allclose(after.mean, 0.0, atol=2.0e-14)
        self.assertAlmostEqual(after.dm2, 3.0, places=12)


if __name__ == "__main__":
    unittest.main()

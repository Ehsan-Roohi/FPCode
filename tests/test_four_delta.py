import numpy as np

from hyqmom_fp import (
    ActivationHysteresis,
    adaptive_tail_memory_fp_step,
    initialize_adaptive_tail_memory,
    maxwellian_moments_35,
    positive_microstate_from_components,
    reconstruct_grad_hyqmom_quadrature,
    regularized_four_delta_state,
)


def test_four_delta_state_satisfies_declared_constraints() -> None:
    state = regularized_four_delta_state()
    assert state.mass_error < 1.0e-14
    assert state.momentum_norm < 1.0e-14
    assert state.energy_trace_error < 1.0e-14
    assert state.central_third_norm > 0.05
    assert np.all(state.weights > 0.0)
    assert np.isclose(np.sum(state.weights), 1.0)
    assert np.all(np.linalg.eigvalsh(state.components[0][2]) > 0.0)


def test_four_delta_state_is_inside_gaussian_gqmom_domain() -> None:
    state = regularized_four_delta_state()
    quadrature = reconstruct_grad_hyqmom_quadrature(state.moments)
    assert quadrature.relative_moment_residual < 1.0e-10
    assert quadrature.minimum_hankel_margin > 0.0


def test_no_donor_rule_retains_a_sensor_safe_microstate() -> None:
    components = ((1.0, np.zeros(3), np.eye(3)),)
    microstate, moments, _ = positive_microstate_from_components(
        components, points_per_component=256, seed=20_260_814
    )
    policy = ActivationHysteresis(
        release_hold_steps=1,
        minimum_active_steps=0,
    )
    adaptive = initialize_adaptive_tail_memory(
        moments,
        candidate_microstate=microstate,
        hysteresis=policy,
        force_causal_birth=True,
    )
    updated, diagnostics = adaptive_tail_memory_fp_step(
        adaptive,
        1.0e-3,
        1.0,
        hysteresis=policy,
        sensor_interval_steps=1,
        causal_reactivation_available=False,
    )
    assert updated.mode == "micro"
    assert updated.microstate is not None
    assert diagnostics.release_blocked_without_causal_donor


def test_no_donor_option_does_not_change_macro_moments_definition() -> None:
    moments = maxwellian_moments_35(1.0, np.zeros(3), 1.0)
    assert np.isclose(moments[0], 1.0)

"""Structural tests for the Stage-32 development gate."""

from hyqmom_fp import stage25_hysteresis
from riemann35_patch.stage31.run_heldout_shock import configuration as stage31_config
from riemann35_patch.stage32.run_direction_aware_precursor import (
    BLIND_VALIDATION_MACH,
    FRONT_OBSERVABLES,
    configuration,
)


def test_stage32_changes_only_the_causal_front_observable_view() -> None:
    previous = stage31_config("workstation")
    current = configuration("workstation")
    frozen = (
        "mach",
        "initial_active_half_width",
        "sensor_interval_steps",
        "release_sensor_interval_steps",
        "macro_equilibrium_tolerance",
        "kinetic_front_on",
        "release_persistence_steps",
        "profile_error_limit_percent",
    )
    assert all(current[name] == previous[name] for name in frozen)
    assert current["kinetic_front_on"] == stage25_hysteresis().tail_on == 0.40
    assert current["kinetic_front_observables"] == FRONT_OBSERVABLES
    assert previous["kinetic_front_observables"] == ("mass",)
    assert current["directional_front_lookahead_steps"] == (
        stage25_hysteresis().minimum_active_steps
    )
    assert previous["directional_front_lookahead_steps"] == 0


def test_stage32_is_development_and_reserves_the_blind_mach() -> None:
    config = configuration("workstation")
    assert config["mach"] == 2.0
    assert config["workstation_pass_label"] == "DEVELOPMENT_PASS"
    assert config["stage31_decision_preserved"] == "WORKSTATION_HOLD"
    assert config["blind_validation_mach"] == BLIND_VALIDATION_MACH == 2.5
    assert config["blind_validation_executed"] is False
    assert config["minimum_left_neighbor_front_births"] == 1
    assert config["minimum_weighted_only_front_births"] == 1

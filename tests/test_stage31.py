"""Structural tests for the Stage-31 held-out shock gate."""

from hyqmom_fp import stage25_hysteresis
from riemann35_patch.stage31.run_heldout_shock import PROFILE_NAMES, configuration


def test_stage31_is_held_out_and_keeps_the_frozen_lifecycle() -> None:
    config = configuration("workstation")
    assert config["mach"] == 2.0
    assert config["mach"] != 3.0
    assert config["initial_active_half_width"] == 2
    assert config["sensor_interval_steps"] == 8
    assert config["release_sensor_interval_steps"] == 4
    assert config["causal_activation_candidates_only"] is True
    assert config["kinetic_front_on"] == stage25_hysteresis().tail_on
    assert config["profile_error_limit_percent"] == 3.0


def test_stage31_gates_physical_and_predictive_profiles() -> None:
    assert PROFILE_NAMES == (
        "rho",
        "velocity_x",
        "theta",
        "stress_xx",
        "heat_flux_x",
        "M300",
        "M400",
        "M420",
    )

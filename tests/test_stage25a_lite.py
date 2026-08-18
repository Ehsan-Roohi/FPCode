"""Low-cost structural tests for the Stage-25A lite campaign."""

from riemann35_patch.stage25a.run_normal_shock_lite import METHODS, lite_configuration


def test_lite_configuration_preserves_the_expensive_physics_controls() -> None:
    config = lite_configuration()
    assert config["mach"] == 3.0
    assert config["x_lower"] == -20.0
    assert config["x_upper"] == 20.0
    assert config["velocity_shape"] == (61, 33, 33)
    assert config["v_lower"] == (-12.0, -10.0, -10.0)
    assert config["v_upper"] == (14.0, 10.0, 10.0)
    assert config["cfl"] == 0.35
    assert config["spatial_cells"] == 80
    assert config["steps"] == 600


def test_lite_campaign_has_one_task_per_method() -> None:
    assert METHODS == ("macro", "adaptive", "full_dvm")

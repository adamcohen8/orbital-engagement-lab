from __future__ import annotations

from pathlib import Path

import pytest

SMOKE_TEST_FILES = {
    "test_api_plugin_validation.py",
    "test_app_io.py",
    "test_optional_dependencies.py",
    "test_public_imports.py",
    "test_quickstart_5min.py",
    "test_orbit_integrators.py",
    "test_sensor_measurements.py",
    "test_validation_release_workflow.py",
}

PRODUCT_TEST_FILES = {
    "test_api.py",
    "test_app_services.py",
    "test_game_mode.py",
    "test_output_index.py",
    "test_plotting_public.py",
    "test_product_contracts.py",
    "test_public_imports.py",
    "test_quickstart_5min.py",
    "test_scenario_yaml_config.py",
    "test_sensor_measurements.py",
    "test_orbital_actuator.py",
    "test_validation_release_workflow.py",
}

VALIDATION_TEST_FILES = {
    "test_de440_hpop.py",
    "test_orbit_atmosphere_models.py",
    "test_orbit_eclipse.py",
    "test_orbit_integrators.py",
    "test_orbit_j3_j4.py",
    "test_orbit_planetary_third_body.py",
    "test_orbit_spherical_harmonics.py",
    "test_orbital_actuator.py",
    "test_plugin_validation.py",
    "test_product_contracts.py",
    "test_sensor_measurements.py",
    "test_validation_release_workflow.py",
}

SLOW_TEST_FILES = {
    "test_api.py",
    "test_game_mode.py",
    "test_master_outputs.py",
    "test_master_simulator.py",
    "test_rl_gym_env.py",
    "test_scenario_yaml_config.py",
}

EXTERNAL_TEST_FILES = {
    "test_cfs_sil.py",
    "test_optional_dependencies.py",
}


def _test_filename(item: pytest.Item) -> str:
    return Path(str(item.fspath)).name


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    for item in items:
        filename = _test_filename(item)
        item.add_marker(pytest.mark.regression)
        if filename in SMOKE_TEST_FILES:
            item.add_marker(pytest.mark.smoke)
        if filename in PRODUCT_TEST_FILES:
            item.add_marker(pytest.mark.product)
        if filename in VALIDATION_TEST_FILES:
            item.add_marker(pytest.mark.validation)
        if filename in SLOW_TEST_FILES:
            item.add_marker(pytest.mark.slow)
        if filename in EXTERNAL_TEST_FILES:
            item.add_marker(pytest.mark.external)

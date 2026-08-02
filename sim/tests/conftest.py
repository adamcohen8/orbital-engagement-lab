from __future__ import annotations

from pathlib import Path

import pytest

SMOKE_TEST_FILES = {
    "test_api_plugin_validation.py",
    "test_public_imports.py",
    "test_public_export_check.py",
    "test_quickstart_5min.py",
    "test_orbit_integrators.py",
    "test_sensor_measurements.py",
    "test_validation_release_workflow.py",
}

PRODUCT_TEST_FILES = {
    "test_api.py",
    "test_game_dashboard.py",
    "test_game_launcher_debrief.py",
    "test_game_runtime_input.py",
    "test_game_scenarios_arcade.py",
    "test_game_training.py",
    "test_interchange_phase1.py",
    "test_output_index.py",
    "test_oel_agents.py",
    "test_plotting_public.py",
    "test_product_contracts.py",
    "test_public_export_check.py",
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

# This lane is deliberately opt-in.  It protects the smallest useful set of
# import, architecture, integrator, release-governance, and supply-chain
# contracts without turning "fast" into an alias for the entire regression
# suite.  Keep additions measured and comfortably sub-minute on CI.
FAST_TEST_FILES = {
    "test_api_plugin_validation.py",
    "test_config_api_architecture.py",
    "test_doctor.py",
    "test_interchange_phase1.py",
    "test_orbit_integrators.py",
    "test_public_imports.py",
    "test_platform_compat.py",
    "test_runtime_architecture.py",
    "test_supply_chain_evidence.py",
    "test_test_suite_architecture.py",
    "test_validation_release_workflow.py",
}

# Ordinary behavioral tests default to the Python implementation in CI.  The
# files below own compiled-backend availability, parity, fallback, and exactness
# coverage and therefore run in a dedicated acceleration-enabled shard.
COMPILED_TEST_FILES = {
    "test_acceleration.py",
    "test_de440_hpop.py",
    "test_orbit_atmosphere_models.py",
    "test_orbit_compiled_force_plan.py",
    "test_orbit_spherical_harmonics.py",
}

SLOW_TESTS = {
    ("test_agent_task.py", "test_agent_task_recipe_with_plots_writes_plot_summary"),
    ("test_dynamics_orbit_determination.py", "test_fit_orbit_cli_writes_reviewable_artifacts"),
    ("test_dynamics_orbit_determination.py", "test_attitude_aware_cd_scale_moves_toward_synthetic_truth"),
    ("test_interchange_phase2.py", "test_golden_dynamics_od_workflow_emits_accepted_product_ready_for_onp"),
    (
        "test_dynamics_orbit_determination.py",
        "test_dynamics_orbit_determination_detects_synthetic_maneuver",
    ),
    ("test_game_scenarios_arcade.py", "test_level_eleven_game_attempt_uses_dynamic_history"),
    (
        "test_intent_hypothesis_workflow.py",
        "test_ihe_synthetic_core_v1_generates_and_benchmarks_all_case_families",
    ),
    ("test_intent_hypothesis_workflow.py", "test_evaluation_pack_separates_visible_inputs_from_hidden_truth"),
    ("test_mendicant_dependency_boundary.py", "test_sim_never_imports_optional_mendicant_integration"),
    ("test_mission_recovery.py", "test_orbit_transfer_planner_grid_refinement_approaches_hohmann_transfer"),
    (
        "test_oel_agents.py",
        "test_public_agent_generated_examples_run_headlessly",
    ),
    ("test_oel_agents.py", "test_public_agent_saved_review_queries_execute"),
    ("test_oel_agents.py", "test_public_agent_saved_review_query_cli_runs_query"),
    ("test_oel_agents.py", "test_public_agent_task_card_review_queries_execute"),
    (
        "test_oel_mcp_phase0.py",
        "test_core_never_imports_optional_mcp_and_sdk_dependency_is_bounded",
    ),
    (
        "test_orbit_determination_validation.py",
        "test_precise_orbit_compare_detects_synthetic_maneuver",
    ),
    ("test_orbit_hcw_lqr_convergence.py", "test_converges_for_starts_within_10km_envelope"),
    (
        "test_od_phase8.py",
        "test_phase8_all_models_report_nonlinear_withheld_disagreement_and_chief_uncertainty",
    ),
    ("test_plotting_public.py", "test_payload_plotting_api_writes_expected_artifacts"),
    ("test_plotting_public.py", "test_plot_outputs_expands_public_plot_presets"),
    ("test_public_export_check.py", "test_generated_public_export_has_no_known_private_surfaces"),
    ("test_public_export_check.py", "test_public_export_can_omit_game_music_for_lean_downloads"),
    ("test_public_export_check.py", "test_public_export_cli_runs_without_site_packages"),
    ("test_public_export_check.py", "test_public_export_contains_only_public_mcp_registry"),
    ("test_quickstart_5min.py", "test_quickstart_5min_runs_headlessly_and_writes_start_here_artifacts"),
    ("test_quickstart_5min.py", "test_process_pool_preserves_exact_parity_with_knowledge_coupled_controllers"),
    ("test_quickstart_5min.py", "test_quickstart_process_pool_object_executor_smoke"),
    ("test_reentry.py", "test_aero_assisted_plane_change_demo_runs_and_exits_reentry"),
    ("test_review_store.py", "test_quickstart_config_can_emit_review_store_tables"),
    (
        "test_rocket_insertion_engagement_config.py",
        "test_rocket_insertion_engagement_config_uses_current_engine",
    ),
    (
        "test_rocket_insertion_engagement_config.py",
        "test_rocket_insertion_engagement_initialization_delay_coasts_before_control",
    ),
    ("test_scale_store_od.py", "test_scale_fit_ground_sensor_od_uses_initial_state_prior_paths"),
    ("test_scale_store_od.py", "test_scale_fit_ground_sensor_od_from_store_and_cli"),
    ("test_scale_store_od.py", "test_synthetic_ground_sensor_od_workflow_and_cli"),
    ("test_scale_store_od.py", "test_synthetic_sgp4_sensor_od_workflow_and_cli"),
    ("test_scale_campaigns.py", "test_scale_sensitivity_parallel_runner_can_propagate_deployment_sweep"),
}

EXTERNAL_TEST_FILES = {
    "test_" + "c" + "f" + "s" + "_sil.py",
}

EXTERNAL_TESTS = {
    (
        "test_attitude_actuator_basilisk_validation.py",
        "test_basilisk_reaction_wheel_pd_rate_recovery_error_contracts_when_available",
    ),
    (
        "test_attitude_actuator_basilisk_validation.py",
        "test_basilisk_reaction_wheel_state_history_matches_oel_when_available",
    ),
    (
        "test_attitude_actuator_basilisk_validation.py",
        "test_basilisk_single_axis_reaction_wheel_matches_oel_sign_and_momentum_when_available",
    ),
    (
        "test_attitude_actuator_basilisk_validation.py",
        "test_basilisk_three_axis_reaction_wheels_match_oel_axis_mapping_when_available",
    ),
    (
        "test_attitude_reference_validation.py",
        "test_basilisk_centered_dipole_direct_torque_matches_oel_equation_when_available",
    ),
    (
        "test_attitude_reference_validation.py",
        "test_basilisk_centered_dipole_field_matches_oel_when_available",
    ),
    (
        "test_attitude_reference_validation.py",
        "test_basilisk_exponential_density_matches_oel_default_when_available",
    ),
    (
        "test_attitude_reference_validation.py",
        "test_basilisk_exponential_drag_direct_force_torque_matches_oel_equation_when_available",
    ),
    (
        "test_attitude_reference_validation.py",
        "test_basilisk_runner_has_clean_optional_dependency_boundary",
    ),
}


def _test_filename(item: pytest.Item) -> str:
    return Path(str(item.fspath)).name


def _test_function_name(item: pytest.Item) -> str:
    return str(item.name).split("[", 1)[0]


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
        if filename in FAST_TEST_FILES:
            item.add_marker(pytest.mark.fast)
        if filename in COMPILED_TEST_FILES:
            item.add_marker(pytest.mark.compiled)
        if (filename, _test_function_name(item)) in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
        if filename in EXTERNAL_TEST_FILES or (filename, _test_function_name(item)) in EXTERNAL_TESTS:
            item.add_marker(pytest.mark.external)

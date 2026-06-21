from sim.config.help import (
    CONFIG_HELP_ENTRIES,
    find_config_help,
    format_config_help,
    format_config_help_list,
    load_config_help_context,
)


def _has_pro_config_help() -> bool:
    return any(entry.path == "analysis.study_type" for entry in CONFIG_HELP_ENTRIES)


def test_config_help_fuzzy_matches_ephemeris_typo():
    matches = find_config_help("emphemeris model")

    assert matches
    assert matches[0][1].path == "simulator.environment.ephemeris_mode"

    text = format_config_help("emphemeris model")
    assert "analytic_enhanced" in text
    assert "spice" in text


def test_config_help_lists_dynamic_plot_topics():
    text = format_config_help("plot preset")

    assert "outputs.plots.preset" in text
    assert "rendezvous" in text


def test_config_help_includes_ground_access_and_mass_property_aliases():
    ground_text = format_config_help("ground access")

    assert "ground_stations" in ground_text
    assert "min_elevation_deg" in ground_text

    mass_text = format_config_help("mass properties")

    assert "objects.<id>.specs.mass_properties" in mass_text
    assert "inertia_reference_point" in mass_text


def test_config_help_includes_controller_bench_workflow_alias():
    text = format_config_help("controller bench")

    assert "controller_bench" in text
    assert "--controller-bench" in text
    assert "docs/controller-bench.md" in text


def test_config_help_routes_reentry_and_deorbit_terms():
    deorbit_text = format_config_help("deorbit")

    assert "simulator.dynamics.reentry" in deorbit_text
    assert "configs/reentry_smoke.yaml" in deorbit_text

    breakup_text = format_config_help("reentry breakup")

    assert "breakup debris" in breakup_text
    assert "not a breakup" in breakup_text


def test_config_help_routes_sensor_error_and_modeled_sensor_terms():
    sensor_text = format_config_help("sensor error")

    assert "objects.<id>.knowledge.sensor_error" in sensor_text
    assert "pos_sigma_km" in sensor_text
    assert "knowledge.sensor" in sensor_text

    optical_text = format_config_help("optical camera")

    assert "not an optical/radar camera hardware model" in optical_text
    assert "limiting magnitude" in optical_text


def test_config_help_routes_central_body_and_non_earth_terms():
    central_text = format_config_help("central body")

    assert "simulator.dynamics.orbit.model" in central_text
    assert "Earth-centered" in central_text
    assert "central_body" in central_text

    mars_text = format_config_help("mars orbit")

    assert "CR3BP" in mars_text
    assert "mu_km3_s2" in mars_text


def test_config_help_routes_rocket_wind_and_weather_terms():
    wind_text = format_config_help("crosswind")

    assert "simulator.dynamics.rocket.wind_enu_m_s" in wind_text
    assert "configs/controller_bench_rocket_case_wind.yaml" in wind_text

    weather_text = format_config_help("weather")

    assert "not a weather forecast" in weather_text
    assert "wind_enu_m_s" in weather_text


def test_config_help_list_includes_field_paths():
    text = format_config_help_list()

    assert "simulator.environment.ephemeris_mode" in text
    assert "outputs.plots.figure_ids" in text


def test_config_help_can_filter_pro_topics_from_public_scope():
    all_text = format_config_help_list()
    public_text = format_config_help_list(scope="public")

    assert "analysis.study_type" not in public_text
    assert "monte_carlo.variations.mode" not in public_text
    assert "simulator.environment.ephemeris_mode" in public_text

    assert find_config_help("sensitivity method", scope="public") == []
    if _has_pro_config_help():
        assert "analysis.study_type" in all_text
        assert "monte_carlo.variations.mode" in all_text
    else:
        assert "analysis.study_type" not in all_text
        assert "monte_carlo.variations.mode" not in all_text


def test_config_help_can_show_current_config_value(tmp_path):
    cfg_path = tmp_path / "example.yaml"
    cfg_path.write_text(
        """
simulator:
  environment:
    ephemeris_mode: analytic_simple
""",
        encoding="utf-8",
    )

    data = load_config_help_context(cfg_path)
    text = format_config_help("emphemeris model", config_data=data, config_path=cfg_path)

    assert "Current Config" in text
    assert "simulator.environment.ephemeris_mode: analytic_simple" in text


def test_config_help_context_handles_placeholder_and_list_paths():
    data = {
        "objects": {
            "target": {
                "kind": "satellite",
                "specs": {
                    "mass_properties": {
                        "center_of_mass_body_m": [0.0, 0.0, 0.0],
                        "inertia_reference_point": "center_of_mass",
                    }
                },
            },
            "rocket": {"kind": "rocket"},
        },
        "ground_stations": [{"id": "colorado_springs", "min_elevation_deg": 10.0}],
        "monte_carlo": {
            "variations": [
                {"parameter_path": "simulator.dt_s", "mode": "choice"},
                {"parameter_path": "target.specs.mass_kg", "mode": "uniform"},
            ]
        },
    }

    object_text = format_config_help("object kind", config_data=data)

    assert "objects.target.kind: satellite" in object_text
    assert "objects.rocket.kind: rocket" in object_text
    ground_text = format_config_help("ground access", config_data=data)
    assert "ground_stations: [{id: colorado_springs, min_elevation_deg: 10.0}]" in ground_text
    mass_text = format_config_help("runtime inertia", config_data=data)
    assert "objects.target.specs.mass_properties:" in mass_text
    assert "inertia_reference_point: center_of_mass" in mass_text
    if _has_pro_config_help():
        mc_text = format_config_help("variation mode", config_data=data)
        assert "monte_carlo.variations[0].mode: choice" in mc_text
        assert "monte_carlo.variations[1].mode: uniform" in mc_text

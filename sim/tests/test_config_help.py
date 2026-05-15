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
            "target": {"kind": "satellite"},
            "rocket": {"kind": "rocket"},
        },
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
    if _has_pro_config_help():
        mc_text = format_config_help("variation mode", config_data=data)
        assert "monte_carlo.variations[0].mode: choice" in mc_text
        assert "monte_carlo.variations[1].mode: uniform" in mc_text

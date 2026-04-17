from __future__ import annotations

from pathlib import Path
import tempfile

from sim.app.services import (
    dump_config_text,
    get_default_config_path,
    get_gui_capabilities,
    list_available_configs,
    load_config,
    run_config_via_api,
    summarize_config,
    validate_config,
)


def test_default_config_available() -> None:
    assert get_default_config_path() in list_available_configs()


def test_load_validate_and_summarize_default_config() -> None:
    cfg_dict = load_config(get_default_config_path())
    cfg = validate_config(cfg_dict)
    summary = summarize_config(cfg)
    assert summary.scenario_name == cfg.scenario_name
    assert summary.output_dir == cfg.outputs.output_dir
    assert summary.analysis_enabled is False
    assert summary.analysis_study_type == "single_run"


def test_dump_config_text_contains_scenario_name() -> None:
    cfg_dict = load_config(get_default_config_path())
    text = dump_config_text(cfg_dict)
    assert "scenario_name" in text


def test_public_gui_capabilities_hide_pro_analysis_workflows() -> None:
    caps = get_gui_capabilities()

    assert "BASIC_SATELLITE" in caps.satellite_presets
    assert "BASIC_TWO_STAGE_STACK" in caps.rocket_preset_stacks
    assert caps.analysis_study_types == []
    assert caps.sensitivity_methods == []
    assert caps.monte_carlo_modes == []
    assert caps.monte_carlo_lhs_modes == []
    assert caps.monte_carlo_parameter_categories == {}
    assert caps.analysis_ui_profiles == {}
    assert "run_dashboard" in caps.figure_ids
    assert "rendezvous_summary" in caps.figure_ids
    assert "sensor_access" in caps.figure_ids
    assert "ground_track_multi" in caps.animation_types


def test_run_config_via_api_executes_single_run() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "gui_service_run.yaml"
        cfg_path.write_text(
            dump_config_text(
                {
                    "scenario_name": "gui_service_run",
                    "rocket": {"enabled": False},
                    "chaser": {"enabled": False},
                    "target": {
                        "enabled": True,
                        "specs": {"mass_kg": 100.0},
                        "initial_state": {
                            "position_eci_km": [7000.0, 0.0, 0.0],
                            "velocity_eci_km_s": [0.0, 7.5, 0.0],
                        },
                    },
                    "simulator": {
                        "duration_s": 2.0,
                        "dt_s": 1.0,
                        "termination": {"earth_impact_enabled": False},
                        "dynamics": {"attitude": {"enabled": False}},
                    },
                    "outputs": {
                        "output_dir": str(Path(tmpdir) / "outputs"),
                        "mode": "save",
                        "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                        "plots": {"enabled": False, "figure_ids": []},
                        "animations": {"enabled": False, "types": []},
                    },
                }
            ),
            encoding="utf-8",
        )

        result = run_config_via_api(cfg_path)

        assert result.returncode == 0
        assert result.scenario_name == "gui_service_run"
        assert "Scenario: gui_service_run" in result.stdout

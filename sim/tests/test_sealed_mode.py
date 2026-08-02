from __future__ import annotations

import pytest

from sim.api import SimulationSession, SimulationWorkspace
from sim.config import scenario_config_from_dict
from sim.security.sealed_mode import SealedModePolicy, validate_sealed_mode


def _base_config() -> dict:
    return {
        "scenario_name": "sealed_mode_test",
        "simulator": {"duration_s": 10.0, "dt_s": 1.0},
        "objects": {
            "target": {
                "enabled": True,
                "orbit_control": {
                    "module": "sim.control.orbit.zero_controller",
                    "class_name": "ZeroController",
                },
            }
        },
        "outputs": {
            "mode": "save",
            "stats": {"print_summary": False, "save_json": True, "save_full_log": False},
        },
    }


def _sealed_errors(root: dict, policy: SealedModePolicy | None = None) -> list[str]:
    return validate_sealed_mode(scenario_config_from_dict(root), policy)


def test_sealed_mode_allows_builtin_sim_plugin_modules() -> None:
    assert _sealed_errors(_base_config()) == []


def test_sealed_mode_blocks_untrusted_plugin_modules() -> None:
    root = _base_config()
    root["objects"]["target"]["orbit_control"]["module"] = "custom_plugins.controller"

    errors = _sealed_errors(root)

    assert any("blocks plugin module 'custom_plugins.controller'" in err for err in errors)
    assert _sealed_errors(root, SealedModePolicy(allow_untrusted_plugin_imports=True)) == []


def test_sealed_mode_blocks_hosted_ai_and_custom_endpoint() -> None:
    root = _base_config()
    root["outputs"]["ai_report"] = {
        "enabled": True,
        "provider": "openai",
        "model": "gpt-5-mini",
        "endpoint": "https://proxy.example.test/v1",
    }

    errors = _sealed_errors(root)

    assert any("blocks hosted AI provider 'openai'" in err for err in errors)
    assert any("blocks custom AI endpoint" in err for err in errors)
    assert _sealed_errors(
        root,
        SealedModePolicy(allow_hosted_ai=True, allow_custom_ai_endpoints=True),
    ) == []


def test_sealed_mode_treats_ai_config_as_enabled_by_default() -> None:
    root = _base_config()
    root["outputs"]["ai_config"] = {
        "provider": "openai",
        "model": "gpt-5-mini",
    }

    errors = _sealed_errors(root)

    assert any("outputs.ai_config.provider" in err and "blocks hosted AI provider 'openai'" in err for err in errors)


def test_sealed_mode_allows_ai_config_dry_run_without_hosted_ai_opt_in() -> None:
    root = _base_config()
    root["outputs"]["ai_config"] = {
        "enabled": True,
        "dry_run": True,
        "provider": "openai",
        "model": "gpt-5-mini",
    }

    assert _sealed_errors(root) == []


def test_sealed_mode_blocks_non_loopback_sil_networking() -> None:
    root = _base_config()
    root["objects"]["target"]["bridge"] = {
        "enabled": True,
        "module": "sim.bridge.local",
        "class_name": "CfsSilExternalIntentBridge",
        "params": {
            "bind_host": "0.0.0.0",
            "cfs_host": "192.0.2.10",
            "allow_non_loopback": True,
        },
    }

    errors = _sealed_errors(root)

    assert any("blocks non-loopback" in err and "UDP networking" in err for err in errors)
    assert _sealed_errors(root, SealedModePolicy(allow_non_loopback_sil=True)) == []


def test_sealed_mode_blocks_implicit_gravity_model_downloads() -> None:
    root = _base_config()
    root["simulator"]["dynamics"] = {
        "orbit": {
            "spherical_harmonics": {
                "enabled": True,
                "degree": 8,
                "order": 8,
                "source": "egm96",
            }
        }
    }

    errors = _sealed_errors(root)

    assert any("spherical_harmonics.allow_download" in err and "blocks gravity-model downloads" in err for err in errors)
    root["simulator"]["dynamics"]["orbit"]["spherical_harmonics"]["allow_download"] = False
    assert _sealed_errors(root) == []
    root["simulator"]["dynamics"]["orbit"]["spherical_harmonics"]["allow_download"] = True
    assert _sealed_errors(root, SealedModePolicy(allow_gravity_model_downloads=True)) == []


def test_sealed_mode_blocks_high_detail_retention() -> None:
    root = _base_config()
    root["outputs"]["stats"]["save_full_log"] = True
    root["outputs"]["review"] = {"enabled": True, "detail": "full"}
    root["outputs"]["monte_carlo"] = {"save_raw_runs": True}
    root["outputs"]["ai_report"] = {
        "enabled": True,
        "provider": "ollama",
        "model": "local-model",
        "data_scope": "full",
    }

    errors = _sealed_errors(root)

    assert any("save_full_log" in err for err in errors)
    assert any("outputs.review.detail" in err for err in errors)
    assert any("save_raw_runs" in err for err in errors)
    assert any("outputs.ai_report.data_scope" in err for err in errors)
    assert _sealed_errors(root, SealedModePolicy(allow_high_detail_outputs=True)) == []


def test_simulation_session_enforces_sealed_mode_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _base_config()
    root["objects"]["target"]["orbit_control"]["module"] = "custom_plugins.controller"
    monkeypatch.setenv("OEL_SEALED_MODE", "1")

    with pytest.raises(ValueError, match="Sealed mode validation failed"):
        SimulationSession.from_config(root)


def test_simulation_session_blocks_runtime_overrides_in_sealed_mode() -> None:
    session = SimulationSession.from_config(_base_config(), sealed_mode=True)

    with pytest.raises(PermissionError, match="controller overrides"):
        session.set_orbit_controller("target", object())

    with pytest.raises(PermissionError, match="external intent providers"):
        session.set_external_intent_provider("target", lambda **_: None)


def test_simulation_workspace_enforces_explicit_sealed_mode() -> None:
    root = _base_config()
    root["outputs"]["stats"]["save_full_log"] = True
    workspace = SimulationWorkspace(sealed_mode=True)

    with pytest.raises(ValueError, match="save_full_log"):
        workspace.from_dict(root)

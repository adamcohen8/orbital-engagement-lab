from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_GAME_RUNTIME_FIELDS = {
    "flight_software_stack",
    "input_profile",
    "hardware_profile",
    "observer_policy",
    "scoring_policy",
    "replay_support",
}


def test_every_maintained_game_level_declares_its_v2_runtime_contract() -> None:
    config_dir = Path(__file__).parents[1] / "game" / "configs"
    for path in sorted(config_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        game = dict(raw.get("metadata", {}).get("game", {}) or {})
        assert REQUIRED_GAME_RUNTIME_FIELDS <= set(game), path.name
        assert game["flight_software_stack"] == "fsw.game_pilot_reference"
        assert game["observer_policy"] in {"truth_assisted", "onboard_only", "hybrid"}
        expected_replay = (
            "none"
            if path.name in {
                "game_mode_basic.yaml",
                "game_training_rpo_arcade_pursuit.yaml",
                "game_training_rpo_sandbox.yaml",
            }
            else "debrief_history_v1"
        )
        assert game["replay_support"] == expected_replay


def test_maintained_game_levels_do_not_enable_legacy_object_knowledge() -> None:
    config_dir = Path(__file__).parents[1] / "game" / "configs"
    for path in sorted(config_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        for object_id, object_config in dict(raw.get("objects", {}) or {}).items():
            assert "knowledge" not in dict(object_config or {}), (
                f"{path.name}: object {object_id!r} must use its declared v2 FSW "
                "navigation path instead of the legacy simulator knowledge subsystem"
            )

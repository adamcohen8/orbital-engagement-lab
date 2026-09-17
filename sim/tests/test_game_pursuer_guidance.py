"""The autonomous Trainer pursuer must track the flagship orbital guidance."""
from pathlib import Path

import yaml


def test_evasion_pursuer_matches_flagship_translation_guidance():
    root = Path(__file__).resolve().parents[2]
    flagship = yaml.safe_load((root / 'configs/ric_pd_10km_experiment.yaml').read_text())
    level = yaml.safe_load((root / 'sim/game/configs/game_training_rpo_11_evasive_target_survival.yaml').read_text())
    expected = flagship['objects']['chaser']['flight_software']['params']
    actual = level['objects']['chaser']['flight_software']['params']
    attitude = {key for key in expected if key.startswith('attitude_')} | {
        'max_attitude_torque_n_m', 'pointing_tolerance_rad', 'require_pointing_for_translation',
    }
    for key in expected.keys() - attitude:
        assert actual[key] == expected[key], f'Pursuer drifted from flagship: {key}'
    assert actual['require_pointing_for_translation'] is False
    assert actual['navigation_initialization'] == 'cold'

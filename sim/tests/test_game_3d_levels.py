"""Desktop catalog coverage and constraint mesh smoke checks."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from sim.api import SimulationConfig
from sim.game.dashboard_3d import SandboxCamera
from sim.game.runner_config import _game_3d_enabled
from sim.game.training import RPOTrainingConfig

CONFIGS = Path(__file__).resolve().parents[1] / "game/configs"
EXCLUDED = {"game_training_rpo_bonus_cislunar_rendezvous.yaml", "game_training_rpo_bonus_drag_racing.yaml"}


def test_space_force_3d_in_track_basis_mirrors_positive_i():
    camera = SandboxCamera()
    oel_positive_i_x = float(np.array([0.0, 1.0, 0.0]) @ camera.basis()[0])
    camera.in_track_sign = -1
    space_force_positive_i_x = float(np.array([0.0, 1.0, 0.0]) @ camera.basis()[0])
    assert oel_positive_i_x > 0
    assert space_force_positive_i_x == pytest.approx(-oel_positive_i_x)


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda path: path.stem)
def test_catalog_3d_availability_and_render(path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pytest.importorskip("pygame")
    from sim.game.constraint_mesh import region_mesh
    from sim.game.dashboard import PygameRPODashboard
    from sim.game.session import GamePhysicsSession

    config = SimulationConfig.from_yaml(path)
    assert _game_3d_enabled(config) == (path.name not in EXCLUDED)
    if path.name in EXCLUDED:
        return
    training = RPOTrainingConfig.from_metadata(config.scenario.metadata)
    for region in training.forbidden_regions:
        faces, lines = region_mesh(region)
        assert faces and lines
        assert all(np.all(np.isfinite(row)) and row.shape[1] == 3 for row in faces + lines)
    kwargs = {
        field.name: getattr(training, field.name)
        for field in fields(PygameRPODashboard)
        if hasattr(training, field.name)
    }
    dashboard = PygameRPODashboard(**kwargs, fullscreen=False, sandbox_3d_enabled=True)
    try:
        dashboard.push_snapshot(GamePhysicsSession(config).reset())
        dashboard._sandbox_camera().enabled = True
        dashboard.draw(level_title=path.stem)
        assert "2D" in dashboard._camera_buttons
        # All views remain navigable after changing perspective.
        dashboard._sandbox_camera().yaw += 0.8
        dashboard.draw(level_title=path.stem)
    finally:
        dashboard.close()

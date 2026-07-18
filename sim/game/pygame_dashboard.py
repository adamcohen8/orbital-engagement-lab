# ruff: noqa: F401,F821,I001
"""Compatibility façade for the Pygame RPO dashboard."""

from . import dashboard_common as _common
from . import geometry as _geometry
from . import prediction as _prediction
from . import camera as _camera
from . import dashboard_state as _state
from . import dashboard_layout as _layout
from . import dashboard_prediction as _prediction_mixin
from . import dashboard_hud as _hud
from . import dashboard_overlays as _overlays
from . import dashboard_camera as _camera_mixin
from . import dashboard_text as _text
from .dashboard import PygameRPODashboard

DASHBOARD_CAPABILITY_FAMILIES = {
    "state": "sim.game.dashboard_state",
    "layout": "sim.game.dashboard_layout",
    "prediction": "sim.game.dashboard_prediction",
    "panels": "sim.game.dashboard_layout",
    "hud": "sim.game.dashboard_hud",
    "overlays": "sim.game.dashboard_overlays",
    "camera": "sim.game.dashboard_camera",
    "geometry": "sim.game.geometry",
}

for _module in (_common, _geometry, _prediction, _camera, _state, _layout, _prediction_mixin, _hud, _overlays, _camera_mixin, _text):
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

PygameRPODashboard.__module__ = __name__
del _module

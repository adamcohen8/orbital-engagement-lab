# ruff: noqa: F401,F821,I001
"""Compatibility façade for the RPO trainer run loop."""

from . import runner_common as _common
from . import runner_models as _models
from . import runner_config as _config
from . import tutorial_runtime as _tutorial
from . import recording_runtime as _recording
from . import attempt_lifecycle as _attempt
from . import mission_metrics as _metrics
from . import game_loop as _loop

RUNNER_CAPABILITY_FAMILIES = {
    "models": "sim.game.runner_models",
    "config": "sim.game.runner_config",
    "tutorials": "sim.game.tutorial_runtime",
    "recording": "sim.game.recording_runtime",
    "attempt_lifecycle": "sim.game.attempt_lifecycle",
    "metrics": "sim.game.mission_metrics",
    "loop": "sim.game.game_loop",
}

for _module in (_common, _models, _config, _tutorial, _recording, _attempt, _metrics, _loop):
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

for _name in (
    "GameRunResult", "GuidedTutorialRuntime", "RICPrimerRuntime", "OperatorTutorialStage",
    "OperatorTutorialRuntime", "OperatorBurnCinematicRuntime", "SandboxSetupValues",
):
    globals()[_name].__module__ = __name__
del _module, _name

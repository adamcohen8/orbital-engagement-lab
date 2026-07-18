# ruff: noqa: F401,F821,I001
"""Compatibility façade for game training configuration, history, and scoring."""

from . import training_models as _models
from . import scoring as _scoring
from . import training_geometry as _geometry
from . import coaching as _coaching
from . import criteria as _criteria
from . import training_history as _history

TRAINING_CAPABILITY_FAMILIES = {
    "models": "sim.game.training_models",
    "history": "sim.game.training_history",
    "criteria": "sim.game.criteria",
    "scoring": "sim.game.scoring",
    "coaching": "sim.game.coaching",
    "geometry": "sim.game.training_geometry",
}

for _module in (_models, _scoring, _geometry, _coaching, _criteria, _history):
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

for _name in (
    "ForbiddenRegionConfig", "ApproachGateConfig", "InspectionGateConfig",
    "SunAngleConstraintConfig", "RequiredPhaseBurnConfig", "GuidedTutorialBurnConfig",
    "GuidedTutorialSpeedStepConfig", "RPOTrainingConfig", "RPOTrainingScore", "RPOTrainingTracker",
):
    globals()[_name].__module__ = __name__
del _module, _name

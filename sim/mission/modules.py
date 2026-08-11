# ruff: noqa: F401,I001
"""Compatibility façade for OEL mission strategies and executions.

Implementations live in focused modules under :mod:`sim.mission.strategies`
and :mod:`sim.mission.execution`. Existing imports remain supported.
"""

from .strategies import base as _base
from .strategies import rocket as _rocket
from .strategies import satellite as _satellite
from . import executive as _executive
from .execution import burns as _burns
from .execution import integrated as _integrated
from .execution import pointing as _pointing
from .execution import safe_hold as _safe_hold
from .execution import reference_commands as _reference_commands
from . import legacy_modules as _legacy
from .registries import MISSION_EXECUTION_FAMILIES, MISSION_STRATEGY_FAMILIES

for _module in (
    _base,
    _satellite,
    _executive,
    _rocket,
    _pointing,
    _burns,
    _integrated,
    _safe_hold,
    _reference_commands,
    _legacy,
):
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

for _name, _value in tuple(globals().items()):
    if isinstance(_value, type) and _value.__module__.startswith("sim.mission."):
        _value.__module__ = __name__

del _module, _name, _value

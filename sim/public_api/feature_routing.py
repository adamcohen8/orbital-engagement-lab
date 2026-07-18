from __future__ import annotations

import importlib
import sys
from typing import Any

from sim.scenarios import ScenarioBuilder as ScenarioBuilder
from sim.scenarios import ValidationIssue as ValidationIssue


def _require_private_workflow(module_name: str, symbol_name: str, feature: str) -> Any:
    facade = sys.modules.get("sim.api")
    override = getattr(facade, "_require_private_workflow", None)
    if override is not None and override is not _require_private_workflow:
        return override(module_name, symbol_name, feature)
    try:
        module = importlib.import_module(module_name)
        symbol = getattr(module, symbol_name)
    except Exception as exc:
        raise ImportError(f"{feature} are available in the private/product distribution.") from exc
    if getattr(symbol, "__name__", "") == "_unavailable":
        symbol()
    return symbol

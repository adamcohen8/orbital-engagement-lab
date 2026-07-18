"""Stable public Python API façade."""

from __future__ import annotations

import importlib as importlib
import inspect as inspect
import json as json
import warnings as warnings
from collections.abc import Iterable as Iterable
from collections.abc import Mapping as Mapping
from dataclasses import dataclass as dataclass
from pathlib import Path as Path
from typing import Any as Any
from typing import Callable as Callable
from typing import Union as Union

import numpy as np  # noqa: F401 - historical public module attribute

from sim.config import SimulationScenarioConfig as SimulationScenarioConfig
from sim.config import enabled_object_ids as enabled_object_ids
from sim.config import load_simulation_yaml as load_simulation_yaml
from sim.config import scenario_config_from_dict as scenario_config_from_dict
from sim.config import validate_scenario_plugins as validate_scenario_plugins
from sim.core.models import Command as Command
from sim.core.models import StateBelief as StateBelief
from sim.execution import create_single_run_engine as create_single_run_engine
from sim.execution import run_simulation_scenario as run_simulation_scenario
from sim.execution.study import analysis_study_type as analysis_study_type
from sim.execution.validation import validate_generated_batch_configs as validate_generated_batch_configs
from sim.public_api.config import ControllerFactory as ControllerFactory
from sim.public_api.config import MetricCallback as MetricCallback
from sim.public_api.config import SimulationConfig as SimulationConfig
from sim.public_api.config import _api_sealed_policy as _api_sealed_policy
from sim.public_api.config import _canonicalize_api_config_dict as _canonicalize_api_config_dict
from sim.public_api.controller_adapters import _CallableControllerAdapter as _CallableControllerAdapter
from sim.public_api.controller_adapters import _CallableMissionAdapter as _CallableMissionAdapter
from sim.public_api.controller_adapters import _coerce_controller_return as _coerce_controller_return
from sim.public_api.controller_adapters import _compatible_call as _compatible_call
from sim.public_api.controller_adapters import _controller_object as _controller_object
from sim.public_api.controller_adapters import _mission_object as _mission_object
from sim.public_api.feature_routing import _require_private_workflow as _require_private_workflow
from sim.public_api.results import _ECI_REL_STATE_COLUMNS as _ECI_REL_STATE_COLUMNS
from sim.public_api.results import _RIC_STATE_COLUMNS as _RIC_STATE_COLUMNS
from sim.public_api.results import _STATE_COLUMNS as _STATE_COLUMNS
from sim.public_api.results import MetricStudyResult as MetricStudyResult
from sim.public_api.results import SimulationResult as SimulationResult
from sim.public_api.results import _aggregate_custom_metrics as _aggregate_custom_metrics
from sim.public_api.results import _artifact_paths as _artifact_paths
from sim.public_api.results import _as_2d_state_history as _as_2d_state_history
from sim.public_api.results import _as_array_map as _as_array_map
from sim.public_api.results import _as_nested_array_map as _as_nested_array_map
from sim.public_api.results import _closest_approach_metric as _closest_approach_metric
from sim.public_api.results import _evaluate_metric_callbacks as _evaluate_metric_callbacks
from sim.public_api.results import _json_safe_metric_value as _json_safe_metric_value
from sim.public_api.results import _metric_callback_name as _metric_callback_name
from sim.public_api.results import _numeric_metric_value as _numeric_metric_value
from sim.public_api.results import _range_scale as _range_scale
from sim.public_api.results import _records_dataframe as _records_dataframe
from sim.public_api.session import HostedSimulationSession as HostedSimulationSession
from sim.public_api.session import SimulationSession as SimulationSession
from sim.public_api.session import TrustedSimulationSession as TrustedSimulationSession
from sim.public_api.snapshots import SimulationSnapshot as SimulationSnapshot
from sim.public_api.workspace import HostedSimulationWorkspace as HostedSimulationWorkspace
from sim.public_api.workspace import SimulationWorkspace as SimulationWorkspace
from sim.public_api.workspace import TrustedSimulationWorkspace as TrustedSimulationWorkspace
from sim.scenarios import ScenarioArtifact as ScenarioArtifact
from sim.scenarios import ScenarioBuilder as ScenarioBuilder
from sim.scenarios import ValidationIssue as ValidationIssue
from sim.scenarios import ValidationReport as ValidationReport
from sim.security import ConfigPathPolicy as ConfigPathPolicy
from sim.security.sealed_mode import SealedModePolicy as SealedModePolicy
from sim.security.sealed_mode import sealed_mode_enabled as sealed_mode_enabled
from sim.security.sealed_mode import validate_sealed_mode as validate_sealed_mode

# Keep repr/pickle/import identity compatible with the historical façade.
SimulationConfig.__module__ = __name__
SimulationSnapshot.__module__ = __name__
_CallableControllerAdapter.__module__ = __name__
_CallableMissionAdapter.__module__ = __name__
MetricStudyResult.__module__ = __name__
SimulationResult.__module__ = __name__
SimulationSession.__module__ = __name__
HostedSimulationSession.__module__ = __name__
SimulationWorkspace.__module__ = __name__
HostedSimulationWorkspace.__module__ = __name__

"""Compatibility façade for OEL runtime construction helpers.

Implementations live in :mod:`sim.runtime`.  Existing imports remain valid,
including the historical ``EARTH_MU_KM3_S2`` monkeypatch seam used by tests and
legacy integrations.
"""
# ruff: noqa: F401,I001

from __future__ import annotations

import logging
from functools import wraps
from typing import Any

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.elements import coe_to_rv_eci as _coe_to_rv_eci
from sim.runtime import knowledge_factory as _knowledge_factory
from sim.runtime import rocket_factory as _rocket_factory
from sim.runtime import satellite_factory as _satellite_factory
from sim.runtime import state_initialization as _state_initialization
from sim.runtime.actuator_factory import (
    _angle_value_rad as _angle_value_rad,
)
from sim.runtime.actuator_factory import (
    _apply_thruster_mount_defaults as _apply_thruster_mount_defaults,
)
from sim.runtime.actuator_factory import (
    _array_or_none as _array_or_none,
)
from sim.runtime.actuator_factory import (
    _build_electric_propulsion as _build_electric_propulsion,
)
from sim.runtime.actuator_factory import (
    _build_gimbaled_thruster as _build_gimbaled_thruster,
)
from sim.runtime.actuator_factory import (
    _build_rcs_cluster as _build_rcs_cluster,
)
from sim.runtime.actuator_factory import (
    _build_reaction_wheels as _build_reaction_wheels,
)
from sim.runtime.actuator_factory import (
    _build_satellite_actuator_stack_from_specs as _build_satellite_actuator_stack_from_specs,
)
from sim.runtime.actuator_factory import (
    _initial_state_nonnegative_float as _initial_state_nonnegative_float,
)
from sim.runtime.actuator_factory import (
    _resolve_satellite_inertia_kg_m2 as _resolve_satellite_inertia_kg_m2,
)
from sim.runtime.actuator_factory import (
    _resolve_satellite_isp_s as _resolve_satellite_isp_s,
)
from sim.runtime.actuator_factory import (
    _satellite_spec_float as _satellite_spec_float,
)
from sim.runtime.actuator_factory import (
    _satellite_spec_vector3 as _satellite_spec_vector3,
)
from sim.runtime.commands import (
    _attitude_state13_from_belief as _attitude_state13_from_belief,
)
from sim.runtime.commands import (
    _combine_commands as _combine_commands,
)
from sim.runtime.commands import (
    _command_to_dict as _command_to_dict,
)
from sim.runtime.commands import (
    _deep_set as _deep_set,
)
from sim.runtime.commands import (
    _relative_orbit_state12 as _relative_orbit_state12,
)
from sim.runtime.commands import (
    _rocket_state_to_truth as _rocket_state_to_truth,
)
from sim.runtime.commands import (
    _sample_variation as _sample_variation,
)
from sim.runtime.commands import (
    _to_jsonable_value as _to_jsonable_value,
)
from sim.runtime.commands import (
    _truth_from_state6 as _truth_from_state6,
)
from sim.runtime.commands import (
    _truth_state6 as _truth_state6,
)
from sim.runtime.compat import (
    _cached_compatibility_plan as _cached_compatibility_plan,
)
from sim.runtime.compat import (
    _call_with_compat_kwargs as _call_with_compat_kwargs,
)
from sim.runtime.compat import (
    _compatibility_plan as _compatibility_plan,
)
from sim.runtime.compat import (
    _compatible_keyword_args as _compatible_keyword_args,
)
from sim.runtime.compat import (
    _module_obj as _module_obj,
)
from sim.runtime.knowledge_factory import (
    _knowledge_ekf_diag as _knowledge_ekf_diag,
)
from sim.runtime.knowledge_factory import (
    _knowledge_maneuver_detection_config as _knowledge_maneuver_detection_config,
)
from sim.runtime.mission_runtime import (
    _deploy_from_rocket as _deploy_from_rocket,
)
from sim.runtime.mission_runtime import (
    _run_mission_execution as _run_mission_execution,
)
from sim.runtime.mission_runtime import (
    _run_mission_modules as _run_mission_modules,
)
from sim.runtime.mission_runtime import (
    _run_mission_strategy as _run_mission_strategy,
)
from sim.runtime.models import AgentRuntime as AgentRuntime
from sim.runtime.rocket_factory import (
    _build_rocket_guidance as _build_rocket_guidance,
)
from sim.runtime.rocket_factory import (
    _earth_impact_policy_for_object as _earth_impact_policy_for_object,
)
from sim.runtime.rocket_factory import (
    _orbital_elements_basic as _orbital_elements_basic,
)
from sim.runtime.rocket_factory import (
    _resolve_rocket_stack as _resolve_rocket_stack,
)
from sim.runtime.rocket_factory import (
    _rocket_altitude_km as _rocket_altitude_km,
)
from sim.runtime.satellite_factory import (
    _build_orbit_propagator as _build_orbit_propagator,
)
from sim.runtime.satellite_factory import (
    _geometry_profile_path_from_specs as _geometry_profile_path_from_specs,
)
from sim.runtime.satellite_factory import (
    _load_geometry_area_profile_from_specs as _load_geometry_area_profile_from_specs,
)
from sim.runtime.satellite_factory import (
    _scenario_uses_aerodynamic_lift as _scenario_uses_aerodynamic_lift,
)
from sim.runtime.state_initialization import (
    _apply_chaser_relative_init_from_target as _apply_chaser_relative_init_from_target,
)
from sim.runtime.state_initialization import (
    _apply_relative_cislunar_init_from_reference as _apply_relative_cislunar_init_from_reference,
)
from sim.runtime.state_initialization import (
    _apply_relative_init_from_reference as _apply_relative_init_from_reference,
)
from sim.runtime.state_initialization import (
    _resolve_chaser_relative_ric_init as _resolve_chaser_relative_ric_init,
)
from sim.runtime.state_initialization import (
    _resolve_relative_cislunar_init as _resolve_relative_cislunar_init,
)

logger = logging.getLogger(__name__)


def _sync_environment_constants() -> None:
    _state_initialization.EARTH_MU_KM3_S2 = EARTH_MU_KM3_S2
    _satellite_factory.EARTH_MU_KM3_S2 = EARTH_MU_KM3_S2
    _knowledge_factory.EARTH_MU_KM3_S2 = EARTH_MU_KM3_S2
    _rocket_factory.EARTH_MU_KM3_S2 = EARTH_MU_KM3_S2
    _rocket_factory.EARTH_RADIUS_KM = EARTH_RADIUS_KM


@wraps(_state_initialization._rv_from_initial_state)
def _rv_from_initial_state(*args: Any, **kwargs: Any) -> Any:
    _sync_environment_constants()
    return _state_initialization._rv_from_initial_state(*args, **kwargs)


@wraps(_state_initialization._default_truth_from_agent)
def _default_truth_from_agent(*args: Any, **kwargs: Any) -> Any:
    _sync_environment_constants()
    return _state_initialization._default_truth_from_agent(*args, **kwargs)


@wraps(_satellite_factory._create_satellite_runtime)
def _create_satellite_runtime(*args: Any, **kwargs: Any) -> Any:
    _sync_environment_constants()
    return _satellite_factory._create_satellite_runtime(*args, **kwargs)


@wraps(_knowledge_factory._build_knowledge_base)
def _build_knowledge_base(*args: Any, **kwargs: Any) -> Any:
    _sync_environment_constants()
    return _knowledge_factory._build_knowledge_base(*args, **kwargs)


@wraps(_rocket_factory._create_rocket_runtime)
def _create_rocket_runtime(*args: Any, **kwargs: Any) -> Any:
    _sync_environment_constants()
    return _rocket_factory._create_rocket_runtime(*args, **kwargs)


# Preserve the legacy pickle and introspection paths while retaining one class
# object shared by the focused implementation modules and this façade.
AgentRuntime.__module__ = __name__

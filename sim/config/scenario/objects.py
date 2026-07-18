from __future__ import annotations

import math
from typing import Any

from sim.config.scenario.models import (
    AgentSection,
    AlgorithmPointer,
    BridgePointer,
    GroundStationSection,
)
from sim.config.scenario.presets import _AGENT_FRAGMENT_KEYS, _AGENT_PRESET_KEYS
from sim.config.scenario.primitives import (
    _as_dict,
    _parse_bool,
    _parse_float,
    _parse_optional_float,
    _reject_unknown_fields,
)

__all__ = [
    '_parse_algorithm_pointer',
    '_parse_bridge_pointer',
    '_parse_agent_section',
    '_INITIAL_STATE_AUX_KEYS',
    '_INITIAL_STATE_FORM_KEYS',
    '_INITIAL_STATE_ALLOWED_KEYS',
    '_parse_initial_state_section',
    '_reject_unsupported_agent_body_overrides',
    '_parse_ground_station_section',
    '_parse_ground_station_measurements',
    '_parse_objects_section',
    '_parse_ground_stations_section',
]

def _parse_algorithm_pointer(value: Any) -> AlgorithmPointer | None:
    if value is None:
        return None
    if isinstance(value, str):
        return AlgorithmPointer(module=value)
    d = _as_dict(value, "algorithm_pointer")
    if d.get("file") not in (None, ""):
        raise ValueError("Algorithm pointers do not support 'file'; use importable 'module' paths instead.")
    return AlgorithmPointer(
        kind=str(d.get("kind", "python")),
        module=d.get("module"),
        class_name=d.get("class_name"),
        function=d.get("function"),
        file=d.get("file"),
        params=dict(d.get("params", {}) or {}),
    )


def _parse_bridge_pointer(value: Any) -> BridgePointer | None:
    if value is None:
        return None
    d = _as_dict(value, "bridge")
    return BridgePointer(
        enabled=_parse_bool(d.get("enabled", False), "bridge.enabled"),
        mode=str(d.get("mode", "sil")),
        endpoint=d.get("endpoint"),
        module=d.get("module"),
        class_name=d.get("class_name"),
        params=dict(d.get("params", {}) or {}),
    )


def _parse_agent_section(
    value: Any,
    role: str,
    *,
    object_id: str | None = None,
    default_enabled: bool | None = None,
    default_kind: str | None = None,
) -> AgentSection:
    d = _as_dict(value, role)
    _reject_unsupported_agent_body_overrides(d, role)
    _reject_unknown_fields(d, role, set(_AGENT_FRAGMENT_KEYS) | set(_AGENT_PRESET_KEYS))
    objectives = d.get("mission_objectives", []) or []
    if not isinstance(objectives, list):
        raise ValueError(f"Section '{role}.mission_objectives' must be a list.")
    guidance_modifiers = d.get("guidance_modifiers", []) or []
    if not isinstance(guidance_modifiers, list):
        raise ValueError(f"Section '{role}.guidance_modifiers' must be a list.")
    default_enabled_by_role = {"rocket": False, "chaser": False, "target": True}
    resolved_object_id = str(d.get("object_id", object_id or role)).strip()
    if not resolved_object_id:
        raise ValueError(f"Section '{role}.object_id' must be non-empty.")
    resolved_role = str(d.get("role", role.split(".")[-1]))
    resolved_kind = (
        str(d.get("kind", default_kind or ("rocket" if resolved_role == "rocket" else "satellite"))).strip().lower()
    )
    if resolved_kind not in {"satellite", "rocket"}:
        raise ValueError(f"Section '{role}.kind' must be one of: satellite, rocket.")
    propagation_method = str(d.get("propagation_method", d.get("propagation_family", "")) or "").strip().lower()
    if propagation_method and propagation_method not in {"special", "general"}:
        raise ValueError(f"Section '{role}.propagation_method' must be one of: special, general.")
    general = _as_dict(d.get("general"), f"{role}.general")
    if resolved_kind != "rocket" and d.get("guidance") is not None:
        raise ValueError(
            f"Section '{role}.guidance' is no longer supported. "
            "Use mission_objectives for mission logic and orbit_control/attitude_control for controllers."
        )
    base_guidance = d.get("base_guidance")
    legacy_guidance = d.get("guidance")
    if resolved_kind == "rocket" and base_guidance is None and legacy_guidance is not None:
        base_guidance = legacy_guidance
    resolved_default_enabled = (
        bool(default_enabled_by_role.get(role, True)) if default_enabled is None else bool(default_enabled)
    )
    return AgentSection(
        object_id=resolved_object_id,
        kind=resolved_kind,
        enabled=_parse_bool(d.get("enabled", resolved_default_enabled), f"{role}.enabled"),
        role=resolved_role,
        propagation_method=propagation_method,
        general=general,
        specs=dict(d.get("specs", {}) or {}),
        initial_state=_parse_initial_state_section(d.get("initial_state"), role),
        reference_orbit=dict(d.get("reference_orbit", {}) or {}),
        guidance=_parse_algorithm_pointer(legacy_guidance),
        base_guidance=_parse_algorithm_pointer(base_guidance),
        guidance_modifiers=[p for p in (_parse_algorithm_pointer(x) for x in guidance_modifiers) if p is not None],
        orbit_control=_parse_algorithm_pointer(d.get("orbit_control")),
        attitude_control=_parse_algorithm_pointer(d.get("attitude_control")),
        mission_strategy=_parse_algorithm_pointer(d.get("mission_strategy")),
        mission_execution=_parse_algorithm_pointer(d.get("mission_execution")),
        mission_objectives=[p for p in (_parse_algorithm_pointer(x) for x in objectives) if p is not None],
        bridge=_parse_bridge_pointer(d.get("bridge")),
        knowledge=dict(d.get("knowledge", {}) or {}),
    )


_INITIAL_STATE_AUX_KEYS = {
    "attitude_quat_bn",
    "angular_rate_body_rad_s",
    "epoch_jd_utc",
    "relative_to",
    "deploy_time_s",
    "deploy_dv_body_m_s",
    "initialization_delay_s",
}
_INITIAL_STATE_FORM_KEYS = {
    "position_eci_km",
    "tle",
    "coes",
    "cr3bp_rotating",
    "cr3bp_halo",
    "relative_to_target_ric",
    "relative_ric_rect",
    "relative_ric_curv",
    "relative_to_target_cislunar",
    "relative_cislunar",
    "launch_lat_deg",
    "source",
    "default_circular_earth",
}
_INITIAL_STATE_ALLOWED_KEYS = _INITIAL_STATE_AUX_KEYS | _INITIAL_STATE_FORM_KEYS | {
    "velocity_eci_km_s",
    "launch_lon_deg",
    "launch_alt_km",
    "launch_azimuth_deg",
}


def _parse_initial_state_section(value: Any, role: str) -> dict[str, Any]:
    path = f"{role}.initial_state"
    state = _as_dict(value, path)
    if not state:
        # Retained only for the legacy implicit quickstart/default-object path.
        # Any non-empty state is strictly validated below so typos cannot fall
        # through to that runtime default.
        return {}
    # Give a targeted diagnostic for a common indentation mistake instead of
    # reporting only that the misplaced generic ``state`` key is unknown.
    relative_target = state.get("relative_to_target_ric")
    if "state" in state and isinstance(relative_target, dict) and "state" not in relative_target:
        raise ValueError(
            f"{path}.relative_to_target_ric.state must be a length-6 finite numeric list nested "
            "under relative_to_target_ric."
        )
    _reject_unknown_fields(state, path, _INITIAL_STATE_ALLOWED_KEYS)

    forms = [key for key in _INITIAL_STATE_FORM_KEYS if key in state]
    if len(forms) != 1:
        raise ValueError(
            f"{path} must define exactly one orbital-state form; found: "
            + (", ".join(sorted(forms)) if forms else "none")
            + "."
        )
    form = forms[0]
    if form == "position_eci_km" and "velocity_eci_km_s" not in state:
        raise ValueError(f"{path}.position_eci_km requires {path}.velocity_eci_km_s.")
    if form != "position_eci_km" and "velocity_eci_km_s" in state:
        raise ValueError(f"{path}.velocity_eci_km_s is only valid with position_eci_km.")
    if form == "launch_lat_deg":
        # Azimuth is optional and has a documented/runtime due-east default.
        required = {"launch_lat_deg", "launch_lon_deg", "launch_alt_km"}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"{path} launch state is missing: {', '.join(missing)}.")
    if form == "default_circular_earth" and not _parse_bool(
        state.get("default_circular_earth"), f"{path}.default_circular_earth"
    ):
        raise ValueError(f"{path}.default_circular_earth must be true when selected.")
    for key, length in (
        ("position_eci_km", 3),
        ("velocity_eci_km_s", 3),
        ("angular_rate_body_rad_s", 3),
        ("attitude_quat_bn", 4),
        ("deploy_dv_body_m_s", 3),
        ("relative_ric_rect", 6),
        ("relative_ric_curv", 6),
        ("relative_cislunar", 6),
    ):
        if key not in state:
            continue
        raw = state[key]
        if not isinstance(raw, (list, tuple)) or len(raw) != length:
            raise ValueError(f"{path}.{key} must be a length-{length} list.")
        values = [_parse_float(item, f"{path}.{key}") for item in raw]
        if key == "attitude_quat_bn" and math.sqrt(sum(item * item for item in values)) <= 0.0:
            raise ValueError(f"{path}.attitude_quat_bn must have nonzero norm.")
    if "initialization_delay_s" in state:
        delay_s = _parse_float(state["initialization_delay_s"], f"{path}.initialization_delay_s")
        if delay_s < 0.0:
            raise ValueError(f"{path}.initialization_delay_s must be nonnegative.")
    return dict(state)


def _reject_unsupported_agent_body_overrides(d: dict[str, Any], role: str) -> None:
    unsupported_root_keys = {
        "central_body",
        "primary_body",
        "body",
        "mu_km3_s2",
        "gravitational_parameter_km3_s2",
    }
    root_hits = sorted(str(key) for key in d if str(key) in unsupported_root_keys)
    if root_hits:
        raise ValueError(
            f"Section '{role}' has unsupported central-body field(s): {', '.join(root_hits)}. "
            "Object-level central-body overrides are not supported; public OEL orbit scenarios are "
            "Earth-centered except for documented CR3BP/cislunar modes."
        )

    orbit = d.get("orbit")
    if orbit is None:
        return
    if not isinstance(orbit, dict):
        raise ValueError(f"Section '{role}.orbit' must be a mapping/object when provided.")
    unsupported_orbit_keys = sorted(str(key) for key in orbit if str(key) in unsupported_root_keys)
    if unsupported_orbit_keys:
        raise ValueError(
            f"Section '{role}.orbit' has unsupported central-body field(s): "
            f"{', '.join(unsupported_orbit_keys)}. Object-level central-body overrides are not supported; "
            "public OEL orbit scenarios are Earth-centered except for documented CR3BP/cislunar modes."
        )


def _parse_ground_station_section(value: Any, index: int) -> GroundStationSection:
    d = _as_dict(value, f"ground_stations[{index}]")
    allowed_keys = {
        "id",
        "name",
        "lat_deg",
        "lon_deg",
        "alt_km",
        "altitude_km",
        "min_elevation_deg",
        "max_range_km",
        "enabled",
        "measurements",
    }
    unknown_keys = sorted(str(key) for key in d if str(key) not in allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"ground_stations[{index}] has unsupported field(s): {', '.join(unknown_keys)}. "
            "Ground stations are fixed geometric access sites; RF links or moving platforms are not modeled here."
        )
    raw_id = d.get("id", d.get("name", f"ground_station_{index + 1}"))
    station_id = str(raw_id or "").strip()
    if not station_id:
        raise ValueError(f"ground_stations[{index}].id must be non-empty.")
    if "lat_deg" not in d:
        raise ValueError(f"ground_stations[{index}].lat_deg is required.")
    if "lon_deg" not in d:
        raise ValueError(f"ground_stations[{index}].lon_deg is required.")
    lat_deg = _parse_float(d.get("lat_deg"), f"ground_stations[{index}].lat_deg")
    lon_deg = _parse_float(d.get("lon_deg"), f"ground_stations[{index}].lon_deg")
    alt_km = _parse_float(d.get("alt_km", d.get("altitude_km", 0.0)), f"ground_stations[{index}].alt_km")
    min_elevation_deg = _parse_float(
        d.get("min_elevation_deg", 0.0),
        f"ground_stations[{index}].min_elevation_deg",
    )
    max_range_km = _parse_optional_float(d.get("max_range_km"), f"ground_stations[{index}].max_range_km")
    if not (-90.0 <= lat_deg <= 90.0):
        raise ValueError(f"ground_stations[{index}].lat_deg must be between -90 and 90.")
    if max_range_km is not None and max_range_km <= 0.0:
        raise ValueError(f"ground_stations[{index}].max_range_km must be positive when provided.")
    measurements = _parse_ground_station_measurements(d.get("measurements", {}), index)
    return GroundStationSection(
        id=station_id,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_km=alt_km,
        min_elevation_deg=min_elevation_deg,
        max_range_km=max_range_km,
        enabled=_parse_bool(d.get("enabled", True), f"ground_stations[{index}].enabled"),
        measurements=measurements,
    )


def _parse_ground_station_measurements(value: Any, index: int) -> dict[str, Any]:
    if value in (None, False) or value == {}:
        return {}
    if value is True:
        return {"enabled": True}
    d = _as_dict(value, f"ground_stations[{index}].measurements")
    allowed_keys = {
        "enabled",
        "measurement_type",
        "update_cadence_s",
        "seed",
        "range_sigma_km",
        "range_rate_sigma_km_s",
        "angle_sigma_deg",
        "noise",
    }
    unknown_keys = sorted(str(key) for key in d if str(key) not in allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"ground_stations[{index}].measurements has unsupported field(s): {', '.join(unknown_keys)}."
        )
    out = dict(d)
    out["enabled"] = _parse_bool(out.get("enabled", True), f"ground_stations[{index}].measurements.enabled")
    if "update_cadence_s" in out:
        cadence = _parse_float(out.get("update_cadence_s"), f"ground_stations[{index}].measurements.update_cadence_s")
        if cadence <= 0.0:
            raise ValueError(f"ground_stations[{index}].measurements.update_cadence_s must be positive.")
        out["update_cadence_s"] = cadence
    if "seed" in out:
        out["seed"] = int(_parse_float(out.get("seed"), f"ground_stations[{index}].measurements.seed"))
    for key in ("range_sigma_km", "range_rate_sigma_km_s", "angle_sigma_deg"):
        if key in out:
            sigma = _parse_float(out.get(key), f"ground_stations[{index}].measurements.{key}")
            if sigma < 0.0:
                raise ValueError(f"ground_stations[{index}].measurements.{key} must be non-negative.")
            out[key] = sigma
    if "noise" in out:
        noise = _as_dict(out["noise"], f"ground_stations[{index}].measurements.noise")
        noise_allowed = {"range_sigma_km", "range_rate_sigma_km_s", "angle_sigma_deg"}
        noise_unknown = sorted(str(key) for key in noise if str(key) not in noise_allowed)
        if noise_unknown:
            raise ValueError(
                f"ground_stations[{index}].measurements.noise has unsupported field(s): {', '.join(noise_unknown)}."
            )
        out["noise"] = {
            key: _parse_float(value, f"ground_stations[{index}].measurements.noise.{key}")
            for key, value in noise.items()
        }
        for key, sigma in out["noise"].items():
            if sigma < 0.0:
                raise ValueError(f"ground_stations[{index}].measurements.noise.{key} must be non-negative.")
    if "measurement_type" in out:
        measurement_type = str(out["measurement_type"] or "").strip().lower()
        if measurement_type not in {"az_el_range", "az_el_range_rate"}:
            raise ValueError(
                f"ground_stations[{index}].measurements.measurement_type must be "
                "'az_el_range' or 'az_el_range_rate'."
            )
        out["measurement_type"] = measurement_type
    return out


def _parse_objects_section(
    value: Any,
) -> dict[str, AgentSection]:
    objects: dict[str, AgentSection] = {}
    if value is not None:
        raw_objects = _as_dict(value, "objects")
        for object_id, raw_agent in raw_objects.items():
            oid = str(object_id).strip()
            if not oid:
                raise ValueError("objects keys must be non-empty object ids.")
            agent = _parse_agent_section(
                raw_agent,
                role=f"objects.{oid}",
                object_id=oid,
                default_enabled=True,
                default_kind="rocket" if oid == "rocket" else "satellite",
            )
            if agent.object_id != oid:
                raise ValueError(f"objects.{oid}.object_id must match its object id key.")
            objects[oid] = agent

    return objects


def _parse_ground_stations_section(value: Any) -> list[GroundStationSection]:
    if value is None:
        return []
    if isinstance(value, dict):
        items = []
        for key, child in value.items():
            child_dict = _as_dict(child, f"ground_stations.{key}")
            child_dict.setdefault("id", str(key))
            items.append(child_dict)
    elif isinstance(value, list):
        items = list(value)
    else:
        raise ValueError("ground_stations must be a list or mapping.")
    stations = [_parse_ground_station_section(item, idx) for idx, item in enumerate(items)]
    seen: set[str] = set()
    for station in stations:
        if station.id in seen:
            raise ValueError(f"ground_stations contains duplicate id: {station.id}")
        seen.add(station.id)
    return stations

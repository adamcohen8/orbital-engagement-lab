from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import dataclass
from typing import Any

from sim.actuators.presets import available_actuator_preset_names, resolve_actuator_specs_from_satellite_specs
from sim.config.object_refs import configured_objects, object_parameter_prefix
from sim.digital_twin.mass_properties import validate_mass_properties
from sim.dynamics.orbit.tle import parse_tle_lines


@dataclass(frozen=True)
class PluginContract:
    methods_all: tuple[str, ...] = ()
    methods_any: tuple[str, ...] = ()
    allow_function: bool = False


_CONTRACTS = {
    "guidance": PluginContract(methods_all=("command",), allow_function=False),
    "orbit_control": PluginContract(methods_all=("act",), allow_function=False),
    "attitude_control": PluginContract(methods_all=("act",), allow_function=False),
    "mission_strategy": PluginContract(methods_all=(), methods_any=("update", "plan", "decide"), allow_function=True),
    "mission_execution": PluginContract(methods_all=(), methods_any=("update", "execute", "act"), allow_function=True),
    "bridge": PluginContract(
        methods_all=(),
        methods_any=("step", "start", "send_command", "receive_command", "external_intent"),
        allow_function=True,
    ),
    "mission_objective": PluginContract(
        methods_all=(), methods_any=("evaluate", "update", "check", "act"), allow_function=True
    ),
}


def _validate_pointer(pointer: Any, contract: PluginContract, path: str, *, import_plugins: bool = True) -> list[str]:
    errs: list[str] = []
    if pointer is None:
        return errs
    if not getattr(pointer, "module", None):
        errs.append(f"{path}: missing 'module'.")
        return errs
    class_name = getattr(pointer, "class_name", None)
    function = getattr(pointer, "function", None)
    if class_name and function:
        errs.append(f"{path}: define either 'class_name' or 'function', not both.")
        return errs
    if function and not contract.allow_function:
        errs.append(f"{path}: function pointers are not allowed for this plugin type.")
        return errs
    if not class_name and not function:
        errs.append(f"{path}: must define either 'class_name' or 'function'.")
        return errs
    if not import_plugins:
        return errs

    try:
        mod = importlib.import_module(pointer.module)
    except Exception as ex:
        errs.append(f"{path}: failed to import module '{pointer.module}': {ex}")
        return errs

    if class_name:
        if not hasattr(mod, class_name):
            errs.append(f"{path}: class '{class_name}' not found in module '{pointer.module}'.")
            return errs
        cls = getattr(mod, class_name)
        if not inspect.isclass(cls):
            errs.append(f"{path}: '{class_name}' in module '{pointer.module}' is not a class.")
            return errs
        for m in contract.methods_all:
            if not _class_has_callable(cls, m):
                errs.append(f"{path}: class '{class_name}' missing required callable method '{m}'.")
        if contract.methods_any:
            if not any(_class_has_callable(cls, m) for m in contract.methods_any):
                errs.append(f"{path}: class '{class_name}' must implement one of {list(contract.methods_any)}.")
        return errs

    if function:
        if not hasattr(mod, function):
            errs.append(f"{path}: function '{function}' not found in module '{pointer.module}'.")
            return errs
        fn = getattr(mod, function)
        if not callable(fn):
            errs.append(f"{path}: '{function}' in module '{pointer.module}' is not callable.")
        return errs

    errs.append(f"{path}: must define either 'class_name' or 'function'.")
    return errs


def _class_has_callable(cls: type, method_name: str) -> bool:
    for base in cls.__mro__:
        if method_name in base.__dict__:
            attr = base.__dict__[method_name]
            if isinstance(attr, (staticmethod, classmethod)):
                return callable(attr.__func__)
            return callable(attr)
    return False


def validate_scenario_plugins(cfg: Any, *, import_plugins: bool = True) -> list[str]:
    errs: list[str] = []
    orbit_cfg = dict(getattr(getattr(getattr(cfg, "simulator", None), "dynamics", {}), "orbit", {}) or {})
    default_propagation_method = str(orbit_cfg.get("propagation_method", "special") or "special").strip().lower()
    for object_id, agent in configured_objects(cfg).items():
        if not getattr(agent, "enabled", False):
            continue
        path = object_parameter_prefix(str(object_id))
        propagation_method = str(
            getattr(agent, "propagation_method", "") or default_propagation_method or "special"
        ).strip().lower()
        errs.extend(_validate_object_propagation(agent, propagation_method, path))
        errs.extend(_validate_object_mass_properties(getattr(agent, "specs", {}) or {}, f"{path}.specs"))
        errs.extend(_validate_initial_state(getattr(agent, "initial_state", {}) or {}, f"{path}.initial_state"))
        errs.extend(_validate_object_knowledge(getattr(agent, "knowledge", {}) or {}, f"{path}.knowledge"))
        if str(getattr(agent, "kind", "satellite")).strip().lower() == "rocket":
            errs.extend(
                _validate_pointer(
                    getattr(agent, "guidance", None),
                    _CONTRACTS["guidance"],
                    f"{path}.guidance",
                    import_plugins=import_plugins,
                )
            )
            errs.extend(
                _validate_pointer(
                    getattr(agent, "base_guidance", None),
                    _CONTRACTS["guidance"],
                    f"{path}.base_guidance",
                    import_plugins=import_plugins,
                )
            )
            for i, p in enumerate(getattr(agent, "guidance_modifiers", []) or []):
                errs.extend(
                    _validate_rocket_guidance_modifier(
                        p, f"{path}.guidance_modifiers[{i}]", import_plugins=import_plugins
                    )
                )
        else:
            errs.extend(_validate_satellite_actuator_specs(getattr(agent, "specs", {}) or {}, f"{path}.specs"))
        errs.extend(
            _validate_pointer(
                getattr(agent, "orbit_control", None),
                _CONTRACTS["orbit_control"],
                f"{path}.orbit_control",
                import_plugins=import_plugins,
            )
        )
        errs.extend(
            _validate_pointer(
                getattr(agent, "attitude_control", None),
                _CONTRACTS["attitude_control"],
                f"{path}.attitude_control",
                import_plugins=import_plugins,
            )
        )
        errs.extend(
            _validate_pointer(
                getattr(agent, "mission_strategy", None),
                _CONTRACTS["mission_strategy"],
                f"{path}.mission_strategy",
                import_plugins=import_plugins,
            )
        )
        errs.extend(
            _validate_pointer(
                getattr(agent, "mission_execution", None),
                _CONTRACTS["mission_execution"],
                f"{path}.mission_execution",
                import_plugins=import_plugins,
            )
        )
        bridge = getattr(agent, "bridge", None)
        if bridge is not None and getattr(bridge, "enabled", False):
            errs.extend(_validate_pointer(bridge, _CONTRACTS["bridge"], f"{path}.bridge", import_plugins=import_plugins))
            if _is_cfs_sil_bridge(bridge):
                errs.extend(_validate_cfs_sil_timing(cfg, f"{path}.bridge"))
        for i, p in enumerate(getattr(agent, "mission_objectives", []) or []):
            errs.extend(
                _validate_pointer(
                    p, _CONTRACTS["mission_objective"], f"{path}.mission_objectives[{i}]", import_plugins=import_plugins
                )
            )
    return errs


def _validate_object_propagation(agent: Any, propagation_method: str, path: str) -> list[str]:
    errs: list[str] = []
    method = str(propagation_method or "special").strip().lower()
    if method not in {"special", "general"}:
        return [f"{path}.propagation_method: must be one of: special, general."]
    if method != "general":
        return errs

    kind = str(getattr(agent, "kind", "satellite") or "satellite").strip().lower()
    if kind != "satellite":
        errs.append(f"{path}.propagation_method=general is only supported for satellite objects.")
    general = dict(getattr(agent, "general", {}) or {})
    model = str(general.get("model", "") or "").strip().lower()
    if model != "sgp4":
        errs.append(f"{path}.general.model must be 'sgp4' when propagation_method=general.")
    initial_state = dict(getattr(agent, "initial_state", {}) or {})
    tle_block = initial_state.get("tle")
    if not isinstance(tle_block, dict):
        errs.append(f"{path}.propagation_method=general with general.model=sgp4 requires initial_state.tle.")
    else:
        try:
            lines = tle_block.get("lines")
            if isinstance(lines, (list, tuple)) and len(lines) >= 2:
                line1 = str(lines[0])
                line2 = str(lines[1])
            else:
                line1 = str(tle_block.get("line1", "") or "")
                line2 = str(tle_block.get("line2", "") or "")
            elements = parse_tle_lines(line1, line2, require_checksum=bool(tle_block.get("require_checksum", False)))
            if float(elements.mean_motion_rev_per_day) <= 0.0:
                errs.append(f"{path}.initial_state.tle: OGP mean motion must be positive.")
            if float(elements.eccentricity) < 0.0 or float(elements.eccentricity) >= 1.0:
                errs.append(f"{path}.initial_state.tle: OGP eccentricity must be in [0, 1).")
        except Exception as ex:
            errs.append(f"{path}.initial_state.tle: invalid TLE for OGP propagation: {ex}")
    unsupported_initial_forms = [key for key in initial_state if str(key) != "tle"]
    if unsupported_initial_forms:
        errs.append(
            f"{path}.propagation_method=general with general.model=sgp4 does not support initial_state field(s): "
            + ", ".join(sorted(unsupported_initial_forms))
            + ". Use initial_state.tle."
        )
    if getattr(agent, "orbit_control", None) is not None:
        errs.append(f"{path}.orbit_control is not supported for passive general-propagation SGP4 objects.")
    if getattr(agent, "attitude_control", None) is not None:
        errs.append(f"{path}.attitude_control is not supported for passive general-propagation SGP4 objects.")
    if getattr(agent, "mission_strategy", None) is not None or getattr(agent, "mission_execution", None) is not None:
        errs.append(f"{path}.mission_strategy/mission_execution are not supported for passive general-propagation SGP4 objects.")
    if list(getattr(agent, "mission_objectives", []) or []):
        errs.append(f"{path}.mission_objectives are not supported for passive general-propagation SGP4 objects.")
    allowed_general_keys = {"model", "output_frame", "frame_transform", "max_tle_age_days_warning"}
    unknown_general_keys = sorted(str(key) for key in general if str(key) not in allowed_general_keys)
    if unknown_general_keys:
        errs.append(f"{path}.general has unsupported field(s): {', '.join(unknown_general_keys)}.")
    output_frame = str(general.get("output_frame", "eci") or "eci").strip().lower()
    if output_frame not in {"eci", "teme"}:
        errs.append(f"{path}.general.output_frame must be 'eci' or 'teme' for SGP4 v1.")
    default_transform = "native" if output_frame == "teme" else "teme_as_eci"
    frame_transform = str(general.get("frame_transform", default_transform) or default_transform).strip().lower()
    if output_frame == "eci" and frame_transform not in {"teme_as_eci", "teme_to_eci_iau80"}:
        errs.append(
            f"{path}.general.frame_transform must be 'teme_as_eci' or 'teme_to_eci_iau80' "
            "when output_frame is 'eci'."
        )
    if output_frame == "teme" and frame_transform not in {"native", "none", "identity", "teme"}:
        errs.append(f"{path}.general.frame_transform must be 'native' when output_frame is 'teme'.")
    return errs


def _validate_object_knowledge(knowledge: dict[str, Any], path: str) -> list[str]:
    raw = dict(knowledge or {})
    if "sensor" not in raw:
        return []
    return [
        (
            f"{path}.sensor: unsupported modeled-sensor configuration block. "
            "Use knowledge.sensor_error for measurement error assumptions and knowledge.estimation for estimator "
            "settings; optical/radar camera hardware fields are not modeled through this config path."
        )
    ]


def _validate_object_mass_properties(specs: dict[str, Any], path: str) -> list[str]:
    raw = dict(specs or {})
    errs: list[str] = []
    if ("dry_mass_kg" in raw) or ("fuel_mass_kg" in raw):
        dry = _parse_finite_float(raw.get("dry_mass_kg", 0.0), f"{path}.dry_mass_kg", errs)
        fuel = _parse_finite_float(raw.get("fuel_mass_kg", 0.0), f"{path}.fuel_mass_kg", errs)
        if dry is not None and fuel is not None:
            if dry < 0.0:
                errs.append(f"{path}.dry_mass_kg: must be >= 0.")
            if fuel < 0.0:
                errs.append(f"{path}.fuel_mass_kg: must be >= 0.")
            if dry + fuel <= 0.0:
                errs.append(f"{path}: dry_mass_kg + fuel_mass_kg must be > 0.")
    elif "mass_kg" in raw:
        mass = _parse_finite_float(raw.get("mass_kg"), f"{path}.mass_kg", errs)
        if mass is not None and mass <= 0.0:
            errs.append(f"{path}.mass_kg: must be > 0.")
    result = validate_mass_properties(raw, path=f"{path}.mass_properties")
    errs.extend(result.errors)
    return errs


def _validate_initial_state(initial_state: dict[str, Any], path: str) -> list[str]:
    errs: list[str] = []
    raw = dict(initial_state or {})
    tle = raw.get("tle")
    if tle is not None:
        if not isinstance(tle, dict):
            errs.append(f"{path}.tle: must be a mapping/object.")
        else:
            allowed_tle_keys = {
                "line1",
                "line2",
                "lines",
                "require_checksum",
                "propagate_to_initial_epoch",
            }
            unknown_tle_keys = sorted(str(key) for key in tle if str(key) not in allowed_tle_keys)
            if unknown_tle_keys:
                errs.append(
                    f"{path}.tle has unsupported field(s): {', '.join(unknown_tle_keys)}. "
                    "TLE initial states normally initialize OEL numerical propagation; use object-level "
                    "propagation_method: general with general.model: sgp4 for passive SGP4 propagation."
                )
    rel_block = raw.get("relative_to_target_ric")
    if rel_block is not None:
        if not isinstance(rel_block, dict):
            errs.append(f"{path}.relative_to_target_ric: must be a mapping/object.")
        else:
            rel = dict(rel_block)
            frame = str(rel.get("frame", "rect") or "").strip().lower()
            if frame not in {"rect", "curv"}:
                errs.append(f"{path}.relative_to_target_ric.frame: must be 'rect' or 'curv'.")
            reference_frame = str(rel.get("reference_frame", rel.get("origin", "target")) or "").strip().lower()
            if reference_frame.replace("-", "_") not in {"target", "moon", "moon_ric", "lunar", "lunar_ric"}:
                errs.append(
                    f"{path}.relative_to_target_ric.reference_frame: must be 'target', 'moon', or 'moon_ric'."
                )
            if "state" not in rel:
                errs.append(f"{path}.relative_to_target_ric.state: must be a length-6 finite numeric list.")
            else:
                errs.extend(
                    _validate_numeric_sequence(
                        rel.get("state"),
                        f"{path}.relative_to_target_ric.state",
                        length=6,
                    )
                )
    if "relative_ric_rect" in raw:
        errs.extend(_validate_numeric_sequence(raw.get("relative_ric_rect"), f"{path}.relative_ric_rect", length=6))
    if "relative_ric_curv" in raw:
        errs.extend(_validate_numeric_sequence(raw.get("relative_ric_curv"), f"{path}.relative_ric_curv", length=6))
    rel_cislunar = raw.get("relative_to_target_cislunar")
    if rel_cislunar is not None:
        if not isinstance(rel_cislunar, dict):
            errs.append(f"{path}.relative_to_target_cislunar: must be a mapping/object.")
        elif "state" not in rel_cislunar:
            errs.append(f"{path}.relative_to_target_cislunar.state: must be a length-6 finite numeric list.")
        else:
            errs.extend(
                _validate_numeric_sequence(
                    rel_cislunar.get("state"),
                    f"{path}.relative_to_target_cislunar.state",
                    length=6,
                )
            )
    if "relative_cislunar" in raw:
        errs.extend(_validate_numeric_sequence(raw.get("relative_cislunar"), f"{path}.relative_cislunar", length=6))
    if "position_eci_km" in raw:
        errs.extend(_validate_numeric_sequence(raw.get("position_eci_km"), f"{path}.position_eci_km", length=3))
        try:
            pos = [float(x) for x in list(raw.get("position_eci_km"))]
        except (TypeError, ValueError):
            pos = []
        if len(pos) == 3 and all(math.isfinite(x) for x in pos):
            if math.sqrt(sum(x * x for x in pos)) <= 0.0:
                errs.append(f"{path}.position_eci_km: must be nonzero.")
    if "velocity_eci_km_s" in raw:
        errs.extend(_validate_numeric_sequence(raw.get("velocity_eci_km_s"), f"{path}.velocity_eci_km_s", length=3))
    cr3bp_rotating = raw.get("cr3bp_rotating")
    if cr3bp_rotating is not None:
        if not isinstance(cr3bp_rotating, dict):
            errs.append(f"{path}.cr3bp_rotating: must be a mapping/object.")
        else:
            errs.extend(
                _validate_numeric_sequence(
                    cr3bp_rotating.get("state_km_s", cr3bp_rotating.get("state")),
                    f"{path}.cr3bp_rotating.state_km_s",
                    length=6,
                )
            )
    return errs


def _parse_finite_float(value: Any, path: str, errs: list[str]) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errs.append(f"{path}: must be a finite number.")
        return None
    if not math.isfinite(parsed):
        errs.append(f"{path}: must be a finite number.")
        return None
    return parsed


def _validate_numeric_sequence(value: Any, path: str, *, length: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return [f"{path}: must be a length-{length} finite numeric list."]
    if len(value) != length:
        return [f"{path}: must contain exactly {length} values."]
    errs: list[str] = []
    for i, item in enumerate(value):
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            errs.append(f"{path}[{i}]: must be a finite number.")
            continue
        if not math.isfinite(numeric):
            errs.append(f"{path}[{i}]: must be a finite number.")
    return errs


def _is_cfs_sil_bridge(pointer: Any) -> bool:
    return False


def _validate_cfs_sil_timing(cfg: Any, path: str) -> list[str]:
    return []


def _validate_satellite_actuator_specs(specs: dict[str, Any], path: str) -> list[str]:
    raw = dict(specs or {})
    explicit_block = raw.get("actuators", raw.get("actuator_model"))
    if raw.get("actuator_preset") in (None, "") and explicit_block in (None, ""):
        return []
    if explicit_block is not None and not isinstance(explicit_block, dict):
        return [f"{path}.actuators: must be a mapping/object when provided."]

    try:
        resolved = resolve_actuator_specs_from_satellite_specs(raw)
    except KeyError as exc:
        choices = ", ".join(available_actuator_preset_names())
        return [f"{path}.actuator_preset: {exc.args[0]} Choices: {choices}."]
    if not isinstance(resolved, dict):
        return [f"{path}.actuators: must resolve to a mapping/object."]
    if not bool(resolved.get("enabled", True)):
        return []

    errs: list[str] = []
    allowed = {"enabled", "preset", "orbital", "attitude", "faults"}
    errs.extend(_validate_allowed_keys(resolved, allowed, f"{path}.actuators"))
    orbital = resolved.get("orbital")
    if orbital is not None:
        if not isinstance(orbital, dict):
            errs.append(f"{path}.actuators.orbital: must be a mapping/object.")
        else:
            errs.extend(_validate_orbital_actuator_block(dict(orbital), f"{path}.actuators.orbital"))
    attitude = resolved.get("attitude")
    if attitude is not None:
        if not isinstance(attitude, dict):
            errs.append(f"{path}.actuators.attitude: must be a mapping/object.")
        else:
            errs.extend(_validate_attitude_actuator_block(dict(attitude), f"{path}.actuators.attitude"))
    faults = resolved.get("faults")
    if faults is not None:
        if not isinstance(faults, dict):
            errs.append(f"{path}.actuators.faults: must be a mapping/object.")
        else:
            errs.extend(_validate_fault_block(dict(faults), f"{path}.actuators.faults"))
    return errs


def _validate_allowed_keys(raw: dict[str, Any], allowed: set[str], path: str) -> list[str]:
    return [f"{path}.{key}: unknown actuator configuration key." for key in sorted(set(raw) - allowed)]


def _validate_finite_float(
    value: Any,
    path: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    required: bool = False,
) -> list[str]:
    if value is None:
        return [f"{path}: is required."] if required else []
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return [f"{path}: must be a finite number."]
    if not math.isfinite(parsed):
        return [f"{path}: must be a finite number."]
    if min_value is not None and parsed < min_value:
        return [f"{path}: must be >= {min_value:g}."]
    if max_value is not None and parsed > max_value:
        return [f"{path}: must be <= {max_value:g}."]
    return []


def _validate_vector(
    value: Any,
    path: str,
    *,
    lengths: tuple[int, ...],
    required: bool = False,
    nonzero: bool = False,
    min_value: float | None = None,
) -> list[str]:
    if value is None:
        return [f"{path}: is required."] if required else []
    try:
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            vals = [float(value)]
        else:
            vals = [float(x) for x in list(value)]
    except (TypeError, ValueError):
        return [f"{path}: must be a finite numeric vector."]
    if len(vals) not in lengths:
        expected = " or ".join(str(n) for n in lengths)
        return [f"{path}: must contain {expected} values."]
    if not all(math.isfinite(x) for x in vals):
        return [f"{path}: must contain only finite numbers."]
    if min_value is not None and any(x < min_value for x in vals):
        return [f"{path}: values must be >= {min_value:g}."]
    if nonzero and math.sqrt(sum(x * x for x in vals)) <= 0.0:
        return [f"{path}: must be nonzero."]
    return []


def _validate_orbital_actuator_block(raw: dict[str, Any], path: str) -> list[str]:
    errs: list[str] = []
    allowed = {
        "max_accel_km_s2",
        "max_thrust_n",
        "min_impulse_bit_km_s",
        "max_throttle_rate_km_s2_s",
        "isp_s",
        "thruster_direction_body",
        "thruster_position_body_m",
        "couple_to_attitude",
        "lag_tau_s",
        "rcs_cluster",
        "electric_propulsion",
        "gimbaled_thruster",
    }
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    for key in ("max_accel_km_s2", "max_thrust_n", "min_impulse_bit_km_s", "max_throttle_rate_km_s2_s", "lag_tau_s"):
        errs.extend(_validate_finite_float(raw.get(key), f"{path}.{key}", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("isp_s"), f"{path}.isp_s", min_value=0.0))
    errs.extend(_validate_vector(raw.get("thruster_direction_body"), f"{path}.thruster_direction_body", lengths=(3,), nonzero=True))
    errs.extend(_validate_vector(raw.get("thruster_position_body_m"), f"{path}.thruster_position_body_m", lengths=(3,)))
    errs.extend(_validate_rcs_cluster(raw.get("rcs_cluster"), f"{path}.rcs_cluster"))
    errs.extend(_validate_electric_propulsion(raw.get("electric_propulsion"), f"{path}.electric_propulsion"))
    errs.extend(_validate_gimbaled_thruster(raw.get("gimbaled_thruster"), f"{path}.gimbaled_thruster"))
    return errs


def _validate_rcs_cluster(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    if not bool(raw.get("enabled", True)):
        return []
    errs: list[str] = []
    allowed = {"enabled", "allocation_mode", "pulse_quantum_s", "duty_cycle", "isp_s", "thrusters"}
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    mode = str(raw.get("allocation_mode", "force_torque")).strip()
    if mode not in {"force_torque", "force_only", "torque_only"}:
        errs.append(f"{path}.allocation_mode: must be one of force_torque, force_only, torque_only.")
    errs.extend(_validate_finite_float(raw.get("pulse_quantum_s"), f"{path}.pulse_quantum_s", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("duty_cycle"), f"{path}.duty_cycle", min_value=0.0, max_value=1.0))
    errs.extend(_validate_finite_float(raw.get("isp_s"), f"{path}.isp_s", min_value=0.0))
    thrusters = raw.get("thrusters", [])
    if not isinstance(thrusters, list) or len(thrusters) == 0:
        return [*errs, f"{path}.thrusters: must be a non-empty list."]
    for idx, thruster in enumerate(thrusters):
        t_path = f"{path}.thrusters[{idx}]"
        if not isinstance(thruster, dict):
            errs.append(f"{t_path}: must be a mapping/object.")
            continue
        allowed_thruster = {
            "name",
            "position_body_m",
            "force_direction_body",
            "max_thrust_n",
            "min_impulse_bit_n_s",
            "isp_s",
        }
        errs.extend(_validate_allowed_keys(thruster, allowed_thruster, t_path))
        errs.extend(_validate_vector(thruster.get("position_body_m"), f"{t_path}.position_body_m", lengths=(3,), required=True))
        errs.extend(
            _validate_vector(
                thruster.get("force_direction_body"),
                f"{t_path}.force_direction_body",
                lengths=(3,),
                required=True,
                nonzero=True,
            )
        )
        errs.extend(_validate_finite_float(thruster.get("max_thrust_n"), f"{t_path}.max_thrust_n", min_value=0.0, required=True))
        errs.extend(_validate_finite_float(thruster.get("min_impulse_bit_n_s"), f"{t_path}.min_impulse_bit_n_s", min_value=0.0))
        errs.extend(_validate_finite_float(thruster.get("isp_s"), f"{t_path}.isp_s", min_value=0.0))
    return errs


def _validate_electric_propulsion(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    if not bool(raw.get("enabled", True)):
        return []
    errs: list[str] = []
    allowed = {
        "enabled",
        "max_thrust_n",
        "isp_s",
        "duty_cycle",
        "max_power_w",
        "power_per_newton_w",
        "throttle_time_constant_s",
    }
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    errs.extend(_validate_finite_float(raw.get("max_thrust_n"), f"{path}.max_thrust_n", min_value=0.0, required=True))
    errs.extend(_validate_finite_float(raw.get("isp_s"), f"{path}.isp_s", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("duty_cycle"), f"{path}.duty_cycle", min_value=0.0, max_value=1.0))
    errs.extend(_validate_finite_float(raw.get("max_power_w"), f"{path}.max_power_w", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("power_per_newton_w"), f"{path}.power_per_newton_w", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("throttle_time_constant_s"), f"{path}.throttle_time_constant_s", min_value=0.0))
    return errs


def _validate_gimbaled_thruster(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    if not bool(raw.get("enabled", True)):
        return []
    errs: list[str] = []
    allowed = {
        "enabled",
        "neutral_direction_body",
        "position_body_m",
        "max_gimbal_angle_rad",
        "max_gimbal_angle_deg",
        "max_gimbal_rate_rad_s",
        "max_gimbal_rate_deg_s",
        "response_time_constant_s",
    }
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    errs.extend(_validate_vector(raw.get("neutral_direction_body"), f"{path}.neutral_direction_body", lengths=(3,), required=True, nonzero=True))
    errs.extend(_validate_vector(raw.get("position_body_m"), f"{path}.position_body_m", lengths=(3,)))
    for key in (
        "max_gimbal_angle_rad",
        "max_gimbal_angle_deg",
        "max_gimbal_rate_rad_s",
        "max_gimbal_rate_deg_s",
        "response_time_constant_s",
    ):
        errs.extend(_validate_finite_float(raw.get(key), f"{path}.{key}", min_value=0.0))
    return errs


def _validate_attitude_actuator_block(raw: dict[str, Any], path: str) -> list[str]:
    errs: list[str] = []
    allowed = {
        "reaction_wheels",
        "magnetorquers",
        "thruster_pulse",
        "control_moment_gyros",
        "wheel_desaturation",
    }
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    errs.extend(_validate_reaction_wheels(raw.get("reaction_wheels"), f"{path}.reaction_wheels"))
    errs.extend(_validate_magnetorquers(raw.get("magnetorquers"), f"{path}.magnetorquers"))
    errs.extend(_validate_thruster_pulse(raw.get("thruster_pulse"), f"{path}.thruster_pulse"))
    errs.extend(_validate_cmg(raw.get("control_moment_gyros"), f"{path}.control_moment_gyros"))
    errs.extend(_validate_wheel_desaturation(raw.get("wheel_desaturation"), f"{path}.wheel_desaturation"))
    return errs


def _validate_reaction_wheels(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    if not bool(raw.get("enabled", True)):
        return []
    errs: list[str] = []
    allowed = {
        "enabled",
        "max_torque_nm",
        "max_momentum_nms",
        "wheel_axes_body",
        "wheel_inertia_kg_m2",
        "max_speed_rad_s",
        "torque_time_constant_s",
        "viscous_friction_nms",
        "coulomb_friction_nm",
    }
    errs.extend(_validate_allowed_keys(raw, allowed, path))
    errs.extend(_validate_vector(raw.get("max_torque_nm"), f"{path}.max_torque_nm", lengths=(1, 3), required=True, min_value=0.0))
    errs.extend(_validate_vector(raw.get("max_momentum_nms"), f"{path}.max_momentum_nms", lengths=(1, 3), required=True, min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("torque_time_constant_s"), f"{path}.torque_time_constant_s", min_value=0.0))
    return errs


def _validate_magnetorquers(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    if not bool(raw.get("enabled", True)):
        return []
    errs = _validate_allowed_keys(raw, {"enabled", "max_dipole_a_m2"}, path)
    errs.extend(_validate_vector(raw.get("max_dipole_a_m2"), f"{path}.max_dipole_a_m2", lengths=(1, 3), required=True, min_value=0.0))
    return errs


def _validate_thruster_pulse(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    if not bool(raw.get("enabled", True)):
        return []
    errs = _validate_allowed_keys(raw, {"enabled", "max_torque_nm", "pulse_quantum_s"}, path)
    errs.extend(_validate_vector(raw.get("max_torque_nm"), f"{path}.max_torque_nm", lengths=(3,), required=True, min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("pulse_quantum_s"), f"{path}.pulse_quantum_s", min_value=0.0))
    return errs


def _validate_cmg(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    if not bool(raw.get("enabled", True)):
        return []
    allowed = {"enabled", "max_torque_nm", "momentum_nms", "gimbal_rate_limit_rad_s", "torque_time_constant_s"}
    errs = _validate_allowed_keys(raw, allowed, path)
    errs.extend(_validate_vector(raw.get("max_torque_nm"), f"{path}.max_torque_nm", lengths=(1, 3), required=True, min_value=0.0))
    errs.extend(_validate_vector(raw.get("momentum_nms"), f"{path}.momentum_nms", lengths=(1, 3), required=True, min_value=0.0))
    errs.extend(_validate_vector(raw.get("gimbal_rate_limit_rad_s"), f"{path}.gimbal_rate_limit_rad_s", lengths=(1, 3), min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("torque_time_constant_s"), f"{path}.torque_time_constant_s", min_value=0.0))
    return errs


def _validate_wheel_desaturation(raw: Any, path: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{path}: must be a mapping/object."]
    if not bool(raw.get("enabled", True)):
        return []
    allowed = {"enabled", "momentum_fraction_threshold", "unload_gain_s_inv", "max_unload_torque_nm"}
    errs = _validate_allowed_keys(raw, allowed, path)
    errs.extend(_validate_finite_float(raw.get("momentum_fraction_threshold"), f"{path}.momentum_fraction_threshold", min_value=0.0, max_value=1.0))
    errs.extend(_validate_finite_float(raw.get("unload_gain_s_inv"), f"{path}.unload_gain_s_inv", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("max_unload_torque_nm"), f"{path}.max_unload_torque_nm", min_value=0.0))
    return errs


def _validate_fault_block(raw: dict[str, Any], path: str) -> list[str]:
    errs = _validate_allowed_keys(
        raw,
        {"stuck_off", "thrust_scale", "torque_scale", "thrust_bias_eci_km_s2", "torque_bias_body_nm"},
        path,
    )
    errs.extend(_validate_finite_float(raw.get("thrust_scale"), f"{path}.thrust_scale", min_value=0.0))
    errs.extend(_validate_finite_float(raw.get("torque_scale"), f"{path}.torque_scale", min_value=0.0))
    errs.extend(_validate_vector(raw.get("thrust_bias_eci_km_s2"), f"{path}.thrust_bias_eci_km_s2", lengths=(3,)))
    errs.extend(_validate_vector(raw.get("torque_bias_body_nm"), f"{path}.torque_bias_body_nm", lengths=(3,)))
    return errs


def _validate_rocket_guidance_modifier(pointer: Any, path: str, *, import_plugins: bool = True) -> list[str]:
    errs: list[str] = []
    if pointer is None:
        return errs
    if getattr(pointer, "kind", "python") != "python":
        return [f"{path}: only kind='python' is supported."]
    if not getattr(pointer, "module", None):
        return [f"{path}: 'module' is required for python pointers."]
    class_name = getattr(pointer, "class_name", None)
    if not class_name:
        return [f"{path}: must define 'class_name'."]
    if not import_plugins:
        return errs
    try:
        mod = importlib.import_module(str(pointer.module))
    except Exception as ex:
        return [f"{path}: failed to import module '{pointer.module}': {ex}"]
    if not hasattr(mod, class_name):
        return [f"{path}: class '{class_name}' not found in module '{pointer.module}'."]
    cls = getattr(mod, class_name)
    if not inspect.isclass(cls):
        return [f"{path}: '{class_name}' in module '{pointer.module}' is not a class."]
    if not _class_has_callable(cls, "command"):
        errs.append(f"{path}: class '{class_name}' missing required callable method 'command'.")
    return errs

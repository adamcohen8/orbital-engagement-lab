from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlgorithmPointer:
    kind: str = "python"
    module: str | None = None
    class_name: str | None = None
    function: str | None = None
    file: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BridgePointer:
    enabled: bool = False
    mode: str = "sil"
    endpoint: str | None = None
    module: str | None = None
    class_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSection:
    object_id: str = ""
    kind: str = "satellite"
    enabled: bool = True
    role: str = "agent"
    specs: dict[str, Any] = field(default_factory=dict)
    initial_state: dict[str, Any] = field(default_factory=dict)
    reference_orbit: dict[str, Any] = field(default_factory=dict)
    guidance: AlgorithmPointer | None = None
    base_guidance: AlgorithmPointer | None = None
    guidance_modifiers: list[AlgorithmPointer] = field(default_factory=list)
    orbit_control: AlgorithmPointer | None = None
    attitude_control: AlgorithmPointer | None = None
    mission_strategy: AlgorithmPointer | None = None
    mission_execution: AlgorithmPointer | None = None
    mission_objectives: list[AlgorithmPointer] = field(default_factory=list)
    bridge: BridgePointer | None = None
    knowledge: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundStationSection:
    id: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_elevation_deg: float = 0.0
    max_range_km: float | None = None
    enabled: bool = True


@dataclass(frozen=True)
class SimulatorSection:
    scenario_type: str = "auto"
    duration_s: float = 3600.0
    dt_s: float = 1.0
    initial_jd_utc: float | None = None
    dynamics: "SimulatorDynamicsSection" = field(default_factory=lambda: SimulatorDynamicsSection())
    environment: "SimulatorEnvironmentSection" = field(default_factory=lambda: SimulatorEnvironmentSection())
    plugin_validation: "SimulatorPluginValidationSection" = field(default_factory=lambda: SimulatorPluginValidationSection())
    termination: "SimulatorTerminationSection" = field(default_factory=lambda: SimulatorTerminationSection())

    def __post_init__(self) -> None:
        object.__setattr__(self, "dynamics", SimulatorDynamicsSection(self.dynamics))
        object.__setattr__(self, "environment", SimulatorEnvironmentSection(self.environment))
        object.__setattr__(self, "plugin_validation", SimulatorPluginValidationSection(self.plugin_validation))
        object.__setattr__(self, "termination", SimulatorTerminationSection(self.termination))


class _TypedConfigDict(dict):
    _defaults: dict[str, Any] = {}

    def __init__(self, value: Any = None, **overrides: Any) -> None:
        data = deepcopy(self._defaults)
        if value is not None:
            data.update(deepcopy(dict(value)))
        data.update(deepcopy(overrides))
        super().__init__(data)


class SimulatorDynamicsSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {}

    @property
    def orbit(self) -> dict[str, Any]:
        return dict(self.get("orbit", {}) or {})

    @property
    def attitude(self) -> dict[str, Any]:
        return dict(self.get("attitude", {}) or {})

    @property
    def rocket(self) -> dict[str, Any]:
        return dict(self.get("rocket", {}) or {})


class SimulatorEnvironmentSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {}

    @property
    def atmosphere_env(self) -> dict[str, Any]:
        return dict(self.get("atmosphere_env", {}) or {})


class SimulatorPluginValidationSection(_TypedConfigDict):
    _defaults = {"strict": True}

    @property
    def strict(self) -> bool:
        return bool(self.get("strict", True))


class SimulatorTerminationSection(_TypedConfigDict):
    _defaults = {
        "earth_impact_enabled": True,
        "earth_radius_km": 6378.137,
    }

    @property
    def earth_impact_enabled(self) -> bool:
        return bool(self.get("earth_impact_enabled", True))

    @property
    def earth_radius_km(self) -> float:
        return float(self.get("earth_radius_km", 6378.137))


class OutputStatsSection(_TypedConfigDict):
    _defaults = {
        "print_summary": True,
        "save_json": True,
        "save_full_log": True,
    }

    @property
    def print_summary(self) -> bool:
        return bool(self.get("print_summary", True))

    @property
    def save_json(self) -> bool:
        return bool(self.get("save_json", True))

    @property
    def save_full_log(self) -> bool:
        return bool(self.get("save_full_log", True))


class OutputPlotsSection(_TypedConfigDict):
    _defaults = {
        "enabled": True,
        "figure_ids": [],
        "dpi": 150,
    }

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled", True))

    @property
    def figure_ids(self) -> list[Any]:
        return list(self.get("figure_ids", []) or [])

    @property
    def dpi(self) -> int:
        return int(self.get("dpi", 150))


class OutputAnimationsSection(_TypedConfigDict):
    _defaults = {
        "enabled": False,
        "types": [],
        "fps": 30.0,
    }

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled", False))

    @property
    def types(self) -> list[Any]:
        return list(self.get("types", []) or [])

    @property
    def fps(self) -> float:
        return float(self.get("fps", 30.0))


class OutputMonteCarloSection(_TypedConfigDict):
    _defaults = {
        "save_histograms": False,
        "display_histograms": False,
        "save_ops_dashboard": True,
        "display_ops_dashboard": False,
        "save_iteration_summaries": False,
        "success_termination_reasons": ["rocket_orbit_insertion"],
    }

    @property
    def save_histograms(self) -> bool:
        return bool(self.get("save_histograms", False))

    @property
    def display_histograms(self) -> bool:
        return bool(self.get("display_histograms", False))

    @property
    def save_iteration_summaries(self) -> bool:
        return bool(self.get("save_iteration_summaries", False))

    @property
    def success_termination_reasons(self) -> list[Any]:
        return list(self.get("success_termination_reasons", ["rocket_orbit_insertion"]) or [])


class OutputAIReportSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled", False))

    @property
    def provider(self) -> str:
        return str(self.get("provider", "ollama") or "ollama")

    @property
    def model(self) -> str:
        return str(self.get("model", "") or "")


class OutputAIConfigSection(_TypedConfigDict):
    _defaults: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.get("enabled", True))

    @property
    def provider(self) -> str:
        return str(self.get("provider", "ollama") or "ollama")

    @property
    def model(self) -> str:
        return str(self.get("model", "") or "")


@dataclass(frozen=True)
class OutputsSection:
    output_dir: str = "outputs"
    mode: str = "interactive"
    stats: OutputStatsSection = field(default_factory=OutputStatsSection)
    plots: OutputPlotsSection = field(default_factory=OutputPlotsSection)
    animations: OutputAnimationsSection = field(default_factory=OutputAnimationsSection)
    monte_carlo: OutputMonteCarloSection = field(default_factory=OutputMonteCarloSection)
    ai_report: OutputAIReportSection = field(default_factory=OutputAIReportSection)
    ai_config: OutputAIConfigSection = field(default_factory=OutputAIConfigSection)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stats", OutputStatsSection(self.stats))
        object.__setattr__(self, "plots", OutputPlotsSection(self.plots))
        object.__setattr__(self, "animations", OutputAnimationsSection(self.animations))
        object.__setattr__(self, "monte_carlo", OutputMonteCarloSection(self.monte_carlo))
        object.__setattr__(self, "ai_report", OutputAIReportSection(self.ai_report))
        object.__setattr__(self, "ai_config", OutputAIConfigSection(self.ai_config))


@dataclass(frozen=True)
class MonteCarloVariation:
    parameter_path: str
    mode: str = "choice"
    options: list[Any] = field(default_factory=list)
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None


@dataclass(frozen=True)
class MonteCarloSection:
    enabled: bool = False
    iterations: int = 1
    base_seed: int = 0
    parallel_enabled: bool = False
    parallel_workers: int = 0
    variations: list[MonteCarloVariation] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisExecutionSection:
    parallel_enabled: bool = False
    parallel_workers: int = 0
    failure_policy: str = "fail_fast"


@dataclass(frozen=True)
class AnalysisBaselineSection:
    enabled: bool = False
    mode: str = "none"
    summary_json: str = ""


@dataclass(frozen=True)
class AnalysisMonteCarloSection:
    iterations: int = 1
    base_seed: int = 0
    variations: list[MonteCarloVariation] = field(default_factory=list)


@dataclass(frozen=True)
class SensitivityParameter:
    parameter_path: str
    values: list[Any] = field(default_factory=list)
    distribution: str = "uniform"
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None


@dataclass(frozen=True)
class SensitivitySection:
    method: str = "one_at_a_time"
    samples: int = 0
    seed: int = 0
    parameters: list[SensitivityParameter] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisSection:
    enabled: bool = False
    study_type: str = "monte_carlo"
    execution: AnalysisExecutionSection = field(default_factory=AnalysisExecutionSection)
    metrics: list[str] = field(default_factory=list)
    baseline: AnalysisBaselineSection = field(default_factory=AnalysisBaselineSection)
    monte_carlo: AnalysisMonteCarloSection = field(default_factory=AnalysisMonteCarloSection)
    sensitivity: SensitivitySection = field(default_factory=SensitivitySection)


@dataclass(frozen=True)
class SimulationScenarioConfig:
    scenario_name: str = "unnamed_scenario"
    scenario_description: str = ""
    rocket: AgentSection = field(default_factory=lambda: AgentSection(enabled=False, role="rocket"))
    chaser: AgentSection = field(default_factory=lambda: AgentSection(enabled=False, role="chaser"))
    target: AgentSection = field(default_factory=lambda: AgentSection(enabled=True, role="target"))
    objects: dict[str, AgentSection] = field(default_factory=dict)
    ground_stations: list[GroundStationSection] = field(default_factory=list)
    simulator: SimulatorSection = field(default_factory=SimulatorSection)
    outputs: OutputsSection = field(default_factory=OutputsSection)
    monte_carlo: MonteCarloSection = field(default_factory=MonteCarloSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _plain_config_data(self)


def _plain_config_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain_config_data(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {_plain_config_data(k): _plain_config_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain_config_data(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_config_data(item) for item in value]
    return deepcopy(value)


def _as_dict(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping/object.")
    return dict(value)


_AGENT_PRESET_KEYS = ("preset", "preset_yaml", "preset_path")
_AGENT_FRAGMENT_KEYS = {
    "object_id",
    "kind",
    "enabled",
    "role",
    "specs",
    "initial_state",
    "reference_orbit",
    "guidance",
    "base_guidance",
    "guidance_modifiers",
    "orbit_control",
    "attitude_control",
    "mission_strategy",
    "mission_execution",
    "mission_objectives",
    "bridge",
    "knowledge",
}
_PRESET_METADATA_KEYS = {
    "name",
    "description",
    "preset_type",
    "object_type",
    "version",
    "metadata",
}


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def _resolve_preset_path(preset_ref: str, base_dir: Path | None, role: str) -> Path:
    ref_path = Path(preset_ref).expanduser()
    candidates: list[Path] = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        if base_dir is not None:
            candidates.append(base_dir / ref_path)
        candidates.append(Path.cwd() / ref_path)
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / ref_path)
        candidates.append(repo_root / "sim" / "presets" / "objects" / ref_path)
        if ref_path.suffix == "":
            candidates.append(repo_root / "sim" / "presets" / "objects" / ref_path.with_suffix(".yaml"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not resolve {role} preset YAML '{preset_ref}'. Checked: {checked}")


def _load_yaml_mapping(path: Path, section_name: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load simulation YAML configs. Install with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{section_name} YAML root must be a mapping/object.")
    return dict(raw)


def _agent_fragment_from_preset(preset: dict[str, Any], role: str, preset_path: Path) -> dict[str, Any]:
    if "specs" in preset:
        fragment = {k: v for k, v in preset.items() if k not in _PRESET_METADATA_KEYS}
        fragment["specs"] = dict(fragment.get("specs", {}) or {})
        return fragment

    if any(k in preset for k in _AGENT_FRAGMENT_KEYS - {"specs"}):
        return {k: v for k, v in preset.items() if k not in _PRESET_METADATA_KEYS}

    specs = {k: v for k, v in preset.items() if k not in _PRESET_METADATA_KEYS}
    if not specs:
        raise ValueError(f"{role} preset YAML '{preset_path}' does not define specs.")
    return {"specs": specs}


def _resolve_agent_preset(value: Any, role: str, base_dir: Path | None) -> dict[str, Any]:
    d = _as_dict(value, role)
    preset_ref = next((d.get(key) for key in _AGENT_PRESET_KEYS if d.get(key) is not None), None)
    if preset_ref is None:
        return d
    if not isinstance(preset_ref, str) or not preset_ref.strip():
        raise ValueError(f"{role}.preset must be a non-empty YAML file path.")

    preset_path = _resolve_preset_path(preset_ref.strip(), base_dir=base_dir, role=role)
    preset_raw = _load_yaml_mapping(preset_path, f"{role} preset")
    preset_fragment = _agent_fragment_from_preset(preset_raw, role=role, preset_path=preset_path)
    local_fragment = {k: v for k, v in d.items() if k not in _AGENT_PRESET_KEYS}
    merged = _deep_merge_dicts(preset_fragment, local_fragment)

    local_specs = local_fragment.get("specs")
    merged_specs = merged.get("specs")
    if isinstance(local_specs, dict) and isinstance(merged_specs, dict):
        if "mass_kg" in local_specs and "dry_mass_kg" not in local_specs and "fuel_mass_kg" not in local_specs:
            merged_specs.pop("dry_mass_kg", None)
            merged_specs.pop("fuel_mass_kg", None)

    return merged


def _resolve_agent_presets(root: dict[str, Any], base_dir: Path | None) -> dict[str, Any]:
    resolved = dict(root)
    for role in ("rocket", "chaser", "target"):
        if role in resolved:
            resolved[role] = _resolve_agent_preset(resolved.get(role), role=role, base_dir=base_dir)
    objects_raw = resolved.get("objects")
    if isinstance(objects_raw, dict):
        resolved_objects = {}
        for object_id, object_value in objects_raw.items():
            resolved_objects[str(object_id)] = _resolve_agent_preset(
                object_value,
                role=f"objects.{object_id}",
                base_dir=base_dir,
            )
        resolved["objects"] = resolved_objects
    return resolved


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean true/false value, not {value!r}.")


def _is_bool_like_key(key: str) -> bool:
    normalized = key.strip().lower()
    if normalized in {
        "enabled",
        "strict",
        "j2",
        "j3",
        "j4",
        "drag",
        "srp",
        "third_body_moon",
        "third_body_sun",
        "parallel_enabled",
    }:
        return True
    return normalized.startswith(
        (
            "use_",
            "save_",
            "display_",
            "print_",
            "require_",
        )
    )


def _enforce_strict_booleans(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _is_bool_like_key(str(key)) and not isinstance(child, bool):
                raise ValueError(f"{child_path} must be a boolean true/false value, not {child!r}.")
            _enforce_strict_booleans(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _enforce_strict_booleans(child, f"{path}[{idx}]")


def _parse_float(value: Any, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number.") from exc
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be a finite number.")
    return out


def _parse_optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _parse_float(value, field_name)


def _validate_integer_multiple(
    *,
    numerator: float,
    denominator: float,
    numerator_name: str,
    denominator_name: str,
) -> None:
    ratio = numerator / denominator
    nearest = round(ratio)
    tol = 1e-9 * max(1.0, abs(ratio))
    if abs(ratio - nearest) > tol:
        raise ValueError(
            f"{numerator_name} must be an integer multiple of {denominator_name}; "
            f"got {numerator_name}={numerator:g}, {denominator_name}={denominator:g}."
        )


def _validate_sim_timing(out: SimulatorSection) -> None:
    if out.dt_s <= 0.0:
        raise ValueError("simulator.dt_s must be positive.")
    if out.duration_s <= 0.0:
        raise ValueError("simulator.duration_s must be positive.")
    _validate_integer_multiple(
        numerator=out.duration_s,
        denominator=out.dt_s,
        numerator_name="simulator.duration_s",
        denominator_name="simulator.dt_s",
    )

    dynamics = dict(out.dynamics or {})
    timing_fields = (
        ("simulator.dynamics.orbit.orbit_substep_s", dict(dynamics.get("orbit", {}) or {}).get("orbit_substep_s")),
        (
            "simulator.dynamics.attitude.attitude_substep_s",
            dict(dynamics.get("attitude", {}) or {}).get("attitude_substep_s"),
        ),
    )
    for field_name, raw in timing_fields:
        substep = _parse_optional_float(raw, field_name)
        if substep is None:
            continue
        if substep <= 0.0:
            raise ValueError(f"{field_name} must be positive when provided.")
        if substep > out.dt_s:
            raise ValueError(f"{field_name} must be less than or equal to simulator.dt_s.")
        _validate_integer_multiple(
            numerator=out.dt_s,
            denominator=substep,
            numerator_name="simulator.dt_s",
            denominator_name=field_name,
        )


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
    resolved_kind = str(d.get("kind", default_kind or ("rocket" if resolved_role == "rocket" else "satellite"))).strip().lower()
    if resolved_kind not in {"satellite", "rocket"}:
        raise ValueError(f"Section '{role}.kind' must be one of: satellite, rocket.")
    if resolved_kind != "rocket" and d.get("guidance") is not None:
        raise ValueError(
            f"Section '{role}.guidance' is no longer supported. "
            "Use mission_objectives for mission logic and orbit_control/attitude_control for controllers."
        )
    base_guidance = d.get("base_guidance")
    legacy_guidance = d.get("guidance")
    if resolved_kind == "rocket" and base_guidance is None and legacy_guidance is not None:
        base_guidance = legacy_guidance
    resolved_default_enabled = bool(default_enabled_by_role.get(role, True)) if default_enabled is None else bool(default_enabled)
    return AgentSection(
        object_id=resolved_object_id,
        kind=resolved_kind,
        enabled=_parse_bool(d.get("enabled", resolved_default_enabled), f"{role}.enabled"),
        role=resolved_role,
        specs=dict(d.get("specs", {}) or {}),
        initial_state=dict(d.get("initial_state", {}) or {}),
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


def _parse_ground_station_section(value: Any, index: int) -> GroundStationSection:
    d = _as_dict(value, f"ground_stations[{index}]")
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
    return GroundStationSection(
        id=station_id,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_km=alt_km,
        min_elevation_deg=min_elevation_deg,
        max_range_km=max_range_km,
        enabled=_parse_bool(d.get("enabled", True), f"ground_stations[{index}].enabled"),
    )


def _parse_objects_section(
    value: Any,
    *,
    legacy_agents: dict[str, AgentSection],
    has_objects_section: bool,
    legacy_keys_present: set[str],
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

    for legacy_id, legacy_agent in legacy_agents.items():
        if legacy_id in objects:
            continue
        if legacy_id in legacy_keys_present:
            objects[legacy_id] = legacy_agent
        elif not has_objects_section and legacy_id == "target":
            objects[legacy_id] = legacy_agent
        elif not has_objects_section and bool(legacy_agent.enabled):
            objects[legacy_id] = legacy_agent

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


def _parse_simulator_section(value: Any) -> SimulatorSection:
    d = _as_dict(value, "simulator")
    plugin_validation = {"strict": True}
    plugin_validation.update(dict(d.get("plugin_validation", {}) or {}))
    termination = {"earth_impact_enabled": True, "earth_radius_km": 6378.137}
    termination.update(dict(d.get("termination", {}) or {}))
    plugin_validation["strict"] = _parse_bool(plugin_validation.get("strict", True), "simulator.plugin_validation.strict")
    termination["earth_impact_enabled"] = _parse_bool(
        termination.get("earth_impact_enabled", True),
        "simulator.termination.earth_impact_enabled",
    )
    termination["earth_radius_km"] = _parse_float(
        termination.get("earth_radius_km", 6378.137),
        "simulator.termination.earth_radius_km",
    )
    out = SimulatorSection(
        scenario_type=str(d.get("scenario_type", "auto")),
        duration_s=_parse_float(d.get("duration_s", 3600.0), "simulator.duration_s"),
        dt_s=_parse_float(d.get("dt_s", 1.0), "simulator.dt_s"),
        initial_jd_utc=_parse_optional_float(d.get("initial_jd_utc"), "simulator.initial_jd_utc"),
        dynamics=dict(d.get("dynamics", {}) or {}),
        environment=dict(d.get("environment", {}) or {}),
        plugin_validation=plugin_validation,
        termination=termination,
    )
    _validate_sim_timing(out)
    if not out.scenario_type.strip():
        raise ValueError("simulator.scenario_type must be non-empty.")
    return out


def _parse_mc_variation(value: Any) -> MonteCarloVariation:
    d = _as_dict(value, "monte_carlo.variation")
    path = d.get("parameter_path")
    if not isinstance(path, str) or not path:
        raise ValueError("monte_carlo.variations[*].parameter_path must be a non-empty string.")
    return MonteCarloVariation(
        parameter_path=path,
        mode=str(d.get("mode", "choice")),
        options=list(d.get("options", []) or []),
        low=float(d["low"]) if d.get("low") is not None else None,
        high=float(d["high"]) if d.get("high") is not None else None,
        mean=float(d["mean"]) if d.get("mean") is not None else None,
        std=float(d["std"]) if d.get("std") is not None else None,
    )


def _parse_monte_carlo_section(value: Any) -> MonteCarloSection:
    d = _as_dict(value, "monte_carlo")
    vars_raw = d.get("variations", []) or []
    if not isinstance(vars_raw, list):
        raise ValueError("monte_carlo.variations must be a list.")
    out = MonteCarloSection(
        enabled=_parse_bool(d.get("enabled", False), "monte_carlo.enabled"),
        iterations=int(d.get("iterations", 1)),
        base_seed=int(d.get("base_seed", 0)),
        parallel_enabled=_parse_bool(d.get("parallel_enabled", False), "monte_carlo.parallel_enabled"),
        parallel_workers=int(d.get("parallel_workers", 0)),
        variations=[_parse_mc_variation(v) for v in vars_raw],
    )
    if out.iterations <= 0:
        raise ValueError("monte_carlo.iterations must be positive.")
    if out.parallel_workers < 0:
        raise ValueError("monte_carlo.parallel_workers must be >= 0.")
    return out


def _parse_analysis_execution_section(value: Any, *, fallback: MonteCarloSection | None = None) -> AnalysisExecutionSection:
    d = _as_dict(value, "analysis.execution")
    default_parallel_enabled = bool(fallback.parallel_enabled) if fallback is not None else False
    default_parallel_workers = int(fallback.parallel_workers) if fallback is not None else 0
    out = AnalysisExecutionSection(
        parallel_enabled=_parse_bool(
            d.get("parallel_enabled", default_parallel_enabled),
            "analysis.execution.parallel_enabled",
        ),
        parallel_workers=int(d.get("parallel_workers", default_parallel_workers)),
        failure_policy=str(d.get("failure_policy", "fail_fast") or "fail_fast").strip().lower(),
    )
    if out.parallel_workers < 0:
        raise ValueError("analysis.execution.parallel_workers must be >= 0.")
    if out.failure_policy not in {"fail_fast", "continue"}:
        raise ValueError("analysis.execution.failure_policy must be one of: fail_fast, continue.")
    return out


def _parse_analysis_baseline_section(value: Any) -> AnalysisBaselineSection:
    d = _as_dict(value, "analysis.baseline")
    summary_json = str(d.get("summary_json", "") or "")
    enabled = _parse_bool(d.get("enabled", False), "analysis.baseline.enabled")
    raw_mode = str(d.get("mode", "") or "").strip().lower()
    if not raw_mode:
        raw_mode = "file" if summary_json else ("run" if enabled else "none")
    if raw_mode not in {"none", "run", "file"}:
        raise ValueError("analysis.baseline.mode must be one of: none, run, file.")
    if raw_mode == "file" and not summary_json:
        raise ValueError("analysis.baseline.summary_json is required when mode is 'file'.")
    return AnalysisBaselineSection(
        enabled=bool(enabled or raw_mode in {"run", "file"}),
        mode=raw_mode,
        summary_json=summary_json,
    )


def _parse_analysis_monte_carlo_section(value: Any, *, fallback: MonteCarloSection | None = None) -> AnalysisMonteCarloSection:
    d = _as_dict(value, "analysis.monte_carlo")
    vars_raw = d.get("variations")
    if vars_raw is None:
        variations = list(fallback.variations) if fallback is not None else []
    else:
        if not isinstance(vars_raw, list):
            raise ValueError("analysis.monte_carlo.variations must be a list.")
        variations = [_parse_mc_variation(v) for v in vars_raw]
    default_iterations = int(fallback.iterations) if fallback is not None else 1
    default_base_seed = int(fallback.base_seed) if fallback is not None else 0
    out = AnalysisMonteCarloSection(
        iterations=int(d.get("iterations", default_iterations)),
        base_seed=int(d.get("base_seed", default_base_seed)),
        variations=variations,
    )
    if out.iterations <= 0:
        raise ValueError("analysis.monte_carlo.iterations must be positive.")
    return out


def _parse_sensitivity_parameter(value: Any) -> SensitivityParameter:
    d = _as_dict(value, "analysis.sensitivity.parameter")
    path = d.get("parameter_path", d.get("path"))
    if not isinstance(path, str) or not path:
        raise ValueError("analysis.sensitivity.parameters[*].parameter_path must be a non-empty string.")
    values = d.get("values", [])
    if not isinstance(values, list):
        raise ValueError("analysis.sensitivity.parameters[*].values must be a list.")
    distribution = str(d.get("distribution", "uniform")).strip().lower()
    if distribution not in {"uniform", "normal"}:
        raise ValueError("analysis.sensitivity.parameters[*].distribution must be one of: uniform, normal.")
    return SensitivityParameter(
        parameter_path=path,
        values=list(values),
        distribution=distribution,
        low=float(d["low"]) if d.get("low") is not None else None,
        high=float(d["high"]) if d.get("high") is not None else None,
        mean=float(d["mean"]) if d.get("mean") is not None else None,
        std=float(d["std"]) if d.get("std") is not None else None,
    )


def _parse_sensitivity_section(value: Any) -> SensitivitySection:
    d = _as_dict(value, "analysis.sensitivity")
    params_raw = d.get("parameters", []) or []
    if not isinstance(params_raw, list):
        raise ValueError("analysis.sensitivity.parameters must be a list.")
    out = SensitivitySection(
        method=str(d.get("method", "one_at_a_time")),
        samples=int(d.get("samples", 0)),
        seed=int(d.get("seed", 0)),
        parameters=[_parse_sensitivity_parameter(v) for v in params_raw],
    )
    if out.method not in {"one_at_a_time", "lhs", "two_parameter_grid"}:
        raise ValueError("analysis.sensitivity.method must be one of: one_at_a_time, lhs, two_parameter_grid.")
    if out.samples < 0:
        raise ValueError("analysis.sensitivity.samples must be >= 0.")
    return out


def _parse_analysis_section(value: Any, *, legacy_mc: MonteCarloSection) -> AnalysisSection:
    d = _as_dict(value, "analysis")
    metrics = d.get("metrics", []) or []
    if not isinstance(metrics, list):
        raise ValueError("analysis.metrics must be a list.")
    out = AnalysisSection(
        enabled=_parse_bool(d.get("enabled", False), "analysis.enabled"),
        study_type=str(d.get("study_type", "monte_carlo")).strip().lower(),
        execution=_parse_analysis_execution_section(d.get("execution"), fallback=legacy_mc),
        metrics=[str(x) for x in metrics],
        baseline=_parse_analysis_baseline_section(d.get("baseline")),
        monte_carlo=_parse_analysis_monte_carlo_section(d.get("monte_carlo"), fallback=legacy_mc),
        sensitivity=_parse_sensitivity_section(d.get("sensitivity")),
    )
    if out.study_type not in {"monte_carlo", "sensitivity"}:
        raise ValueError("analysis.study_type must be one of: monte_carlo, sensitivity.")
    return out


def _analysis_from_legacy_monte_carlo(mc: MonteCarloSection) -> AnalysisSection:
    return AnalysisSection(
        enabled=bool(mc.enabled),
        study_type="monte_carlo",
        execution=AnalysisExecutionSection(
            parallel_enabled=bool(mc.parallel_enabled),
            parallel_workers=int(mc.parallel_workers),
        ),
        monte_carlo=AnalysisMonteCarloSection(
            iterations=int(mc.iterations),
            base_seed=int(mc.base_seed),
            variations=list(mc.variations),
        ),
    )


def _normalize_analysis_and_monte_carlo(
    legacy_mc: MonteCarloSection,
    analysis: AnalysisSection,
) -> tuple[MonteCarloSection, AnalysisSection]:
    if analysis.enabled and analysis.study_type == "monte_carlo":
        normalized_mc = MonteCarloSection(
            enabled=True,
            iterations=int(analysis.monte_carlo.iterations),
            base_seed=int(analysis.monte_carlo.base_seed),
            parallel_enabled=bool(analysis.execution.parallel_enabled),
            parallel_workers=int(analysis.execution.parallel_workers),
            variations=list(analysis.monte_carlo.variations),
        )
        return normalized_mc, analysis
    if analysis.enabled and analysis.study_type == "sensitivity":
        normalized_mc = MonteCarloSection(
            enabled=False,
            iterations=max(int(legacy_mc.iterations), 1),
            base_seed=int(legacy_mc.base_seed),
            parallel_enabled=False,
            parallel_workers=0,
            variations=[],
        )
        return normalized_mc, analysis
    if legacy_mc.enabled:
        return legacy_mc, _analysis_from_legacy_monte_carlo(legacy_mc)
    return legacy_mc, analysis


def _parse_outputs_section(value: Any) -> OutputsSection:
    d = _as_dict(value, "outputs")
    out = OutputsSection(
        output_dir=str(d.get("output_dir", "outputs")),
        mode=str(d.get("mode", "interactive")),
        stats=dict(d.get("stats", {}) or {}),
        plots=dict(d.get("plots", {}) or {}),
        animations=dict(d.get("animations", {}) or {}),
        monte_carlo=dict(d.get("monte_carlo", {}) or {}),
        ai_report=dict(d.get("ai_report", {}) or {}),
        ai_config=dict(d.get("ai_config", {}) or {}),
    )
    if out.mode not in ("interactive", "save", "both"):
        raise ValueError("outputs.mode must be one of: interactive, save, both.")
    if not out.output_dir.strip():
        raise ValueError("outputs.output_dir must be non-empty.")
    return out


def scenario_config_from_dict(data: dict[str, Any], source_path: str | Path | None = None) -> SimulationScenarioConfig:
    root = _as_dict(data, "root")
    base_dir = None if source_path is None else Path(source_path).expanduser().resolve().parent
    root = _resolve_agent_presets(root, base_dir=base_dir)
    _enforce_strict_booleans(root)
    legacy_mc = _parse_monte_carlo_section(root.get("monte_carlo"))
    analysis = _parse_analysis_section(root.get("analysis"), legacy_mc=legacy_mc)
    normalized_mc, normalized_analysis = _normalize_analysis_and_monte_carlo(legacy_mc, analysis)
    legacy_keys_present = {key for key in ("rocket", "chaser", "target") if key in root}
    has_objects_section = "objects" in root
    rocket = _parse_agent_section(root.get("rocket"), role="rocket", object_id="rocket", default_kind="rocket")
    chaser = _parse_agent_section(root.get("chaser"), role="chaser", object_id="chaser", default_kind="satellite")
    target = _parse_agent_section(
        root.get("target"),
        role="target",
        object_id="target",
        default_kind="satellite",
        default_enabled=(False if has_objects_section and "target" not in legacy_keys_present else None),
    )
    legacy_agents = {"rocket": rocket, "chaser": chaser, "target": target}
    objects = _parse_objects_section(
        root.get("objects"),
        legacy_agents=legacy_agents,
        has_objects_section=has_objects_section,
        legacy_keys_present=legacy_keys_present,
    )
    rocket = objects.get("rocket", rocket)
    chaser = objects.get("chaser", chaser)
    target = objects.get("target", target)
    cfg = SimulationScenarioConfig(
        scenario_name=str(root.get("scenario_name", "unnamed_scenario")),
        scenario_description=str(root.get("scenario_description", "") or ""),
        rocket=rocket,
        chaser=chaser,
        target=target,
        objects=objects,
        ground_stations=_parse_ground_stations_section(root.get("ground_stations")),
        simulator=_parse_simulator_section(root.get("simulator")),
        outputs=_parse_outputs_section(root.get("outputs")),
        monte_carlo=normalized_mc,
        analysis=normalized_analysis,
        metadata=dict(root.get("metadata", {}) or {}),
    )
    for object_id, section in dict(cfg.objects or {}).items():
        if bool(dict(section.reference_orbit or {}).get("enabled", False)) and (not bool(section.enabled)):
            raise ValueError(f"{object_id}.reference_orbit.enabled requires {object_id}.enabled to be true.")
    return cfg


def load_simulation_yaml(path: str | Path) -> SimulationScenarioConfig:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load simulation YAML configs. Install with `pip install pyyaml`.") from exc
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Simulation YAML root must be a mapping/object.")
    return scenario_config_from_dict(raw, source_path=p)

from __future__ import annotations

from collections.abc import Iterable

from sim.config.scenario_yaml import AgentSection, SimulationScenarioConfig

LEGACY_OBJECT_IDS = {"rocket", "chaser", "target"}


def configured_objects(cfg: SimulationScenarioConfig) -> dict[str, AgentSection]:
    objects = dict(getattr(cfg, "objects", {}) or {})
    if objects:
        return objects
    return {
        "rocket": cfg.rocket,
        "chaser": cfg.chaser,
        "target": cfg.target,
    }


def object_section(cfg: SimulationScenarioConfig, object_id: str) -> AgentSection | None:
    return configured_objects(cfg).get(str(object_id))


def object_parameter_prefix(object_id: str) -> str:
    object_id = str(object_id)
    return object_id if object_id in LEGACY_OBJECT_IDS else f"objects.{object_id}"


def iter_object_sections(
    cfg: SimulationScenarioConfig,
    *,
    enabled_only: bool = False,
    kind: str | None = None,
) -> Iterable[tuple[str, AgentSection]]:
    desired_kind = str(kind).strip().lower() if kind is not None else ""
    for object_id, section in configured_objects(cfg).items():
        if enabled_only and not bool(section.enabled):
            continue
        if desired_kind and str(section.kind).strip().lower() != desired_kind:
            continue
        yield str(object_id), section


def enabled_object_ids(
    cfg: SimulationScenarioConfig,
    *,
    kind: str | None = None,
) -> list[str]:
    return [object_id for object_id, _ in iter_object_sections(cfg, enabled_only=True, kind=kind)]


def relative_reference_for_object(cfg: SimulationScenarioConfig, object_id: str) -> str | None:
    object_id = str(object_id)
    section = object_section(cfg, object_id)
    if section is None:
        return None
    initial_state = dict(getattr(section, "initial_state", {}) or {})
    explicit = str(initial_state.get("relative_to", "") or "").strip()
    if explicit:
        return explicit
    if "relative_to_target_ric" in initial_state and "target" in configured_objects(cfg):
        return "target"
    if object_id == "chaser" and "target" in configured_objects(cfg):
        return "target"
    return None


def default_reference_object_id(
    cfg: SimulationScenarioConfig,
    *,
    available_ids: Iterable[str] | None = None,
) -> str | None:
    available = None if available_ids is None else {str(item) for item in available_ids}

    def is_available(object_id: str) -> bool:
        return available is None or object_id in available

    for object_id, section in iter_object_sections(cfg, enabled_only=True):
        if is_available(object_id) and bool(dict(section.reference_orbit or {}).get("enabled", False)):
            return object_id
    if is_available("target") and "target" in configured_objects(cfg):
        return "target"
    for object_id, _ in iter_object_sections(cfg, enabled_only=True, kind="satellite"):
        if is_available(object_id):
            return object_id
    for object_id, _ in iter_object_sections(cfg, enabled_only=True):
        if is_available(object_id):
            return object_id
    if available:
        return sorted(available)[0]
    return None


def default_pair_object_ids(
    cfg: SimulationScenarioConfig,
    *,
    available_ids: Iterable[str] | None = None,
) -> tuple[str, str] | None:
    available = None if available_ids is None else {str(item) for item in available_ids}

    def is_available(object_id: str) -> bool:
        return available is None or object_id in available

    for object_id, _ in iter_object_sections(cfg, enabled_only=True):
        reference_id = relative_reference_for_object(cfg, object_id)
        if reference_id and object_id != reference_id and is_available(object_id) and is_available(reference_id):
            return object_id, reference_id

    if is_available("chaser") and is_available("target"):
        objects = configured_objects(cfg)
        if "chaser" in objects and "target" in objects:
            return "chaser", "target"

    satellite_ids = [
        object_id
        for object_id, _ in iter_object_sections(cfg, enabled_only=True, kind="satellite")
        if is_available(object_id)
    ]
    if len(satellite_ids) >= 2:
        return satellite_ids[0], satellite_ids[1]

    object_ids = [object_id for object_id, _ in iter_object_sections(cfg, enabled_only=True) if is_available(object_id)]
    if len(object_ids) >= 2:
        return object_ids[0], object_ids[1]

    if available and len(available) >= 2:
        a_id, b_id = sorted(available)[:2]
        return a_id, b_id
    return None

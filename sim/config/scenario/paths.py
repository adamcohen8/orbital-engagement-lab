from __future__ import annotations

from typing import Any

from sim.config.scenario.primitives import (
    _as_dict,
)
from sim.security import ConfigPathPolicy

__all__ = [
    '_validate_config_read_paths',
    '_resolve_geometry_profile_paths',
    '_resolve_geometry_profile_path_in_specs',
]

def _validate_config_read_paths(root: dict[str, Any], path_policy: ConfigPathPolicy | None) -> None:
    if path_policy is None:
        return
    simulator = _as_dict(root.get("simulator"), "simulator")
    frames = _as_dict(simulator.get("frames"), "simulator.frames")
    if frames.get("eop_path") not in (None, ""):
        path_policy.resolve_input_file(
            str(frames["eop_path"]),
            purpose="simulator.frames.eop_path",
            base_dir=path_policy.workspace_root,
            must_exist=False,
        )
    dynamics = _as_dict(simulator.get("dynamics"), "simulator.dynamics")
    orbit = _as_dict(dynamics.get("orbit"), "simulator.dynamics.orbit")
    path_fields = (
        ("drag_eop_path", "simulator.dynamics.orbit.drag_eop_path"),
        ("de440_coeff_path", "simulator.dynamics.orbit.de440_coeff_path"),
        ("de440_eop_path", "simulator.dynamics.orbit.de440_eop_path"),
    )
    for key, purpose in path_fields:
        if orbit.get(key) not in (None, ""):
            path_policy.resolve_input_file(
                str(orbit[key]),
                purpose=purpose,
                base_dir=path_policy.workspace_root,
                must_exist=False,
            )
    sh = _as_dict(orbit.get("spherical_harmonics"), "simulator.dynamics.orbit.spherical_harmonics")
    sh_path_fields = (
        ("coeff_path", "simulator.dynamics.orbit.spherical_harmonics.coeff_path"),
        ("source_path", "simulator.dynamics.orbit.spherical_harmonics.source_path"),
        ("eop_path", "simulator.dynamics.orbit.spherical_harmonics.eop_path"),
    )
    for key, purpose in sh_path_fields:
        if sh.get(key) not in (None, ""):
            path_policy.resolve_input_file(
                str(sh[key]),
                purpose=purpose,
                base_dir=path_policy.workspace_root,
                must_exist=False,
            )
    _resolve_geometry_profile_paths(root, path_policy)


def _resolve_geometry_profile_paths(root: dict[str, Any], path_policy: ConfigPathPolicy) -> None:
    objects = root.get("objects")
    if isinstance(objects, dict):
        for object_id, section in objects.items():
            if isinstance(section, dict):
                _resolve_geometry_profile_path_in_specs(
                    dict(section.get("specs", {}) or {}),
                    section,
                    path_policy,
                    f"objects.{object_id}.specs",
                )


def _resolve_geometry_profile_path_in_specs(
    specs: dict[str, Any],
    section: dict[str, Any],
    path_policy: ConfigPathPolicy,
    purpose_prefix: str,
) -> None:
    if not specs:
        return
    updates: list[tuple[dict[str, Any], str, str]] = []
    for key in ("geometry_profile_path", "area_profile_path", "attitude_area_profile_path"):
        value = specs.get(key)
        if value not in (None, ""):
            updates.append((specs, key, f"{purpose_prefix}.{key}"))
    geometry = specs.get("geometry")
    if isinstance(geometry, dict):
        for key in ("profile_path", "area_profile_path", "attitude_area_profile_path"):
            value = geometry.get(key)
            if value not in (None, ""):
                updates.append((geometry, key, f"{purpose_prefix}.geometry.{key}"))
    aero = specs.get("aero")
    if isinstance(aero, dict):
        for key in ("geometry_profile_path", "area_profile_path"):
            value = aero.get(key)
            if value not in (None, ""):
                updates.append((aero, key, f"{purpose_prefix}.aero.{key}"))
    if not updates:
        return
    for target, key, purpose in updates:
        resolved = path_policy.resolve_input_file(
            str(target[key]),
            purpose=purpose,
            base_dir=path_policy.config_dir,
            must_exist=True,
        )
        target[key] = str(resolved)
    section["specs"] = specs

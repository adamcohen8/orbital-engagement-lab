from __future__ import annotations

import ast
from pathlib import Path

import yaml

from sim.single_run_support import _SatelliteStepper

ROOT = Path(__file__).resolve().parents[2]
LEGACY_FIELDS = {
    "orbit_control",
    "attitude_control",
    "mission_strategy",
    "mission_execution",
    "mission_objectives",
    "bridge",
}


def _satellite_sections(path: Path) -> list[dict[str, object]]:
    root = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(root, dict):
        return []
    sections: list[dict[str, object]] = []
    objects = root.get("objects")
    if isinstance(objects, dict):
        sections.extend(
            section
            for section in objects.values()
            if isinstance(section, dict) and str(section.get("kind", "satellite")) == "satellite"
        )
    for key in ("target", "chaser"):
        section = root.get(key)
        if isinstance(section, dict) and str(section.get("kind", "satellite")) == "satellite":
            sections.append(section)
    return sections


def test_satellite_stepper_has_only_v2_or_trajectory_only_paths() -> None:
    tree = ast.parse(Path(ROOT / "sim/single_run_support.py").read_text(encoding="utf-8"))
    stepper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_SatelliteStepper")
    step = next(node for node in stepper.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    called_attributes = {
        node.func.attr for node in ast.walk(step) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called_attributes == {"_step_v2", "_step_trajectory_only"}
    assert not hasattr(_SatelliteStepper, "_step_v1")


def test_maintained_satellite_yaml_uses_only_flight_software_boundary() -> None:
    roots = (
        ROOT / "configs",
        ROOT / "agents/examples",
        ROOT / "examples",
        ROOT / "sim/game/configs",
        ROOT / "validation/configs",
    )
    checked = 0
    for root in roots:
        for path in sorted(root.rglob("*.yaml")):
            if "outputs" in path.relative_to(ROOT).parts:
                continue
            for section in _satellite_sections(path):
                checked += 1
                assert LEGACY_FIELDS.isdisjoint(section), path
                assert (
                    isinstance(section.get("flight_software"), dict)
                    or section.get("runtime_profile") == "trajectory_only"
                ), path
    assert checked > 100


def test_satellite_factory_does_not_construct_v1_decision_components() -> None:
    source = (ROOT / "sim/runtime/satellite_factory.py").read_text(encoding="utf-8")
    assert "build_satellite_flight_software_runtime" in source
    for assignment in (
        "sensor=None",
        "estimator=None",
        "orbit_controller=None",
        "attitude_controller=None",
        "bridge=None",
        "mission_strategy=None",
        "mission_execution=None",
    ):
        assert assignment in source


def test_v2_ric_pd_transfer_uses_subordinate_guidance_not_the_legacy_command_boundary() -> None:
    source = (ROOT / "sim/gnc/orbit_v2.py").read_text(encoding="utf-8")

    assert "guide_relative_state(" in source
    assert "from sim.core.models import StateBelief" not in source
    assert "StateBelief(" not in source

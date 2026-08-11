from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sim.flight_software.schemas import assert_truth_free, canonical_json_bytes

FORBIDDEN_IMPORT_PREFIXES = (
    "sim.actuators",
    "sim.core",
    "sim.dynamics",
    "sim.runtime",
    "sim.sensors",
)

TRANSITIVE_BOUNDARY_EXCEPTIONS = {
    ("sim/gnc/navigation_v2.py", "sim.core.models"): {"Measurement", "StateBelief"},
    ("sim/gnc/orbit_v2.py", "sim.dynamics.orbit.elements"): {
        "coes_target_state_at_current_true_anomaly",
        "orbital_element_feedback_accel",
    },
    ("sim/gnc/orbit_v2.py", "sim.dynamics.orbit.relative_linear"): {
        "RelativeLinearDynamics",
        "solve_discrete_lqr_gain",
    },
    ("sim/gnc/orbit_v2.py", "sim.dynamics.orbit.two_body"): {"propagate_two_body_rk4"},
}


def test_flight_software_contract_package_has_no_simulator_truth_imports() -> None:
    package = Path(__file__).parents[1] / "flight_software"
    violations: list[str] = []
    for source_path in sorted(package.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = (node.module,)
            for module in imported:
                if module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    violations.append(f"{source_path.name}:{node.lineno}: {module}")
    assert violations == []


def test_builtin_stack_transitive_gnc_boundary_uses_only_explicit_dependencies() -> None:
    sim_root = Path(__file__).parents[1]
    violations: list[str] = []
    observed_exceptions: set[tuple[str, str]] = set()
    for package_name in ("flight_software", "gnc"):
        for source_path in sorted((sim_root / package_name).glob("*.py")):
            relative = source_path.relative_to(sim_root.parent).as_posix()
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [(alias.name, {"*"}) for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imports = [(node.module, {alias.name for alias in node.names})]
                else:
                    continue
                for module, names in imports:
                    if not module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                        continue
                    key = (relative, module)
                    allowed = TRANSITIVE_BOUNDARY_EXCEPTIONS.get(key)
                    if allowed is None or not names <= allowed:
                        violations.append(f"{relative}:{node.lineno}: {module} {sorted(names)}")
                    else:
                        observed_exceptions.add(key)
    assert violations == []
    assert observed_exceptions == set(TRANSITIVE_BOUNDARY_EXCEPTIONS)


def test_runtime_truth_firewall_rejects_truth_types_and_truth_named_fields() -> None:
    fake_truth_type = type("StateTruth", (), {})
    fake_truth_type.__module__ = "sim.core.models"
    with pytest.raises(TypeError, match="forbidden simulator-owned"):
        assert_truth_free(fake_truth_type())
    with pytest.raises(TypeError, match="forbidden simulator-truth field"):
        canonical_json_bytes({"truth_state": {"position_m": [1.0, 2.0, 3.0]}})


def test_truth_firewall_allows_explicit_observable_values() -> None:
    assert canonical_json_bytes({"measured_range_m": 12.0}) == b'{"measured_range_m":12.0}'

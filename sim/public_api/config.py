from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Union

from sim.config import (
    SimulationScenarioConfig,
    load_simulation_yaml,
    scenario_config_from_dict,
)
from sim.scenarios import ScenarioBuilder as ScenarioBuilder
from sim.scenarios import ValidationIssue as ValidationIssue
from sim.security import ConfigPathPolicy
from sim.security.sealed_mode import SealedModePolicy, sealed_mode_enabled

if TYPE_CHECKING:
    from sim.public_api.results import SimulationResult

MetricCallback = Callable[["SimulationResult"], Union[Mapping[str, Any], Any]]
ControllerFactory = Callable[[], Any]
def _canonicalize_api_config_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy Python API dict conveniences without relaxing YAML parsing."""
    root = dict(data)
    objects = dict(root.get("objects", {}) or {})
    for object_id in ("rocket", "chaser", "target"):
        if object_id in root:
            objects.setdefault(object_id, root.pop(object_id))
    if objects:
        root["objects"] = objects

    legacy_mc = root.pop("monte_carlo", None)
    if isinstance(legacy_mc, dict) and bool(legacy_mc.get("enabled", False)):
        variations = list(legacy_mc.get("variations", []) or [])
        for variation in variations:
            if not isinstance(variation, dict):
                continue
            path = str(variation.get("parameter_path", "") or "")
            for object_id in ("rocket", "chaser", "target"):
                prefix = f"{object_id}."
                if path.startswith(prefix):
                    variation["parameter_path"] = f"objects.{object_id}.{path[len(prefix):]}"
                    break
        analysis = dict(root.get("analysis", {}) or {})
        analysis.setdefault("enabled", True)
        analysis.setdefault("study_type", "monte_carlo")
        analysis.setdefault(
            "execution",
            {
                "parallel_enabled": bool(legacy_mc.get("parallel_enabled", False)),
                "parallel_workers": int(legacy_mc.get("parallel_workers", 0) or 0),
            },
        )
        analysis.setdefault(
            "monte_carlo",
            {
                "iterations": int(legacy_mc.get("iterations", 1) or 1),
                "base_seed": int(legacy_mc.get("base_seed", 0) or 0),
                "variations": variations,
            },
        )
        root["analysis"] = analysis
    return root


def _api_sealed_policy(
    *,
    sealed_mode: bool = False,
    sealed_policy: SealedModePolicy | None = None,
) -> SealedModePolicy | None:
    if sealed_policy is not None:
        return sealed_policy
    if sealed_mode_enabled(bool(sealed_mode)):
        return SealedModePolicy()
    return None
@dataclass(frozen=True)
class SimulationConfig:
    scenario: SimulationScenarioConfig
    source_path: Path | None = None

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        path_policy: ConfigPathPolicy | None = None,
        allow_external_config_paths: bool = False,
        allow_external_ai_prompt_files: bool = False,
    ) -> SimulationConfig:
        resolved = Path(path).expanduser().resolve()
        return cls(
            scenario=load_simulation_yaml(
                resolved,
                path_policy=path_policy,
                allow_external_config_paths=allow_external_config_paths,
                allow_external_ai_prompt_files=allow_external_ai_prompt_files,
            ),
            source_path=resolved,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        path_policy: ConfigPathPolicy | None = None,
        source_path: str | Path | None = None,
    ) -> SimulationConfig:
        resolved_source = None if source_path is None else Path(source_path).expanduser().resolve()
        return cls(
            scenario=scenario_config_from_dict(
                _canonicalize_api_config_dict(dict(data)),
                source_path=resolved_source,
                path_policy=path_policy,
            ),
            source_path=resolved_source,
        )

    @property
    def scenario_name(self) -> str:
        return str(self.scenario.scenario_name)

    def to_dict(self) -> dict[str, Any]:
        return self.scenario.to_dict()

    def to_scenario_config(self) -> SimulationScenarioConfig:
        return self.scenario

    def with_seed(self, seed: int) -> SimulationConfig:
        root = self.to_dict()
        root.setdefault("metadata", {})["seed"] = int(seed)
        return SimulationConfig(
            scenario=scenario_config_from_dict(root),
            source_path=self.source_path,
        )

    def with_value(self, parameter_path: str, value: Any) -> SimulationConfig:
        from sim.execution.parameter_paths import set_parameter_path_value

        root = self.to_dict()
        set_parameter_path_value(root, str(parameter_path), value)
        return SimulationConfig(
            scenario=scenario_config_from_dict(root),
            source_path=self.source_path,
        )

    def with_output_dir(self, output_dir: str | Path) -> SimulationConfig:
        root = self.to_dict()
        root.setdefault("outputs", {})["output_dir"] = str(output_dir)
        return SimulationConfig(
            scenario=scenario_config_from_dict(root),
            source_path=self.source_path,
        )

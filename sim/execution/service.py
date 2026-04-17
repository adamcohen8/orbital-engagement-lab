from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sim.config import SimulationScenarioConfig, load_simulation_yaml, validate_scenario_plugins
from sim.single_run import _SingleRunEngine, _coerce_noninteractive_for_automation, _run_single_config

StepCallback = Callable[[int, int], None]
BatchCallback = Callable[[int, int], None]
BatchProgressCallback = Callable[[dict[str, Any]], None]


class SimulationExecutionService:
    """Public-core execution service for deterministic single-run scenarios."""

    def load_config(self, config_path: str | Path) -> SimulationScenarioConfig:
        return _coerce_noninteractive_for_automation(load_simulation_yaml(config_path))

    def validate_config(self, cfg: SimulationScenarioConfig) -> list[str]:
        strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
        if not strict_plugins:
            return []
        return list(validate_scenario_plugins(cfg))

    def study_type(self, cfg: SimulationScenarioConfig) -> str:
        if bool(cfg.analysis.enabled):
            return str(cfg.analysis.study_type or "analysis").strip().lower()
        if bool(cfg.monte_carlo.enabled):
            return "monte_carlo"
        return "single_run"

    def is_batch_analysis(self, cfg: SimulationScenarioConfig) -> bool:
        return self.study_type(cfg) != "single_run"

    def create_engine(
        self,
        cfg: SimulationScenarioConfig,
        *,
        step_callback: StepCallback | None = None,
    ) -> _SingleRunEngine:
        self._reject_batch_analysis(cfg)
        return _SingleRunEngine(cfg, step_callback=step_callback)

    def run_single(
        self,
        cfg: SimulationScenarioConfig,
        *,
        step_callback: StepCallback | None = None,
    ) -> dict[str, Any]:
        self._reject_batch_analysis(cfg)
        return _run_single_config(cfg, step_callback=step_callback)

    def run_session_payload(
        self,
        cfg: SimulationScenarioConfig,
        *,
        source_path: str | Path | None = None,
        step_callback: StepCallback | None = None,
    ) -> dict[str, Any]:
        del source_path
        return self.run_single(cfg, step_callback=step_callback)

    def run_config_file(
        self,
        config_path: str | Path,
        *,
        step_callback: StepCallback | None = None,
        batch_callback: BatchCallback | None = None,
        batch_progress_callback: BatchProgressCallback | None = None,
    ) -> dict[str, Any]:
        del batch_callback, batch_progress_callback
        path = Path(config_path).expanduser().resolve()
        cfg = self.load_config(path)
        errors = self.validate_config(cfg)
        if errors:
            msg = "Plugin validation failed:\n- " + "\n- ".join(errors)
            raise ValueError(msg)
        self._reject_batch_analysis(cfg)
        payload = self.run_single(cfg, step_callback=step_callback)
        return self.wrap_single_file_payload(payload=payload, cfg=cfg, config_path=path)

    def run_batch_from_config(
        self,
        cfg: SimulationScenarioConfig,
        *,
        source_path: str | Path | None = None,
        step_callback: StepCallback | None = None,
    ) -> dict[str, Any]:
        del source_path, step_callback
        self._reject_batch_analysis(cfg)
        raise AssertionError("unreachable")

    def wrap_single_file_payload(
        self,
        *,
        payload: dict[str, Any],
        cfg: SimulationScenarioConfig,
        config_path: str | Path,
    ) -> dict[str, Any]:
        return {
            "config_path": str(Path(config_path).expanduser().resolve()),
            "scenario_name": cfg.scenario_name,
            "scenario_description": cfg.scenario_description,
            "monte_carlo": {"enabled": False},
            "run": dict(payload.get("summary", {}) or {}),
        }

    def _reject_batch_analysis(self, cfg: SimulationScenarioConfig) -> None:
        if self.is_batch_analysis(cfg):
            raise ImportError(
                "Monte Carlo, sensitivity, and other batch-analysis workflows are part of "
                "Orbital Engagement Pro. The public core runs deterministic single scenarios."
            )


_DEFAULT_SERVICE = SimulationExecutionService()


def create_single_run_engine(
    cfg: SimulationScenarioConfig,
    *,
    step_callback: StepCallback | None = None,
) -> _SingleRunEngine:
    return _DEFAULT_SERVICE.create_engine(cfg, step_callback=step_callback)


def run_simulation_scenario(
    cfg: SimulationScenarioConfig,
    *,
    source_path: str | Path | None = None,
    step_callback: StepCallback | None = None,
) -> dict[str, Any]:
    return _DEFAULT_SERVICE.run_session_payload(cfg, source_path=source_path, step_callback=step_callback)


def run_simulation_config_file(
    config_path: str | Path,
    *,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
) -> dict[str, Any]:
    return _DEFAULT_SERVICE.run_config_file(
        config_path,
        step_callback=step_callback,
        batch_callback=batch_callback,
        batch_progress_callback=batch_progress_callback,
    )

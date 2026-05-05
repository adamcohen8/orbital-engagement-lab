from __future__ import annotations

from pathlib import Path
from typing import Any

from sim.config import SimulationScenarioConfig, validate_scenario_plugins
from sim.execution.study import analysis_study_type


def prepare_batch_run_configs(cfg: SimulationScenarioConfig) -> list[dict[str, Any]]:
    study_type = analysis_study_type(cfg)
    if study_type not in {"monte_carlo", "sensitivity"}:
        return []
    root = cfg.to_dict()
    outdir = Path(cfg.outputs.output_dir)
    if study_type == "sensitivity":
        from sim.execution.sensitivity import prepare_sensitivity_runs

        sensitivity_method = str(cfg.analysis.sensitivity.method or "one_at_a_time").strip().lower()
        return prepare_sensitivity_runs(cfg=cfg, root=root, outdir=outdir, sensitivity_method=sensitivity_method)

    from sim.execution.campaigns import prepare_monte_carlo_runs

    return prepare_monte_carlo_runs(cfg=cfg, root=root, outdir=outdir)


def validate_generated_batch_configs(cfg: SimulationScenarioConfig) -> dict[str, Any]:
    strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
    try:
        prepared = prepare_batch_run_configs(cfg)
    except Exception as exc:
        return {
            "run_count": 0,
            "errors": [
                {
                    "iteration": None,
                    "parameter_path": None,
                    "parameter_value": None,
                    "error": str(exc),
                }
            ],
        }

    if analysis_study_type(cfg) == "sensitivity":
        from sim.execution.sensitivity import validate_prepared_sensitivity_runs

        return validate_prepared_sensitivity_runs(prepared=prepared, strict_plugins=strict_plugins)

    errors: list[dict[str, Any]] = []
    for idx, item in enumerate(prepared):
        config_dict = dict(item.get("config_dict", {}) or {})
        try:
            run_cfg = item.get("cfg")
            if run_cfg is None:
                from sim.config import scenario_config_from_dict

                run_cfg = scenario_config_from_dict(config_dict)
            if strict_plugins:
                errors.extend(
                    {
                        "iteration": idx,
                        "parameter_path": None,
                        "parameter_value": None,
                        "error": str(err),
                    }
                    for err in validate_scenario_plugins(run_cfg)
                )
        except Exception as exc:
            errors.append(
                {
                    "iteration": idx,
                    "parameter_path": None,
                    "parameter_value": None,
                    "error": str(exc),
                }
            )
    return {"run_count": len(prepared), "errors": errors}

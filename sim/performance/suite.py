from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from sim.api import SimulationConfig, SimulationSession
from sim.config import validate_scenario_plugins
from sim.security.config_paths import ConfigPathPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "configs" / "performance_benchmark_suite.yaml"

_NONDETERMINISTIC_KEYS = {
    "generated_at",
    "generated_at_utc",
    "generated_utc",
    "config_sha256",
    "runtime_profile",
    "runtime_profile_ms",
    "wall_time_s",
    "elapsed_s",
    "elapsed_ms",
    "mean_step_wall_ms",
    "modeled_execution_duration_ns",
    "host_execution_duration_ns",
    "total_step_wall_s",
    "share_of_step_wall",
    "output_dir",
    "output_index_md",
    "review_sqlite_path",
    "config_source_path",
    "effective_config_path",
    "invocation_path",
    "config_path",
    "plot_outputs",
    "animation_outputs",
    "review_outputs",
    "ground_station_access_report_outputs",
    "artifacts",
    "checkpoint_dir",
}
_NONDETERMINISTIC_SUFFIXES = (
    "_runtime_ms",
    "_wall_time_s",
    "_wall_s",
    "_elapsed_s",
    "_elapsed_ms",
)


@dataclass(frozen=True)
class PerformanceCase:
    name: str
    description: str
    category: str
    kind: str
    config_path: str | None
    tags: tuple[str, ...]
    external: bool
    optional: bool
    engine: str
    acceleration_mode: str
    base_overrides: dict[str, Any]
    checks: dict[str, dict[str, Any]]
    profiles: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class PerformanceManifest:
    suite_name: str
    description: str
    default_profile: str
    output_root: str
    profile_defaults: dict[str, dict[str, Any]]
    cases: tuple[PerformanceCase, ...]
    source_path: Path


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def load_performance_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> PerformanceManifest:
    source_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, "performance manifest")
    profile_defaults = {
        str(name): _mapping(settings, f"profiles.{name}")
        for name, settings in _mapping(root.get("profiles"), "profiles").items()
    }
    cases: list[PerformanceCase] = []
    seen: set[str] = set()
    for index, item in enumerate(list(root.get("cases", []) or [])):
        case_raw = _mapping(item, f"cases[{index}]")
        name = str(case_raw.get("name", "") or "").strip()
        if not name:
            raise ValueError(f"cases[{index}].name is required.")
        if name in seen:
            raise ValueError(f"Duplicate performance case name: {name}")
        seen.add(name)
        profiles = {
            str(profile_name): _mapping(settings, f"cases[{index}].profiles.{profile_name}")
            for profile_name, settings in _mapping(case_raw.get("profiles"), f"cases[{index}].profiles").items()
        }
        cases.append(
            PerformanceCase(
                name=name,
                description=str(case_raw.get("description", "") or ""),
                category=str(case_raw.get("category", "core") or "core"),
                kind=str(case_raw.get("kind", "scenario") or "scenario").strip().lower(),
                config_path=(None if case_raw.get("config_path") in (None, "") else str(case_raw["config_path"])),
                tags=tuple(str(tag) for tag in list(case_raw.get("tags", []) or [])),
                external=bool(case_raw.get("external", False)),
                optional=bool(case_raw.get("optional", False)),
                engine=str(case_raw.get("engine", "oel") or "oel").strip().lower(),
                acceleration_mode=str(case_raw.get("acceleration_mode", "auto") or "auto").strip().lower(),
                base_overrides=_mapping(case_raw.get("overrides"), f"cases[{index}].overrides"),
                checks={
                    str(metric): _mapping(rule, f"cases[{index}].checks.{metric}")
                    for metric, rule in _mapping(case_raw.get("checks"), f"cases[{index}].checks").items()
                },
                profiles=profiles,
            )
        )
    default_profile = str(root.get("default_profile", "standard") or "standard")
    if default_profile not in profile_defaults:
        raise ValueError(f"default_profile {default_profile!r} is not defined under profiles.")
    return PerformanceManifest(
        suite_name=str(root.get("suite_name", "oel_performance") or "oel_performance"),
        description=str(root.get("description", "") or ""),
        default_profile=default_profile,
        output_root=str(root.get("output_root", "outputs/performance") or "outputs/performance"),
        profile_defaults=profile_defaults,
        cases=tuple(cases),
        source_path=source_path,
    )


def _deterministic_payload(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            name = str(key)
            if name in _NONDETERMINISTIC_KEYS or name.endswith(_NONDETERMINISTIC_SUFFIXES):
                continue
            out[name] = _deterministic_payload(child)
        return out
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_deterministic_payload(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def physics_payload_hash(payload: Any) -> str:
    digest = hashlib.sha256()
    for chunk in _iter_deterministic_json(payload):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _iter_deterministic_json(value: Any) -> Iterator[str]:
    """Yield the canonical physics JSON without materializing a second payload."""

    if isinstance(value, dict):
        # _deterministic_payload stringifies keys before json.dumps sorts them.
        # Build the same final key/value mapping while retaining only one level
        # of references, rather than recursively cloning the complete payload.
        children: dict[str, Any] = {}
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            name = str(key)
            if name in _NONDETERMINISTIC_KEYS or name.endswith(_NONDETERMINISTIC_SUFFIXES):
                continue
            children[name] = child
        yield "{"
        for index, name in enumerate(sorted(children)):
            if index:
                yield ","
            yield json.dumps(name, ensure_ascii=True)
            yield ":"
            yield from _iter_deterministic_json(children[name])
        yield "}"
        return
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            yield from _iter_deterministic_json(value.item())
            return
        yield "["
        for index, child in enumerate(value):
            if index:
                yield ","
            yield from _iter_deterministic_json(child)
        yield "]"
        return
    if isinstance(value, np.generic):
        yield from _iter_deterministic_json(value.item())
        return
    if isinstance(value, (list, tuple)):
        yield "["
        for index, child in enumerate(value):
            if index:
                yield ","
            yield from _iter_deterministic_json(child)
        yield "]"
        return
    if isinstance(value, Path):
        value = str(value)
    yield json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)


def _set_parameter(root: dict[str, Any], path: str, value: Any) -> None:
    tokens = [token for token in str(path).split(".") if token]
    if not tokens:
        raise ValueError("Override path cannot be empty.")
    current: dict[str, Any] = root
    for token in tokens[:-1]:
        child = current.get(token)
        if child is None:
            child = {}
            current[token] = child
        if not isinstance(child, dict):
            raise ValueError(f"Override path {path!r} crosses non-mapping field {token!r}.")
        current = child
    current[tokens[-1]] = value


def _merged_case_profile(
    manifest: PerformanceManifest,
    case: PerformanceCase,
    profile: str,
    *,
    warmups: int | None,
    repeats: int | None,
) -> dict[str, Any]:
    if profile not in manifest.profile_defaults:
        choices = ", ".join(sorted(manifest.profile_defaults))
        raise ValueError(f"Unknown performance profile {profile!r}; expected one of: {choices}.")
    merged = dict(manifest.profile_defaults[profile])
    merged.update(case.profiles.get(profile, {}))
    merged["warmups"] = int(merged.get("warmups", 0) if warmups is None else warmups)
    merged["repeats"] = int(merged.get("repeats", 1) if repeats is None else repeats)
    if merged["warmups"] < 0 or merged["repeats"] <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive.")
    return merged


def _case_config_path(case: PerformanceCase) -> Path:
    if case.config_path is None:
        raise ValueError(f"Performance case {case.name!r} requires config_path.")
    path = Path(case.config_path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _effective_scenario_config(
    case: PerformanceCase,
    case_profile: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[SimulationConfig, dict[str, Any]]:
    source_path = _case_config_path(case)
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario config {source_path} must contain a mapping.")
    effective = dict(raw)
    overrides = dict(case.base_overrides)
    overrides.update(_mapping(case_profile.get("overrides"), f"{case.name}.profile.overrides"))
    if case_profile.get("duration_s") is not None:
        overrides["simulator.duration_s"] = float(case_profile["duration_s"])
    if case_profile.get("dt_s") is not None:
        overrides["simulator.dt_s"] = float(case_profile["dt_s"])
    for path, value in overrides.items():
        _set_parameter(effective, str(path), value)
    _set_parameter(effective, "outputs.output_dir", str(output_dir))
    path_policy = ConfigPathPolicy.default(
        config_path=source_path,
        write_roots=(output_dir,),
    )
    cfg = SimulationConfig.from_dict(effective, source_path=source_path, path_policy=path_policy)
    errors = validate_scenario_plugins(cfg.scenario)
    if errors:
        raise ValueError("Plugin validation failed:\n- " + "\n- ".join(errors))
    return cfg, effective


def _extract_metric(payload: Any, path: str) -> Any:
    current = payload
    for token in str(path).split("."):
        if not isinstance(current, dict) or token not in current:
            raise KeyError(path)
        current = current[token]
    return current


def _evaluate_checks(payload: dict[str, Any], checks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path, rule in sorted(checks.items()):
        record: dict[str, Any] = {"metric": path, "rule": dict(rule)}
        try:
            actual = _extract_metric(payload, path)
            passed = True
            if "equals" in rule:
                passed = passed and actual == rule["equals"]
            if "min" in rule:
                passed = passed and float(actual) >= float(rule["min"])
            if "max" in rule:
                passed = passed and float(actual) <= float(rule["max"])
            if "truthy" in rule:
                passed = passed and bool(actual) is bool(rule["truthy"])
            if "one_of" in rule:
                passed = passed and actual in list(rule["one_of"])
            if "min_length" in rule:
                passed = passed and len(actual) >= int(rule["min_length"])
            if "path_exists" in rule:
                exists = Path(str(actual)).is_file()
                passed = passed and exists is bool(rule["path_exists"])
            reported_actual = len(actual) if "min_length" in rule else actual
            record.update({"actual": reported_actual, "passed": bool(passed)})
        except Exception as exc:
            record.update({"actual": None, "passed": False, "error": str(exc)})
        out.append(record)
    return out


def _median_stage_totals(profiles: list[dict[str, Any]]) -> dict[str, float]:
    names = sorted(
        {
            str(name)
            for profile in profiles
            for name in dict(profile.get("stage_totals", {}) or {}).keys()
        }
    )
    return {
        name: float(
            statistics.median(
                float(dict(dict(profile.get("stage_totals", {}) or {}).get(name, {}) or {}).get("total_s", 0.0))
                for profile in profiles
            )
        )
        for name in names
    }


def _runtime_profiles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    summary = dict(payload.get("summary", {}) or {})
    if summary:
        summaries.append(summary)
    for run in list(payload.get("runs", []) or []):
        if isinstance(run, dict):
            run_summary = dict(run.get("summary", {}) or {})
            if run_summary:
                summaries.append(run_summary)
    return [
        profile
        for item in summaries
        if (profile := dict(item.get("runtime_profile", {}) or {}))
    ]


def _aggregate_runtime_profile(payload: dict[str, Any]) -> dict[str, Any]:
    profiles = _runtime_profiles(payload)
    stage_names = {
        str(name)
        for profile in profiles
        for name in dict(profile.get("stage_totals", {}) or {})
    }
    if any(
        "general_propagation_step"
        in dict(dict(object_profile or {}).get("stages", {}) or {})
        for profile in profiles
        for object_profile in dict(profile.get("object_totals", {}) or {}).values()
    ):
        stage_names.add("general_propagation_step")
    stage_totals = {
        name: {
            "total_s": sum(
                float(dict(dict(profile.get("stage_totals", {}) or {}).get(name, {}) or {}).get("total_s", 0.0))
                + (
                    sum(
                        float(
                            dict(
                                dict(dict(object_profile or {}).get("stages", {}) or {}).get(
                                    "general_propagation_step", {}
                                )
                                or {}
                            ).get("total_s", 0.0)
                            or 0.0
                        )
                        for object_profile in dict(profile.get("object_totals", {}) or {}).values()
                    )
                    if name == "general_propagation_step"
                    else 0.0
                )
                for profile in profiles
            )
        }
        for name in stage_names
    }
    executor = dict(profiles[0].get("executor", {}) or {}) if profiles else {}
    return {"stage_totals": stage_totals, "executor": executor}


def _scenario_work_counts(payload: dict[str, Any]) -> dict[str, int]:
    profiles = _runtime_profiles(payload)
    outer_steps = 0
    object_steps = 0
    dynamics_count = 0
    estimator_count = 0
    decision_count = 0
    general_propagation_count = 0
    for profile in profiles:
        outer_steps += int(profile.get("completed_steps", 0) or 0)
        object_steps += int(
            dict(dict(profile.get("stage_totals", {}) or {}).get("object_step", {}) or {}).get("count", 0) or 0
        )
        for object_profile in dict(profile.get("object_totals", {}) or {}).values():
            stages = dict(dict(object_profile or {}).get("stages", {}) or {})
            general_propagation_count += int(
                dict(stages.get("general_propagation_step", {}) or {}).get("count", 0) or 0
            )
            dynamics_count += int(
                dict(stages.get("dynamics_step", stages.get("satellite_step", {})) or {}).get("count", 0)
                or 0
            )
            estimator_count += int(dict(stages.get("estimator_update", {}) or {}).get("count", 0) or 0)
            decision_count += int(dict(stages.get("decision", {}) or {}).get("count", 0) or 0)
    return {
        "outer_steps": outer_steps,
        "object_steps": object_steps,
        "dynamics_steps": dynamics_count,
        "estimator_updates": estimator_count,
        "decisions": decision_count,
        "general_propagation_steps": general_propagation_count,
    }


def _simulated_duration_s(payload: dict[str, Any], fallback: float) -> float:
    run_durations = [
        float(dict(run.get("summary", {}) or {}).get("duration_s", 0.0) or 0.0)
        for run in list(payload.get("runs", []) or [])
        if isinstance(run, dict)
    ]
    if run_durations:
        return float(sum(run_durations))
    return float(dict(payload.get("summary", {}) or {}).get("duration_s", fallback) or 0.0)


def _runtime_backend(payload: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    monte_carlo = dict(payload.get("monte_carlo", {}) or {})
    if monte_carlo.get("enabled"):
        return {
            "workflow": "monte_carlo",
            "parallel_enabled": bool(monte_carlo.get("parallel_enabled", False)),
            "parallel_workers": int(monte_carlo.get("parallel_workers", 1) or 1),
            "parallel_fallback_reason": monte_carlo.get("parallel_fallback_reason"),
            "hierarchical_execution": dict(payload.get("hierarchical_execution", {}) or {}),
        }
    return dict(profile.get("executor", {}) or {})


def _reset_scenario_measurement_caches(case: PerformanceCase) -> str:
    """Keep compilation/data warmups while measuring atmosphere trajectories from a cold epoch cache."""
    if case.category != "atmosphere":
        return "warm_process_state"
    model = str(case.base_overrides.get("simulator.environment.atmosphere_model", "")).strip().lower()
    if model == "jacchia70":
        backend = sys.modules.get("sim.dynamics.orbit.jacchia70_backend")
        if backend is not None:
            backend.clear_trajectory_epoch_caches()
    elif model == "harris_priester":
        backend = sys.modules.get("sim.dynamics.orbit.harris_priester_backend")
        if backend is not None:
            backend.clear_trajectory_epoch_caches()
    return "cold_trajectory_epoch_cache"


def _run_scenario_case(
    case: PerformanceCase,
    case_profile: dict[str, Any],
    *,
    scratch_dir: Path,
) -> dict[str, Any]:
    warmups = int(case_profile["warmups"])
    repeats = int(case_profile["repeats"])
    effective_dir = scratch_dir / "effective_configs"
    effective_dir.mkdir(parents=True, exist_ok=True)
    base_cfg, effective = _effective_scenario_config(
        case,
        case_profile,
        output_dir=scratch_dir / "runs" / "template",
    )
    (effective_dir / f"{case.name}.yaml").write_text(
        yaml.safe_dump(effective, sort_keys=False),
        encoding="utf-8",
    )

    def execute(cfg: SimulationConfig, raw: dict[str, Any], invocation_path: Path) -> dict[str, Any]:
        if case.kind != "artifacts":
            return dict(SimulationSession.from_config(cfg).run().payload)
        from sim.execution import run_simulation_config_file

        invocation_path.parent.mkdir(parents=True, exist_ok=True)
        invocation_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return dict(run_simulation_config_file(invocation_path))

    for index in range(warmups):
        cfg, raw = _effective_scenario_config(
            case,
            case_profile,
            output_dir=scratch_dir / "warmups" / f"{index:03d}",
        )
        execute(cfg, raw, scratch_dir / f"warmup_{index:03d}.yaml")
        gc.collect()

    elapsed: list[float] = []
    propagation_elapsed: list[float] = []
    hashes: list[str] = []
    runtime_profiles: list[dict[str, Any]] = []
    first_measurements: dict[str, Any] | None = None
    measurement_cache_policy = "warm_process_state"
    for index in range(repeats):
        cfg, raw = _effective_scenario_config(
            case,
            case_profile,
            output_dir=scratch_dir / "runs" / f"{index:03d}",
        )
        gc.collect()
        measurement_cache_policy = _reset_scenario_measurement_caches(case)
        started = time.perf_counter()
        payload = execute(cfg, raw, scratch_dir / f"repeat_{index:03d}.yaml")
        elapsed.append(float(time.perf_counter() - started))
        hashes.append(physics_payload_hash(payload))
        runtime_profile = _aggregate_runtime_profile(payload)
        runtime_profiles.append(runtime_profile)
        propagation_elapsed.append(
            float(
                dict(
                    dict(runtime_profile.get("stage_totals", {}) or {}).get(
                        "general_propagation_step", {}
                    )
                    or {}
                ).get("total_s", 0.0)
                or 0.0
            )
        )
        if first_measurements is None:
            summary = dict(payload.get("summary", {}) or {})
            run_summaries = [
                dict(run.get("summary", {}) or {})
                for run in list(payload.get("runs", []) or [])
                if isinstance(run, dict)
            ]
            excerpt_source = summary or (run_summaries[0] if run_summaries else {})
            first_measurements = {
                "duration_s": _simulated_duration_s(payload, float(base_cfg.scenario.simulator.duration_s)),
                "work_counts": _scenario_work_counts(payload),
                "checks": _evaluate_checks(payload, case.checks),
                "runtime_backend": _runtime_backend(payload, runtime_profile),
                "summary_excerpt": {
                    "scenario_name": excerpt_source.get("scenario_name", payload.get("scenario_name")),
                    "samples": sum(int(item.get("samples", 0) or 0) for item in run_summaries)
                    if run_summaries
                    else excerpt_source.get("samples"),
                    "objects": excerpt_source.get("objects"),
                    "terminated_early": excerpt_source.get("terminated_early"),
                    "termination_reason": excerpt_source.get("termination_reason"),
                    "run_count": len(list(payload.get("runs", []) or [])) or 1,
                },
            }
        del payload

    assert first_measurements is not None
    median_elapsed = float(statistics.median(elapsed))
    median_propagation_elapsed = float(statistics.median(propagation_elapsed))
    duration_s = float(first_measurements["duration_s"])
    work_counts = dict(first_measurements["work_counts"])
    checks = list(first_measurements["checks"])
    return {
        "status": "passed" if all(item["passed"] for item in checks) and len(set(hashes)) == 1 else "failed",
        "warmups": warmups,
        "repeats": repeats,
        "measurement_cache_policy": measurement_cache_policy,
        "elapsed_s": elapsed,
        "median_elapsed_s": median_elapsed,
        "min_elapsed_s": float(min(elapsed)),
        "max_elapsed_s": float(max(elapsed)),
        # ``elapsed_s`` remains the backwards-compatible end-to-end session
        # measurement.  The dedicated propagation series is accumulated by
        # the runtime profiler immediately around catalog propagator calls and
        # therefore excludes engine construction, payload assembly, and
        # artifact writes while retaining those operations in every repeat.
        "end_to_end_elapsed_s": elapsed,
        "median_end_to_end_elapsed_s": median_elapsed,
        "min_end_to_end_elapsed_s": float(min(elapsed)),
        "max_end_to_end_elapsed_s": float(max(elapsed)),
        "propagation_elapsed_s": propagation_elapsed,
        "median_propagation_elapsed_s": median_propagation_elapsed,
        "min_propagation_elapsed_s": float(min(propagation_elapsed)),
        "max_propagation_elapsed_s": float(max(propagation_elapsed)),
        "duration_s": duration_s,
        "simulated_seconds_per_wall_second": duration_s / max(median_elapsed, 1.0e-12),
        "work_counts": work_counts,
        "outer_steps_per_wall_second": work_counts["outer_steps"] / max(median_elapsed, 1.0e-12),
        "dynamics_steps_per_wall_second": work_counts["dynamics_steps"] / max(median_elapsed, 1.0e-12),
        "general_propagation_steps_per_propagation_second": work_counts[
            "general_propagation_steps"
        ]
        / max(median_propagation_elapsed, 1.0e-12),
        "physics_hashes": hashes,
        "deterministic_parity": len(set(hashes)) == 1,
        "coverage_checks": checks,
        "coverage_passed": all(item["passed"] for item in checks),
        "runtime_backend": first_measurements["runtime_backend"],
        "median_stage_total_s": _median_stage_totals(runtime_profiles),
        "summary_excerpt": first_measurements["summary_excerpt"],
    }


def _run_attitude_reference_case(case: PerformanceCase, case_profile: dict[str, Any]) -> dict[str, Any]:
    from validation.attitude_reference import load_reference_case, run_oel_attitude_reference

    source_path = _case_config_path(case)
    base = load_reference_case(source_path)
    duration_s = float(case_profile.get("duration_s", base.duration_s))
    step_s = float(case_profile.get("step_s", base.step_s))
    reference_case = replace(
        base,
        duration_s=duration_s,
        step_s=step_s,
        orbit_substep_s=step_s,
        attitude_substep_s=step_s,
    )

    def run_once() -> Any:
        if case.engine == "basilisk":
            from validation.attitude_basilisk_reference import run_basilisk_attitude_reference

            return run_basilisk_attitude_reference(reference_case, record_torque_history=False)
        return run_oel_attitude_reference(
            reference_case,
            acceleration_mode=case.acceleration_mode,
            record_torque_history=False,
        )

    warmups = int(case_profile["warmups"])
    repeats = int(case_profile["repeats"])
    for _ in range(warmups):
        history = run_once()
        del history
        gc.collect()
    elapsed: list[float] = []
    hashes: list[str] = []
    sample_count = 0
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        history = run_once()
        elapsed.append(float(time.perf_counter() - started))
        payload = {
            "time_s": history.t_s,
            "quat_bn": history.quat_bn,
            "rate_body_rad_s": history.rate_body_rad_s,
        }
        hashes.append(physics_payload_hash(payload))
        sample_count = int(history.t_s.size)
        del history
    median_elapsed = float(statistics.median(elapsed))
    nominal_steps = int(round(duration_s / step_s))
    return {
        "status": "passed" if len(set(hashes)) == 1 else "failed",
        "warmups": warmups,
        "repeats": repeats,
        "elapsed_s": elapsed,
        "median_elapsed_s": median_elapsed,
        "min_elapsed_s": float(min(elapsed)),
        "max_elapsed_s": float(max(elapsed)),
        "duration_s": duration_s,
        "step_s": step_s,
        "sample_count": sample_count,
        "work_counts": {"outer_steps": nominal_steps, "object_steps": nominal_steps, "dynamics_steps": nominal_steps},
        "simulated_seconds_per_wall_second": duration_s / max(median_elapsed, 1.0e-12),
        "outer_steps_per_wall_second": nominal_steps / max(median_elapsed, 1.0e-12),
        "dynamics_steps_per_wall_second": nominal_steps / max(median_elapsed, 1.0e-12),
        "physics_hashes": hashes,
        "deterministic_parity": len(set(hashes)) == 1,
        "coverage_checks": [],
        "coverage_passed": True,
        "runtime_backend": {
            "engine": case.engine,
            "acceleration_mode": case.acceleration_mode,
            "output_policy": "trajectory_only",
        },
        "median_stage_total_s": {},
    }


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_rows": 0 if not status else len(status.splitlines()),
    }


def _environment_payload() -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "git": _git_metadata(),
    }


def _report_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['suite_name']} Performance Report",
        "",
        str(payload.get("description", "")),
        "",
        f"Profile: `{payload['profile']}`  ",
        f"Generated: `{payload['environment']['generated_at_utc']}`  ",
        f"Python: `{payload['environment']['python_executable']}`  ",
        f"Git commit: `{payload['environment']['git'].get('commit')}`  ",
        f"Dirty workspace: `{payload['environment']['git'].get('dirty')}`",
        "",
        "Atmosphere timing policy: clear trajectory-epoch caches before every measured repetition; configured warmups may prepare code and static data.",
        "",
        "## Results",
        "",
        "| Case | Category | Status | End-to-end median s | Propagation median s | Sim-time speed | Dynamics steps/s | Propagation steps/s | Parity | Coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["cases"]:
        if item["status"] == "skipped":
            lines.append(
                f"| {item['name']} | {item['category']} | skipped | — | — | — | — | — | — | — |"
            )
            continue
        if "median_elapsed_s" not in item:
            lines.append(
                f"| {item['name']} | {item['category']} | {item['status']} | — | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            "| {name} | {category} | {status} | {median:.6f} | {propagation} | {speed:.1f}x | {dynamics:.1f} | {general_propagation} | {parity} | {coverage} |".format(
                name=item["name"],
                category=item["category"],
                status=item["status"],
                median=float(item["median_elapsed_s"]),
                propagation=(
                    f"{float(item.get('median_propagation_elapsed_s', 0.0)):.6f}"
                    if int(dict(item.get("work_counts", {}) or {}).get("general_propagation_steps", 0) or 0) > 0
                    else "—"
                ),
                speed=float(item["simulated_seconds_per_wall_second"]),
                dynamics=float(item["dynamics_steps_per_wall_second"]),
                general_propagation=(
                    f"{float(item.get('general_propagation_steps_per_propagation_second', 0.0)):.1f}"
                    if int(dict(item.get("work_counts", {}) or {}).get("general_propagation_steps", 0) or 0) > 0
                    else "—"
                ),
                parity="pass" if item["deterministic_parity"] else "FAIL",
                coverage="pass" if item["coverage_passed"] else "FAIL",
            )
        )
    skipped = [item for item in payload["cases"] if item["status"] == "skipped"]
    if skipped:
        lines.extend(["", "## Skipped cases", ""])
        for item in skipped:
            lines.append(f"- `{item['name']}`: {item.get('skip_reason', 'not selected')}")
    failed_checks = [
        (item["name"], check)
        for item in payload["cases"]
        if item["status"] != "skipped"
        for check in item.get("coverage_checks", [])
        if not check.get("passed", False)
    ]
    if failed_checks:
        lines.extend(["", "## Failed coverage assertions", ""])
        for case_name, check in failed_checks:
            lines.append(
                f"- `{case_name}` `{check['metric']}`: actual `{check.get('actual')}`, rule `{check.get('rule')}`"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Timing and physics correctness are separate. Each repeated case must produce one deterministic physics hash, and configured coverage assertions must pass. External validation remains the correctness authority for the applicable physics model.",
            "For OGP cases, propagation timing is measured immediately around catalog propagator calls; end-to-end timing still includes engine construction, payload assembly, and artifact writes.",
            "",
            "Comparisons are meaningful only for the same profile, case manifest, hardware, Python environment, acceleration mode, resource policy, and output policy.",
            "",
        ]
    )
    return "\n".join(lines)


def run_performance_suite(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    profile: str | None = None,
    case_names: set[str] | None = None,
    categories: set[str] | None = None,
    include_external: bool = False,
    warmups: int | None = None,
    repeats: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_performance_manifest(manifest_path)
    profile_name = str(profile or manifest.default_profile)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (REPO_ROOT / manifest.output_root / f"{profile_name}_{stamp}").resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    matplotlib_config_dir = root / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    selected_names = set(case_names or ())
    selected_categories = set(categories or ())
    unknown = selected_names - {case.name for case in manifest.cases}
    if unknown:
        raise ValueError(f"Unknown performance cases: {', '.join(sorted(unknown))}")

    results: list[dict[str, Any]] = []
    for case in manifest.cases:
        if selected_names and case.name not in selected_names:
            continue
        if selected_categories and case.category not in selected_categories:
            continue
        base = {
            "name": case.name,
            "description": case.description,
            "category": case.category,
            "kind": case.kind,
            "tags": list(case.tags),
            "external": case.external,
            "optional": case.optional,
        }
        if case.external and not include_external:
            results.append({**base, "status": "skipped", "skip_reason": "external case; pass --include-external"})
            continue
        case_profile = _merged_case_profile(
            manifest,
            case,
            profile_name,
            warmups=warmups,
            repeats=repeats,
        )
        if not bool(case_profile.get("enabled", True)):
            results.append({**base, "status": "skipped", "skip_reason": f"disabled in profile {profile_name}"})
            continue
        try:
            if case.kind == "attitude_reference":
                measured = _run_attitude_reference_case(case, case_profile)
            elif case.kind in {"scenario", "campaign", "artifacts"}:
                measured = _run_scenario_case(case, case_profile, scratch_dir=root / "scratch" / case.name)
            else:
                raise ValueError(f"Unsupported performance case kind: {case.kind}")
            results.append({**base, **measured})
        except Exception as exc:
            if case.optional:
                results.append({**base, "status": "skipped", "skip_reason": f"optional case unavailable: {exc}"})
            else:
                results.append({**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "schema_version": 1,
        "suite_name": manifest.suite_name,
        "description": manifest.description,
        "profile": profile_name,
        "manifest_path": str(manifest.source_path),
        "manifest_sha256": hashlib.sha256(manifest.source_path.read_bytes()).hexdigest(),
        "environment": _environment_payload(),
        "cases": results,
        "cases_passed": sum(item["status"] == "passed" for item in results),
        "cases_failed": sum(item["status"] == "failed" for item in results),
        "cases_skipped": sum(item["status"] == "skipped" for item in results),
        "output_dir": str(root),
    }
    (root / "benchmark_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    report = _report_markdown(payload)
    (root / "benchmark_report.md").write_text(report, encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the OEL full-path performance benchmark suite.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--profile", choices=("smoke", "standard", "full"), default=None)
    parser.add_argument("--case", action="append", default=[], help="Run only a named case; repeatable.")
    parser.add_argument("--category", action="append", default=[], help="Run only a category; repeatable.")
    parser.add_argument("--include-external", action="store_true", help="Include optional external-engine cases.")
    parser.add_argument("--warmups", type=int, default=None, help="Override warm-up count for every selected case.")
    parser.add_argument("--repeats", type=int, default=None, help="Override repeat count for every selected case.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--list", action="store_true", help="List cases without running them.")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable result payload.")
    parser.add_argument("--fail-on-skip", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_performance_manifest(args.manifest)
    if args.list:
        for case in manifest.cases:
            external = " external" if case.external else ""
            optional = " optional" if case.optional else ""
            print(f"{case.name:32s} {case.category:14s} {case.kind}{external}{optional}")
        return 0
    payload = run_performance_suite(
        args.manifest,
        profile=args.profile,
        case_names=set(args.case),
        categories=set(args.category),
        include_external=bool(args.include_external),
        warmups=args.warmups,
        repeats=args.repeats,
        output_dir=args.output_dir,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))
    else:
        print(_report_markdown(payload))
        print(f"Results: {payload['output_dir']}")
    failed = int(payload["cases_failed"])
    skipped = int(payload["cases_skipped"])
    return 1 if failed or (args.fail_on_skip and skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())

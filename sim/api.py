from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

from sim.config import (
    SimulationScenarioConfig,
    enabled_object_ids,
    load_simulation_yaml,
    scenario_config_from_dict,
    validate_scenario_plugins,
)
from sim.core.models import Command, StateBelief
from sim.execution import create_single_run_engine, run_simulation_scenario
from sim.execution.study import analysis_study_type
from sim.execution.validation import validate_generated_batch_configs
from sim.scenarios import (
    ScenarioArtifact,
    ValidationReport,
)
from sim.scenarios import (
    ScenarioBuilder as ScenarioBuilder,
)
from sim.scenarios import (
    ValidationIssue as ValidationIssue,
)
from sim.security import ConfigPathPolicy

MetricCallback = Callable[["SimulationResult"], Union[Mapping[str, Any], Any]]
ControllerFactory = Callable[[], Any]

_STATE_COLUMNS = [
    "x_eci_km",
    "y_eci_km",
    "z_eci_km",
    "vx_eci_km_s",
    "vy_eci_km_s",
    "vz_eci_km_s",
    "q0",
    "q1",
    "q2",
    "q3",
    "wx_body_rad_s",
    "wy_body_rad_s",
    "wz_body_rad_s",
    "mass_kg",
]
_RIC_STATE_COLUMNS = [
    "r_km",
    "i_km",
    "c_km",
    "rdot_km_s",
    "idot_km_s",
    "cdot_km_s",
]
_ECI_REL_STATE_COLUMNS = [
    "dx_eci_km",
    "dy_eci_km",
    "dz_eci_km",
    "dvx_eci_km_s",
    "dvy_eci_km_s",
    "dvz_eci_km_s",
]


def _closest_approach_metric(payload: dict[str, Any]) -> float:
    from sim.execution.metrics import closest_approach_from_run_payload

    return closest_approach_from_run_payload(payload)


def _as_array_map(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, np.ndarray] = {}
    for key, arr in value.items():
        try:
            out[str(key)] = np.array(arr, dtype=float)
        except (TypeError, ValueError):
            continue
    return out


def _as_nested_array_map(value: Any) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, np.ndarray]] = {}
    for key, inner in value.items():
        if not isinstance(inner, dict):
            continue
        out[str(key)] = _as_array_map(inner)
    return out


def _as_2d_state_history(value: Any, *, width: int = 6) -> np.ndarray:
    arr = np.array(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, width), dtype=float)
    if arr.ndim == 1:
        if arr.size < width:
            raise ValueError(f"State history must have at least {width} columns.")
        return arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < width:
        raise ValueError(f"State history must be a 2D array with at least {width} columns.")
    return arr


def _json_safe_metric_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe_metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metric_value(v) for v in value]
    return value


def _metric_callback_name(callback: MetricCallback, index: int) -> str:
    name = str(getattr(callback, "__name__", "") or "").strip()
    if name and name != "<lambda>":
        return name
    return f"metric_{index}"


def _evaluate_metric_callbacks(
    result: SimulationResult,
    metrics: Mapping[str, MetricCallback] | list[MetricCallback] | tuple[MetricCallback, ...] | None,
) -> dict[str, Any]:
    if metrics is None:
        return {}
    out: dict[str, Any] = {}
    if isinstance(metrics, Mapping):
        iterable = list(metrics.items())
    else:
        iterable = [(_metric_callback_name(callback, idx), callback) for idx, callback in enumerate(metrics)]
    for name, callback in iterable:
        value = callback(result)
        if isinstance(value, Mapping):
            for key, item in value.items():
                out[str(key)] = _json_safe_metric_value(item)
        else:
            out[str(name)] = _json_safe_metric_value(value)
    return out


def _aggregate_custom_metrics(run_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({str(key) for row in run_metrics for key in row.keys()})
    aggregate: dict[str, Any] = {"run_count": int(len(run_metrics))}
    for key in keys:
        values = [row.get(key) for row in run_metrics if key in row]
        bool_values = [bool(v) for v in values if isinstance(v, (bool, np.bool_))]
        if bool_values and len(bool_values) == len(values):
            true_count = int(sum(1 for v in bool_values if v))
            aggregate[key] = {
                "true_count": true_count,
                "false_count": int(len(bool_values) - true_count),
                "probability_true": float(true_count / max(len(bool_values), 1)),
            }
            continue
        numeric: list[float] = []
        for value in values:
            if isinstance(value, (bool, np.bool_)):
                continue
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(x):
                numeric.append(x)
        if numeric:
            arr = np.array(numeric, dtype=float)
            aggregate[key] = {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p05": float(np.percentile(arr, 5)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
            }
    return aggregate


def _range_scale(units: str) -> float:
    key = str(units or "km").strip().lower()
    if key in {"km", "kilometer", "kilometers"}:
        return 1.0
    if key in {"m", "meter", "meters"}:
        return 1000.0
    raise ValueError("range units must be 'km' or 'm'.")


def _records_dataframe(records: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return records
    return pd.DataFrame.from_records(records)


def _numeric_metric_value(row: Mapping[str, Any], metric: str) -> float:
    try:
        return float(row.get(metric))
    except (TypeError, ValueError):
        return float("nan")


def _artifact_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _artifact_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_artifact_paths(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _compatible_call(fn: Callable[..., Any], kwargs: dict[str, Any], fallback_kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return fn(**kwargs)

    filtered: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            return fn(**fallback_kwargs)
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY) and name in kwargs:
            filtered[name] = kwargs[name]

    missing_required = [
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and param.default is inspect.Signature.empty
        and name not in filtered
    ]
    if missing_required:
        return fn(**fallback_kwargs)
    return fn(**filtered)


class _CallableControllerAdapter:
    def __init__(self, fn: Callable[..., Any], *, command_kind: str) -> None:
        self.fn = fn
        self.command_kind = str(command_kind)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        ret = _compatible_call(
            self.fn,
            {
                "belief": belief,
                "state": belief.state,
                "t_s": t_s,
                "budget_ms": budget_ms,
            },
            {
                "belief": belief,
                "t_s": t_s,
            },
        )
        return _coerce_controller_return(ret, command_kind=self.command_kind)


class _CallableMissionAdapter:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def update(self, **kwargs: Any) -> dict[str, Any]:
        ret = _compatible_call(self.fn, dict(kwargs), {"truth": kwargs.get("truth"), "t_s": kwargs.get("t_s", 0.0)})
        return dict(ret) if isinstance(ret, Mapping) else {}


def _coerce_controller_return(value: Any, *, command_kind: str) -> Command:
    if value is None:
        return Command.zero()
    if isinstance(value, Command):
        return value
    if isinstance(value, Mapping):
        cmd = Command.zero()
        if "thrust_eci_km_s2" in value:
            cmd.thrust_eci_km_s2 = np.array(value["thrust_eci_km_s2"], dtype=float).reshape(3)
        elif "accel_eci_km_s2" in value:
            cmd.thrust_eci_km_s2 = np.array(value["accel_eci_km_s2"], dtype=float).reshape(3)
        if "torque_body_nm" in value:
            cmd.torque_body_nm = np.array(value["torque_body_nm"], dtype=float).reshape(3)
        if isinstance(value.get("mode_flags"), Mapping):
            cmd.mode_flags.update(dict(value["mode_flags"]))
        return cmd
    arr = np.array(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise TypeError("Controller callables must return Command, mapping, None, or a length-3 vector.")
    if command_kind == "attitude":
        return Command(torque_body_nm=arr.copy(), mode_flags={"mode": "api_attitude_controller"})
    return Command(thrust_eci_km_s2=arr.copy(), mode_flags={"mode": "api_orbit_controller"})


def _controller_object(value: Any, *, command_kind: str) -> Any:
    if value is None:
        return None
    if hasattr(value, "act") and callable(value.act):
        return value
    if callable(value):
        return _CallableControllerAdapter(value, command_kind=command_kind)
    raise TypeError("Controller override must be a controller object with .act(), a callable, or None.")


def _mission_object(value: Any) -> Any:
    if value is None:
        return None
    if any(callable(getattr(value, name, None)) for name in ("update", "plan", "decide", "execute", "act")):
        return value
    if callable(value):
        return _CallableMissionAdapter(value)
    raise TypeError("Mission override must be an object with a mission method, a callable, or None.")


def _require_private_workflow(module_name: str, symbol_name: str, feature: str) -> Any:
    try:
        module = importlib.import_module(module_name)
        symbol = getattr(module, symbol_name)
    except Exception as exc:
        raise ImportError(f"{feature} are available in the private/product distribution.") from exc
    if getattr(symbol, "__name__", "") == "_unavailable":
        symbol()
    return symbol


class MetricStudyResult(dict):
    """Dictionary result with convenience methods for custom metric studies."""

    def metrics_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for run in list(self.get("runs", []) or []):
            row = {
                "iteration": int(dict(run).get("iteration", len(records))),
                "seed": dict(run).get("seed"),
            }
            sampled = dict(dict(run).get("sampled_parameters", {}) or {})
            row.update({f"sampled.{key}": value for key, value in sampled.items()})
            row.update(dict(dict(run).get("metrics", {}) or {}))
            records.append(row)
        return records

    def metrics_dataframe(self) -> Any:
        return _records_dataframe(self.metrics_records())

    def failures(self, metric: str) -> list[dict[str, Any]]:
        return [dict(run) for run in list(self.get("runs", []) or []) if bool(dict(run).get("metrics", {}).get(metric))]

    def top(self, metric: str, *, n: int = 10, smallest: bool = True) -> list[dict[str, Any]]:
        runs = [dict(run) for run in list(self.get("runs", []) or [])]
        runs = [run for run in runs if np.isfinite(_numeric_metric_value(dict(run).get("metrics", {}) or {}, metric))]
        return sorted(
            runs,
            key=lambda run: _numeric_metric_value(dict(run).get("metrics", {}) or {}, metric),
            reverse=not bool(smallest),
        )[: int(max(n, 0))]


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
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        return cls(scenario=scenario_config_from_dict(dict(data)))

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

@dataclass(frozen=True)
class SimulationSnapshot:
    step_index: int
    time_s: float
    truth: dict[str, np.ndarray]
    belief: dict[str, np.ndarray]
    applied_thrust: dict[str, np.ndarray]
    applied_torque: dict[str, np.ndarray]

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.truth.keys()))


@dataclass
class SimulationResult:
    config: SimulationConfig
    payload: dict[str, Any]

    @property
    def analysis(self) -> dict[str, Any]:
        return dict(self.payload.get("analysis", {}) or {})

    @property
    def analysis_study_type(self) -> str:
        analysis = self.analysis
        if bool(analysis.get("enabled", False)):
            return str(analysis.get("study_type", "unknown"))
        if self.is_monte_carlo:
            return "monte_carlo"
        return "single_run"

    @property
    def is_batch_analysis(self) -> bool:
        return self.analysis_study_type in {"monte_carlo", "sensitivity"}

    @property
    def is_monte_carlo(self) -> bool:
        return bool(dict(self.payload.get("monte_carlo", {}) or {}).get("enabled", False))

    @property
    def summary(self) -> dict[str, Any]:
        if isinstance(self.payload.get("summary"), dict):
            return dict(self.payload["summary"])
        if isinstance(self.payload.get("run"), dict):
            return dict(self.payload["run"])
        return {}

    @property
    def time_s(self) -> np.ndarray:
        return np.array(self.payload.get("time_s", []), dtype=float).reshape(-1)

    @property
    def truth(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("truth_by_object", {}))

    @property
    def target_reference_orbit(self) -> np.ndarray:
        arr = np.array(self.payload.get("target_reference_orbit_truth", []), dtype=float)
        if arr.size == 0:
            return np.empty((0, 6), dtype=float)
        if arr.ndim == 1:
            return arr.reshape(-1, 6)
        return arr

    @property
    def belief(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("belief_by_object", {}))

    @property
    def applied_thrust(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("applied_thrust_by_object", {}))

    @property
    def applied_torque(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("applied_torque_by_object", {}))

    @property
    def knowledge(self) -> dict[str, dict[str, np.ndarray]]:
        return _as_nested_array_map(self.payload.get("knowledge_by_observer", {}))

    @property
    def ground_station_access(self) -> dict[str, Any]:
        return dict(self.payload.get("ground_station_access", {}) or {})

    @property
    def artifacts(self) -> dict[str, Any]:
        if self.is_batch_analysis:
            return dict(self.payload.get("artifacts", {}) or {})
        summary = self.summary
        return {
            "plots": dict(summary.get("plot_outputs", {}) or {}),
            "animations": dict(summary.get("animation_outputs", {}) or {}),
        }

    @property
    def output_dir(self) -> Path:
        return Path(self.config.scenario.outputs.output_dir)

    def review(self) -> Any:
        from sim.review import ReviewWorkspace

        return ReviewWorkspace.open(self.output_dir)

    def evidence_manifest(self) -> dict[str, Any]:
        from sim.plotting.style import get_oel_version

        output_dir = self.output_dir
        review_db = output_dir / "review" / "run.sqlite"
        review_schema = output_dir / "review" / "schema.json"
        review_saved_views = output_dir / "review" / "saved_views.json"
        return {
            "schema_version": 1,
            "scenario_name": self.config.scenario_name,
            "scenario_description": self.config.scenario.scenario_description,
            "study_type": self.analysis_study_type,
            "oel_version": get_oel_version(),
            "config": {
                "source_path": str(self.config.source_path) if self.config.source_path is not None else None,
                "output_dir": str(output_dir),
                "duration_s": float(self.config.scenario.simulator.duration_s),
                "dt_s": float(self.config.scenario.simulator.dt_s),
            },
            "summary": dict(self.summary),
            "metrics": dict(self.metrics),
            "artifacts": _artifact_paths(self.artifacts),
            "review": {
                "enabled": bool(self.config.scenario.outputs.review.enabled),
                "detail": str(self.config.scenario.outputs.review.detail),
                "db_path": str(review_db),
                "db_exists": review_db.is_file(),
                "schema_path": str(review_schema),
                "schema_exists": review_schema.is_file(),
                "saved_views_path": str(review_saved_views),
                "saved_views_exists": review_saved_views.is_file(),
            },
        }

    @property
    def metrics(self) -> dict[str, Any]:
        if self.is_monte_carlo:
            return dict(self.payload.get("aggregate_stats", {}) or {})
        if self.analysis_study_type == "sensitivity":
            return {
                "parameter_count": int(self.analysis.get("parameter_count", 0)),
                "run_count": int(self.analysis.get("run_count", 0)),
                "metrics": list(self.analysis.get("metrics", []) or []),
            }
        out = dict(self.summary)
        closest_approach_km = _closest_approach_metric(self.payload)
        if np.isfinite(closest_approach_km):
            out["closest_approach_km"] = float(closest_approach_km)
        return out

    @property
    def num_steps(self) -> int:
        return int(self.time_s.size)

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.truth.keys()))

    @property
    def reference_object_id(self) -> str | None:
        value = self.summary.get("reference_object_id")
        text = str(value or "").strip()
        return text or None

    @property
    def primary_pair(self) -> tuple[str, str] | None:
        raw = self.summary.get("primary_object_pair")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            a = str(raw[0])
            b = str(raw[1])
            if a and b and a != b:
                return a, b
        if "chaser" in self.truth and "target" in self.truth:
            return "chaser", "target"
        pairs = self.pairs()
        return pairs[0] if pairs else None

    def pairs(self) -> list[tuple[str, str]]:
        ids = list(self.object_ids)
        return [(a, b) for idx, a in enumerate(ids) for b in ids[idx + 1 :]]

    def snapshot(self, step_index: int) -> SimulationSnapshot:
        if self.is_batch_analysis:
            raise RuntimeError("Snapshots are only available for single-run results.")
        if step_index < 0 or step_index >= self.num_steps:
            raise IndexError(f"step_index {step_index} is out of range for {self.num_steps} samples.")

        truth = {
            oid: np.array(hist[step_index], dtype=float)
            for oid, hist in self.truth.items()
            if hist.shape[0] > step_index
        }
        belief = {
            oid: np.array(hist[step_index], dtype=float)
            for oid, hist in self.belief.items()
            if hist.shape[0] > step_index
        }
        thrust = {
            oid: np.array(hist[step_index], dtype=float)
            for oid, hist in self.applied_thrust.items()
            if hist.shape[0] > step_index
        }
        torque = {
            oid: np.array(hist[step_index], dtype=float)
            for oid, hist in self.applied_torque.items()
            if hist.shape[0] > step_index
        }
        return SimulationSnapshot(
            step_index=int(step_index),
            time_s=float(self.time_s[step_index]),
            truth=truth,
            belief=belief,
            applied_thrust=thrust,
            applied_torque=torque,
        )

    def state_history(self, object_id: str) -> np.ndarray:
        oid = str(object_id)
        if oid == "target_reference":
            return self.target_reference_orbit
        histories = self.truth
        if oid not in histories:
            raise KeyError(f"Unknown object_id {oid!r}. Available objects: {sorted(histories.keys())}")
        return _as_2d_state_history(histories[oid], width=6)

    def time_window_mask(self, start_s: float | None = None, end_s: float | None = None) -> np.ndarray:
        t_s = self.time_s
        mask = np.ones(t_s.shape, dtype=bool)
        if start_s is not None:
            mask &= t_s >= float(start_s)
        if end_s is not None:
            mask &= t_s <= float(end_s)
        return mask

    def relative_state(
        self,
        deputy: str,
        chief: str,
        *,
        frame: str = "ric_rect",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> np.ndarray:
        dep = self.state_history(deputy)
        ref = self.state_history(chief)
        t_s = self.time_s
        n = int(min(dep.shape[0], ref.shape[0], t_s.size))
        if n <= 0:
            return np.empty((0, 6), dtype=float)
        dep = dep[:n, :6]
        ref = ref[:n, :6]
        mask = self.time_window_mask(start_s=start_s, end_s=end_s)[:n]
        frame_key = str(frame or "ric_rect").strip().lower()
        if frame_key in {"eci", "inertial"}:
            rel = dep - ref
        elif frame_key in {"ric", "ric_rect", "rect", "rectangular"}:
            from sim.utils.frames import eci_relative_to_ric_rect

            rel = np.vstack([eci_relative_to_ric_rect(dep[k, :6], ref[k, :6]) for k in range(n)])
        elif frame_key in {"ric_curv", "curv", "curvilinear"}:
            from sim.utils.frames import eci_relative_to_ric_rect, ric_rect_to_curv

            rows = []
            for k in range(n):
                rect = eci_relative_to_ric_rect(dep[k, :6], ref[k, :6])
                rows.append(ric_rect_to_curv(rect, r0_km=float(np.linalg.norm(ref[k, :3]))))
            rel = np.vstack(rows)
        else:
            raise ValueError("frame must be one of 'eci', 'ric_rect', or 'ric_curv'.")
        return np.array(rel[mask], dtype=float)

    def range_between(
        self,
        a: str,
        b: str,
        *,
        start_s: float | None = None,
        end_s: float | None = None,
        units: str = "km",
    ) -> np.ndarray:
        rel = self.relative_state(a, b, frame="eci", start_s=start_s, end_s=end_s)
        if rel.size == 0:
            return np.array([], dtype=float)
        return np.linalg.norm(rel[:, :3], axis=1) * _range_scale(units)

    def min_range(
        self,
        a: str,
        b: str,
        *,
        start_s: float | None = None,
        end_s: float | None = None,
        units: str = "km",
    ) -> float:
        ranges = self.range_between(a, b, start_s=start_s, end_s=end_s, units=units)
        finite = ranges[np.isfinite(ranges)]
        return float(np.min(finite)) if finite.size else float("nan")

    def time_of_min_range(
        self,
        a: str,
        b: str,
        *,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> float:
        ranges = self.range_between(a, b, start_s=start_s, end_s=end_s)
        times = self.time_s[self.time_window_mask(start_s=start_s, end_s=end_s)]
        n = int(min(ranges.size, times.size))
        if n <= 0:
            return float("nan")
        finite_idx = np.where(np.isfinite(ranges[:n]))[0]
        if finite_idx.size == 0:
            return float("nan")
        local = int(finite_idx[int(np.argmin(ranges[finite_idx]))])
        return float(times[local])

    def collision_event(
        self,
        a: str,
        b: str,
        *,
        radius_km: float,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> dict[str, Any]:
        min_range_km = self.min_range(a, b, start_s=start_s, end_s=end_s, units="km")
        hit = bool(np.isfinite(min_range_km) and min_range_km <= float(radius_km))
        return {
            "event": hit,
            "threshold_km": float(radius_km),
            "min_range_km": min_range_km,
            "time_s": self.time_of_min_range(a, b, start_s=start_s, end_s=end_s),
        }

    def keepout_violations(
        self,
        a: str,
        b: str,
        *,
        radius_km: float,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> list[dict[str, float]]:
        ranges = self.range_between(a, b, start_s=start_s, end_s=end_s)
        times = self.time_s[self.time_window_mask(start_s=start_s, end_s=end_s)]
        n = int(min(ranges.size, times.size))
        return [
            {"time_s": float(times[idx]), "range_km": float(ranges[idx]), "threshold_km": float(radius_km)}
            for idx in range(n)
            if np.isfinite(ranges[idx]) and float(ranges[idx]) <= float(radius_km)
        ]

    def first_crossing(
        self,
        values: np.ndarray,
        *,
        threshold: float,
        direction: str = "below",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> dict[str, Any]:
        arr = np.array(values, dtype=float).reshape(-1)
        time_s = self.time_s
        window_mask = self.time_window_mask(start_s=start_s, end_s=end_s)
        times = time_s[window_mask]
        if arr.size == time_s.size:
            arr = arr[window_mask]
        elif arr.size != times.size:
            n_total = int(min(arr.size, time_s.size))
            arr = arr[:n_total][window_mask[:n_total]]
            times = time_s[:n_total][window_mask[:n_total]]
        n = int(min(arr.size, times.size))
        key = str(direction or "below").strip().lower()
        if key in {"below", "<", "<="}:
            mask = arr[:n] <= float(threshold)
        elif key in {"above", ">", ">="}:
            mask = arr[:n] >= float(threshold)
        else:
            raise ValueError("direction must be 'below' or 'above'.")
        hits = np.where(mask & np.isfinite(arr[:n]))[0]
        if hits.size == 0:
            return {"event": False, "time_s": None, "value": None, "threshold": float(threshold)}
        idx = int(hits[0])
        return {"event": True, "time_s": float(times[idx]), "value": float(arr[idx]), "threshold": float(threshold)}

    def to_records(self, kind: str = "truth", *, object_id: str) -> list[dict[str, Any]]:
        key = str(kind or "truth").strip().lower()
        source_map = {
            "truth": self.truth,
            "belief": self.belief,
            "applied_thrust": self.applied_thrust,
            "thrust": self.applied_thrust,
            "applied_torque": self.applied_torque,
            "torque": self.applied_torque,
        }
        if key not in source_map:
            raise ValueError("kind must be one of 'truth', 'belief', 'applied_thrust', or 'applied_torque'.")
        data = source_map[key]
        oid = str(object_id)
        if oid not in data:
            raise KeyError(f"Unknown object_id {oid!r}. Available objects: {sorted(data.keys())}")
        arr = np.array(data[oid], dtype=float)
        if arr.ndim != 2:
            return []
        if key in {"applied_thrust", "thrust"}:
            columns = ["ax_eci_km_s2", "ay_eci_km_s2", "az_eci_km_s2"]
        elif key in {"applied_torque", "torque"}:
            columns = ["tx_body_nm", "ty_body_nm", "tz_body_nm"]
        else:
            columns = _STATE_COLUMNS[: arr.shape[1]]
        n = int(min(self.time_s.size, arr.shape[0]))
        records = []
        for idx in range(n):
            row = {"time_s": float(self.time_s[idx]), "object_id": oid}
            row.update({columns[j]: float(arr[idx, j]) for j in range(min(len(columns), arr.shape[1]))})
            records.append(row)
        return records

    def to_dataframe(self, kind: str = "truth", *, object_id: str) -> Any:
        return _records_dataframe(self.to_records(kind, object_id=object_id))

    def relative_records(
        self,
        deputy: str,
        chief: str,
        *,
        frame: str = "ric_rect",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> list[dict[str, Any]]:
        rel = self.relative_state(deputy, chief, frame=frame, start_s=start_s, end_s=end_s)
        times = self.time_s[self.time_window_mask(start_s=start_s, end_s=end_s)]
        n = int(min(rel.shape[0], times.size))
        frame_key = str(frame or "ric_rect").strip().lower()
        columns = _ECI_REL_STATE_COLUMNS if frame_key in {"eci", "inertial"} else _RIC_STATE_COLUMNS
        records = []
        for idx in range(n):
            row = {"time_s": float(times[idx]), "deputy": str(deputy), "chief": str(chief), "frame": frame_key}
            row.update({columns[j]: float(rel[idx, j]) for j in range(min(len(columns), rel.shape[1]))})
            records.append(row)
        return records

    def relative_dataframe(
        self,
        deputy: str,
        chief: str,
        *,
        frame: str = "ric_rect",
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> Any:
        return _records_dataframe(
            self.relative_records(deputy, chief, frame=frame, start_s=start_s, end_s=end_s)
        )

    def evaluate_metrics(
        self,
        metrics: Mapping[str, MetricCallback] | list[MetricCallback] | tuple[MetricCallback, ...],
    ) -> dict[str, Any]:
        return _evaluate_metric_callbacks(self, metrics)


class SimulationSession:
    def __init__(self, config: ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any]):
        self._base_config = self._coerce_config(config)
        self._active_config = self._base_config
        self._result: SimulationResult | None = None
        self._step_index = 0
        self._done = False
        self._engine: Any | None = None
        self._external_intent_providers: dict[str, Callable[..., dict[str, Any] | None]] = {}
        self._controller_overrides: dict[tuple[str, str], ControllerFactory] = {}
        self._mission_overrides: dict[tuple[str, str], ControllerFactory] = {}
        self._controller_originals: dict[tuple[str, str], Any] = {}
        self._mission_originals: dict[tuple[str, str], Any] = {}

    @classmethod
    def from_config(
        cls,
        config: ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> SimulationSession:
        return cls(config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimulationSession:
        return cls(SimulationConfig.from_yaml(path))

    @staticmethod
    def _coerce_config(
        config: ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> SimulationConfig:
        if isinstance(config, ScenarioArtifact):
            return config.to_config()
        if isinstance(config, SimulationConfig):
            return config
        if isinstance(config, SimulationScenarioConfig):
            return SimulationConfig(config)
        if isinstance(config, dict):
            return SimulationConfig.from_dict(config)
        raise TypeError(f"Unsupported config type: {type(config)!r}")

    @property
    def config(self) -> SimulationConfig:
        return self._active_config

    @property
    def result(self) -> SimulationResult | None:
        return self._result

    @property
    def done(self) -> bool:
        if self._engine is not None:
            return bool(self._engine.done)
        return bool(self._done)

    def reset(self, seed: int | None = None) -> SimulationSnapshot | None:
        self._active_config = self._base_config.with_seed(seed) if seed is not None else self._base_config
        self._result = None
        self._step_index = 0
        self._done = False
        self._engine = None
        if self._is_batch_analysis(self._active_config.scenario):
            return None
        self._ensure_engine()
        assert self._engine is not None
        snap = self._engine.snapshot(0)
        return SimulationSnapshot(
            step_index=int(snap["step_index"]),
            time_s=float(snap["time_s"]),
            truth=dict(snap["truth"]),
            belief=dict(snap["belief"]),
            applied_thrust=dict(snap["applied_thrust"]),
            applied_torque=dict(snap["applied_torque"]),
        )

    def run(self, *, step_callback: Any | None = None) -> SimulationResult:
        if self._is_batch_analysis(self._active_config.scenario):
            if self._controller_overrides or self._mission_overrides:
                raise RuntimeError("Runtime API controller/mission overrides are only supported for single-run sessions.")
            payload = self._run_batch_analysis(self._active_config)
            self._result = SimulationResult(config=self._active_config, payload=payload)
            self._done = True
            return self._result

        self._ensure_engine(step_callback=step_callback)
        assert self._engine is not None
        payload = self._engine.run()
        self._result = SimulationResult(config=self._active_config, payload=payload)
        self._step_index = max(self._result.num_steps - 1, 0)
        self._done = True
        return self._result

    def step(self) -> SimulationSnapshot:
        if self._is_batch_analysis(self._active_config.scenario):
            raise RuntimeError("SimulationSession.step() is only available for single-run scenarios.")
        self._ensure_engine()
        assert self._engine is not None
        snap = self._engine.step()
        self._step_index = int(snap["step_index"])
        self._done = bool(self._engine.done)
        return SimulationSnapshot(
            step_index=int(snap["step_index"]),
            time_s=float(snap["time_s"]),
            truth=dict(snap["truth"]),
            belief=dict(snap["belief"]),
            applied_thrust=dict(snap["applied_thrust"]),
            applied_torque=dict(snap["applied_torque"]),
        )

    def set_external_intent_provider(
        self,
        object_id: str,
        provider: Callable[..., dict[str, Any] | None] | None,
    ) -> None:
        oid = str(object_id)
        if provider is None:
            self._external_intent_providers.pop(oid, None)
        else:
            self._external_intent_providers[oid] = provider
        if self._engine is not None and hasattr(self._engine, "set_external_intent_provider"):
            self._engine.set_external_intent_provider(oid, provider)

    def set_orbit_controller(self, object_id: str, controller: Any | None) -> None:
        """Attach a trusted Python orbit controller object or callable to a single-run session."""

        self._set_controller_override("orbit", object_id, None if controller is None else lambda: controller)

    def set_orbit_controller_factory(self, object_id: str, factory: ControllerFactory | None) -> None:
        """Attach a factory that creates a fresh orbit controller for each engine reset."""

        self._set_controller_override("orbit", object_id, factory)

    def set_attitude_controller(self, object_id: str, controller: Any | None) -> None:
        """Attach a trusted Python attitude controller object or callable to a single-run session."""

        self._set_controller_override("attitude", object_id, None if controller is None else lambda: controller)

    def set_attitude_controller_factory(self, object_id: str, factory: ControllerFactory | None) -> None:
        """Attach a factory that creates a fresh attitude controller for each engine reset."""

        self._set_controller_override("attitude", object_id, factory)

    def set_mission_strategy(self, object_id: str, strategy: Any | None) -> None:
        self._set_mission_override("strategy", object_id, None if strategy is None else lambda: strategy)

    def set_mission_strategy_factory(self, object_id: str, factory: ControllerFactory | None) -> None:
        self._set_mission_override("strategy", object_id, factory)

    def set_mission_execution(self, object_id: str, execution: Any | None) -> None:
        self._set_mission_override("execution", object_id, None if execution is None else lambda: execution)

    def set_mission_execution_factory(self, object_id: str, factory: ControllerFactory | None) -> None:
        self._set_mission_override("execution", object_id, factory)

    def _set_controller_override(
        self,
        controller_kind: str,
        object_id: str,
        factory: ControllerFactory | None,
    ) -> None:
        kind = str(controller_kind)
        oid = str(object_id)
        key = (kind, oid)
        if factory is None:
            self._controller_overrides.pop(key, None)
        elif not callable(factory):
            raise TypeError("Controller factory must be callable.")
        else:
            self._controller_overrides[key] = factory
        if self._engine is not None:
            self._apply_single_controller_override(kind, oid, factory)

    def _set_mission_override(
        self,
        mission_kind: str,
        object_id: str,
        factory: ControllerFactory | None,
    ) -> None:
        kind = str(mission_kind)
        oid = str(object_id)
        key = (kind, oid)
        if factory is None:
            self._mission_overrides.pop(key, None)
        elif not callable(factory):
            raise TypeError("Mission factory must be callable.")
        else:
            self._mission_overrides[key] = factory
        if self._engine is not None:
            self._apply_single_mission_override(kind, oid, factory)

    def _ensure_engine(self, *, step_callback: Any | None = None) -> None:
        if self._engine is not None:
            if step_callback is not None:
                self._engine.active_step_callback = step_callback
                emit = getattr(self._engine, "_emit_step_callback", None)
                if callable(emit):
                    emit(getattr(self._engine, "current_index", 0))
            return
        scenario = self._active_config.to_scenario_config()
        self._validate_plugins_if_strict(scenario)
        self._engine = create_single_run_engine(scenario, step_callback=step_callback)
        self._controller_originals.clear()
        self._mission_originals.clear()
        if hasattr(self._engine, "set_external_intent_provider"):
            for object_id, provider in self._external_intent_providers.items():
                self._engine.set_external_intent_provider(object_id, provider)
        self._apply_runtime_overrides()

    def _apply_runtime_overrides(self) -> None:
        for (kind, object_id), factory in self._controller_overrides.items():
            self._apply_single_controller_override(kind, object_id, factory)
        for (kind, object_id), factory in self._mission_overrides.items():
            self._apply_single_mission_override(kind, object_id, factory)

    def _agent_for_override(self, object_id: str) -> Any:
        assert self._engine is not None
        agents = getattr(self._engine, "agents", {})
        oid = str(object_id)
        if oid not in agents:
            raise KeyError(f"No active object with id '{oid}' in this session.")
        return agents[oid]

    def _apply_single_controller_override(
        self,
        kind: str,
        object_id: str,
        factory: ControllerFactory | None,
    ) -> None:
        key = (str(kind), str(object_id))
        attr = "attitude_controller" if kind == "attitude" else "orbit_controller"
        if factory is None:
            if self._engine is not None and key in self._controller_originals:
                agent = self._agent_for_override(object_id)
                current = getattr(agent, attr, None)
                original = self._controller_originals.pop(key)
                if current is not None and hasattr(current, "base"):
                    current.base = original
                    if hasattr(current, "_last_eval_t_s"):
                        current._last_eval_t_s = None
                    if hasattr(current, "_last_cmd"):
                        current._last_cmd = Command.zero()
                else:
                    setattr(agent, attr, original)
            return
        agent = self._agent_for_override(object_id)
        controller = _controller_object(factory(), command_kind=kind)
        current = getattr(agent, attr, None)
        if current is not None and hasattr(current, "base"):
            self._controller_originals.setdefault(key, current.base)
            current.base = controller
            if hasattr(current, "_last_eval_t_s"):
                current._last_eval_t_s = None
            if hasattr(current, "_last_cmd"):
                current._last_cmd = Command.zero()
        else:
            self._controller_originals.setdefault(key, current)
            setattr(agent, attr, controller)

    def _apply_single_mission_override(
        self,
        kind: str,
        object_id: str,
        factory: ControllerFactory | None,
    ) -> None:
        key = (str(kind), str(object_id))
        attr = "mission_execution" if kind == "execution" else "mission_strategy"
        if factory is None:
            if self._engine is not None and key in self._mission_originals:
                agent = self._agent_for_override(object_id)
                setattr(agent, attr, self._mission_originals.pop(key))
            return
        agent = self._agent_for_override(object_id)
        self._mission_originals.setdefault(key, getattr(agent, attr, None))
        setattr(agent, attr, _mission_object(factory()))

    @staticmethod
    def _validate_plugins_if_strict(config: SimulationScenarioConfig) -> None:
        if not bool(config.simulator.plugin_validation.get("strict", True)):
            return
        errors = validate_scenario_plugins(config)
        if errors:
            msg = "Plugin validation failed:\n- " + "\n- ".join(errors)
            raise ValueError(msg)

    @staticmethod
    def _is_batch_analysis(config: SimulationScenarioConfig) -> bool:
        return bool(config.monte_carlo.enabled or config.analysis.enabled)

    @staticmethod
    def _run_batch_analysis(config: SimulationConfig) -> dict[str, Any]:
        return run_simulation_scenario(config.to_scenario_config(), source_path=config.source_path)


class SimulationWorkspace:
    """Higher-level programmatic facade for CLI-equivalent workflows."""

    def __init__(
        self,
        *,
        allow_external_config_paths: bool = False,
        allow_external_ai_prompt_files: bool = False,
        read_roots: Iterable[str | Path] = (),
        write_roots: Iterable[str | Path] = (),
        workspace_root: str | Path | None = None,
    ) -> None:
        self.allow_external_config_paths = bool(allow_external_config_paths)
        self.allow_external_ai_prompt_files = bool(allow_external_ai_prompt_files)
        self.read_roots = tuple(read_roots)
        self.write_roots = tuple(write_roots)
        self.workspace_root = workspace_root

    def _path_policy_for(self, path: str | Path) -> ConfigPathPolicy:
        return ConfigPathPolicy.default(
            config_path=path,
            workspace_root=self.workspace_root,
            read_roots=self.read_roots,
            write_roots=self.write_roots,
            allow_external_config_paths=self.allow_external_config_paths,
            allow_external_ai_prompt_files=self.allow_external_ai_prompt_files,
        )

    def load(self, path: str | Path) -> SimulationConfig:
        return SimulationConfig.from_yaml(path, path_policy=self._path_policy_for(path))

    def from_dict(self, data: dict[str, Any]) -> SimulationConfig:
        return SimulationConfig.from_dict(data)

    def session(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> SimulationSession:
        return SimulationSession.from_config(self._coerce_config(config))

    def artifact(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> ScenarioArtifact:
        if isinstance(config, ScenarioArtifact):
            return config
        if isinstance(config, (str, Path)):
            return ScenarioArtifact(self.load(config))
        return ScenarioArtifact.from_config(config)

    def save_config(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        path: str | Path,
        *,
        validate: bool = True,
    ) -> Path:
        artifact = self.artifact(config)
        if validate:
            report = self.validate(artifact)
            if not bool(report.get("ok", False)):
                errors = [str(item) for item in list(report.get("errors", []) or [])]
                detail = "\n- " + "\n- ".join(errors) if errors else ""
                raise ValueError(f"Cannot save invalid scenario artifact.{detail}")
        return artifact.write(path)

    def run(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        *,
        step_callback: Any | None = None,
    ) -> SimulationResult:
        return self.session(config).run(step_callback=step_callback)

    def run_payload(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        *,
        step_callback: Any | None = None,
    ) -> dict[str, Any]:
        return self.run(config, step_callback=step_callback).payload

    def evaluate_metrics(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        metrics: Mapping[str, MetricCallback] | list[MetricCallback] | tuple[MetricCallback, ...],
        *,
        step_callback: Any | None = None,
    ) -> dict[str, Any]:
        return self.run(config, step_callback=step_callback).evaluate_metrics(metrics)

    def run_monte_carlo_metrics(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        metrics: Mapping[str, MetricCallback] | list[MetricCallback] | tuple[MetricCallback, ...],
        *,
        step_callback: Any | None = None,
        batch_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        sim_config = self._coerce_config(config)
        cfg = sim_config.to_scenario_config()
        if analysis_study_type(cfg) != "monte_carlo":
            raise ValueError("run_monte_carlo_metrics() requires a Monte Carlo scenario.")
        from sim.config import scenario_config_from_dict
        from sim.execution.campaigns import prepare_monte_carlo_runs

        root = cfg.to_dict()
        prepared = prepare_monte_carlo_runs(cfg=cfg, root=root, outdir=Path(cfg.outputs.output_dir))
        strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
        runs: list[dict[str, Any]] = []
        total = len(prepared)
        for done, item in enumerate(prepared, start=1):
            iteration = int(item["iteration"])
            config_dict = dict(item["config_dict"])
            config_dict.setdefault("monte_carlo", {})["enabled"] = False
            config_dict.setdefault("analysis", {})["enabled"] = False
            run_cfg = scenario_config_from_dict(config_dict)
            if strict_plugins:
                errors = validate_scenario_plugins(run_cfg)
                if errors:
                    msg = f"Plugin validation failed in Monte Carlo iteration {iteration}:\n- " + "\n- ".join(errors)
                    raise ValueError(msg)
            run_result = SimulationSession.from_config(SimulationConfig(run_cfg)).run(step_callback=step_callback)
            custom_metrics = run_result.evaluate_metrics(metrics)
            runs.append(
                {
                    "iteration": iteration,
                    "seed": int(item.get("seed", run_cfg.metadata.get("seed", 0))),
                    "sampled_parameters": dict(item.get("sampled_parameters", {}) or {}),
                    "summary": run_result.summary,
                    "metrics": custom_metrics,
                }
            )
            if batch_callback is not None:
                batch_callback(done, total)
        run_metrics = [dict(row.get("metrics", {}) or {}) for row in runs]
        return MetricStudyResult({
            "scenario_name": cfg.scenario_name,
            "monte_carlo": {"enabled": True, "iterations": int(len(runs))},
            "runs": runs,
            "custom_metrics": _aggregate_custom_metrics(run_metrics),
        })

    def sweep(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
        *,
        parameter: str,
        values: list[Any] | tuple[Any, ...],
        metrics: Mapping[str, MetricCallback] | list[MetricCallback] | tuple[MetricCallback, ...] | None = None,
        output_dir_template: str | None = None,
        step_callback: Any | None = None,
        batch_callback: Callable[[int, int], None] | None = None,
    ) -> MetricStudyResult:
        _require_private_workflow("sim.execution.sensitivity", "prepare_sensitivity_runs", "Parameter sweeps")
        base = self._coerce_config(config)
        runs: list[dict[str, Any]] = []
        total = len(list(values))
        for idx, value in enumerate(list(values)):
            cfg_i = base.with_value(parameter, value)
            if output_dir_template:
                cfg_i = cfg_i.with_output_dir(
                    str(output_dir_template).format(index=idx, value=value, scenario=cfg_i.scenario_name)
                )
            result = SimulationSession.from_config(cfg_i).run(step_callback=step_callback)
            custom_metrics = result.evaluate_metrics(metrics) if metrics is not None else {}
            runs.append(
                {
                    "iteration": idx,
                    "seed": result.config.scenario.metadata.get("seed"),
                    "sampled_parameters": {str(parameter): value},
                    "summary": result.summary,
                    "metrics": custom_metrics,
                }
            )
            if batch_callback is not None:
                batch_callback(idx + 1, total)
        run_metrics = [dict(row.get("metrics", {}) or {}) for row in runs]
        return MetricStudyResult({
            "scenario_name": base.scenario_name,
            "sweep": {"parameter": str(parameter), "values": list(values), "run_count": int(len(runs))},
            "runs": runs,
            "custom_metrics": _aggregate_custom_metrics(run_metrics),
        })

    def validate(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> dict[str, Any]:
        try:
            sim_config = self._coerce_config(config)
        except Exception as exc:
            return {
                "ok": False,
                "status": "failed",
                "errors": [str(exc)],
                "config_path": self._config_path_text(config),
            }

        cfg = sim_config.to_scenario_config()
        study_type = analysis_study_type(cfg)
        plugin_errors = list(validate_scenario_plugins(cfg))
        strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
        generated: dict[str, Any] = {"run_count": 0, "errors": []}
        if (not plugin_errors or not strict_plugins) and study_type in {"monte_carlo", "sensitivity"}:
            generated = validate_generated_batch_configs(cfg)

        generated_errors = list(generated.get("errors", []) or [])
        errors: list[Any] = []
        if strict_plugins:
            errors.extend(plugin_errors)
        errors.extend(generated_errors)
        return {
            "ok": not errors,
            "status": "ok" if not errors else "failed",
            "config_path": str(sim_config.source_path) if sim_config.source_path is not None else None,
            "scenario_name": cfg.scenario_name,
            "scenario_description": cfg.scenario_description,
            "study_type": study_type,
            "objects": enabled_object_ids(cfg),
            "duration_s": float(cfg.simulator.duration_s),
            "dt_s": float(cfg.simulator.dt_s),
            "output_dir": str(cfg.outputs.output_dir),
            "plugins": {
                "strict": strict_plugins,
                "status": "ok" if not plugin_errors else ("failed" if strict_plugins else "warn"),
                "errors": plugin_errors,
            },
            "generated": generated,
            "errors": errors,
        }

    def validate_report(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> ValidationReport:
        return ValidationReport.from_validation_dict(self.validate(config))

    def validate_controller_bench(
        self,
        config_path: str | Path,
        *,
        compare_names: list[str] | None = None,
    ) -> dict[str, Any]:
        from sim.controller_lab import validate_controller_bench_config

        report = validate_controller_bench_config(config_path, compare_names=compare_names)
        errors = list(report.get("errors", []) or [])
        report["ok"] = not errors
        report["status"] = "ok" if not errors else "failed"
        return report

    def run_controller_bench(
        self,
        config_path: str | Path,
        *,
        compare_names: list[str] | None = None,
    ) -> dict[str, Any]:
        from sim.controller_lab import run_controller_bench

        return run_controller_bench(config_path, compare_names=compare_names)

    def estimate_ai_report_cost(
        self,
        config_path: str | Path,
        *,
        output_dir: str | Path = "",
        controller_bench: bool = False,
        ai_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if controller_bench:
            estimate = _require_private_workflow(
                "run_simulation",
                "_estimate_ai_report_from_controller_bench",
                "AI report workflows",
            )
            return estimate(str(config_path), output_dir=str(output_dir or ""), ai_options=dict(ai_options or {}))
        estimate = _require_private_workflow("run_simulation", "_estimate_ai_report_from_outputs", "AI report workflows")
        return estimate(str(config_path), output_dir=str(output_dir or ""), ai_options=dict(ai_options or {}))

    def create_ai_report(
        self,
        config_path: str | Path,
        *,
        output_dir: str | Path = "",
        controller_bench: bool = False,
        ai_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = dict(ai_options or {})
        allow_custom_endpoint = bool(options.get("allow_custom_endpoint", False))
        if controller_bench:
            create = _require_private_workflow(
                "run_simulation",
                "_create_ai_report_from_controller_bench",
                "AI report workflows",
            )
            return create(
                str(config_path),
                output_dir=str(output_dir or ""),
                ai_options=options,
                allow_custom_endpoint=allow_custom_endpoint,
            )
        create = _require_private_workflow("run_simulation", "_create_ai_report_from_outputs", "AI report workflows")
        return create(
            str(config_path),
            output_dir=str(output_dir or ""),
            ai_options=options,
            allow_custom_endpoint=allow_custom_endpoint,
        )

    def estimate_ai_config_cost(
        self,
        config_path: str | Path,
        *,
        prompt: str = "",
        prompt_file: str | Path = "",
        output_dir: str | Path = "",
        ai_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        estimate = _require_private_workflow("run_simulation", "_estimate_ai_config_cost", "AI config workflows")

        return estimate(
            str(config_path),
            prompt=str(prompt or ""),
            prompt_file=str(prompt_file or ""),
            output_dir=str(output_dir or ""),
            ai_options=dict(ai_options or {}),
        )

    def create_ai_config(
        self,
        config_path: str | Path,
        *,
        prompt: str = "",
        prompt_file: str | Path = "",
        output_dir: str | Path = "",
        ai_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        create = _require_private_workflow("run_simulation", "_create_ai_config_draft", "AI config workflows")
        options = dict(ai_options or {})

        return create(
            str(config_path),
            prompt=str(prompt or ""),
            prompt_file=str(prompt_file or ""),
            output_dir=str(output_dir or ""),
            ai_options=options,
            allow_custom_endpoint=bool(options.get("allow_custom_endpoint", False)),
        )

    @staticmethod
    def _config_path_text(config: Any) -> str | None:
        if isinstance(config, (str, Path)):
            return str(Path(config).expanduser())
        if isinstance(config, ScenarioArtifact) and config.source_path is not None:
            return str(config.source_path)
        return None

    def _coerce_config(
        self,
        config: str | Path | ScenarioArtifact | SimulationConfig | SimulationScenarioConfig | dict[str, Any],
    ) -> SimulationConfig:
        if isinstance(config, ScenarioArtifact):
            return config.to_config()
        if isinstance(config, (str, Path)):
            return SimulationConfig.from_yaml(config, path_policy=self._path_policy_for(config))
        return SimulationSession._coerce_config(config)

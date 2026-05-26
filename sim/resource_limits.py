from __future__ import annotations

import os
import platform
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sim.acceleration.settings import acceleration_settings_from_config
from sim.config import iter_object_sections


class SimulationMemoryBudgetError(MemoryError):
    """Raised before a run allocates more history memory than the active budget allows."""


class ResourcePressureError(RuntimeError):
    """Raised when a guarded batch run should not start on the current machine."""


@dataclass(frozen=True)
class HistoryMemoryEstimate:
    samples: int
    active_objects: int
    knowledge_pairs: int
    array_bytes: int
    estimated_peak_bytes: int
    limit_bytes: int

    @property
    def estimated_peak_mb(self) -> float:
        return float(self.estimated_peak_bytes) / (1024.0 * 1024.0)

    @property
    def limit_mb(self) -> float:
        return float(self.limit_bytes) / (1024.0 * 1024.0)


DEFAULT_MAX_HISTORY_MEMORY_MB = 1024.0
_ENV_MAX_HISTORY_MEMORY_MB = "OEL_MAX_HISTORY_MEMORY_MB"


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    description: str
    max_parallel_workers: int | None = None
    force_serial: bool = False
    disable_plots: bool = False
    checkpoint_enabled: bool = True
    throttle_enabled: bool = True
    min_available_memory_mb: float | None = None
    hard_min_available_memory_mb: float | None = None
    max_load_per_cpu: float | None = None
    pause_seconds: float = 10.0
    max_wait_s: float = 1800.0


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp_s: float
    cpu_count: int
    load_1m: float | None = None
    total_memory_mb: float | None = None
    available_memory_mb: float | None = None
    platform: str = platform.system()

    @property
    def load_per_cpu(self) -> float | None:
        if self.load_1m is None or self.cpu_count <= 0:
            return None
        return float(self.load_1m) / float(self.cpu_count)


@dataclass(frozen=True)
class ResourceEstimate:
    profile: str
    study_type: str
    runs: int
    steps_per_run: int
    requested_workers: int
    effective_workers: int
    active_objects: int
    plots_enabled: bool
    checkpoint_enabled: bool
    estimated_history_mb_per_run: float
    estimated_parallel_history_mb: float
    current_available_memory_mb: float | None
    load_per_cpu: float | None
    acceleration_mode: str
    acceleration_backend: str
    risk: str
    notes: tuple[str, ...] = ()


RESOURCE_PROFILES: dict[str, ResourceProfile] = {
    "off": ResourceProfile(
        name="off",
        description="Do not apply resource safety rewrites or runtime pressure checks.",
        checkpoint_enabled=False,
        throttle_enabled=False,
    ),
    "config": ResourceProfile(
        name="config",
        description="Use the config as written, but keep checkpointing and pressure checks enabled.",
    ),
    "laptop-safe": ResourceProfile(
        name="laptop-safe",
        description="One run at a time, plots disabled, checkpoint/resume enabled, and conservative pressure gates.",
        max_parallel_workers=1,
        force_serial=True,
        disable_plots=True,
        checkpoint_enabled=True,
        throttle_enabled=True,
        min_available_memory_mb=1536.0,
        hard_min_available_memory_mb=768.0,
        max_load_per_cpu=1.25,
        pause_seconds=15.0,
    ),
    "standard": ResourceProfile(
        name="standard",
        description="A balanced local profile with checkpointing, throttling, and at most two workers.",
        max_parallel_workers=2,
        checkpoint_enabled=True,
        throttle_enabled=True,
        min_available_memory_mb=1024.0,
        hard_min_available_memory_mb=512.0,
        max_load_per_cpu=1.75,
        pause_seconds=10.0,
    ),
    "aggressive": ResourceProfile(
        name="aggressive",
        description="Checkpoint/resume stays on, but worker count follows the config unless the system is already unsafe.",
        checkpoint_enabled=True,
        throttle_enabled=True,
        min_available_memory_mb=512.0,
        hard_min_available_memory_mb=256.0,
        max_load_per_cpu=3.0,
        pause_seconds=5.0,
    ),
}


def _positive_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    out = float(value)
    if out <= 0:
        raise ValueError("max history memory must be positive.")
    return out


def configured_history_memory_limit_mb(cfg: Any) -> float:
    external_limit = _positive_float_or_none(os.environ.get(_ENV_MAX_HISTORY_MEMORY_MB))
    if external_limit is None:
        external_limit = DEFAULT_MAX_HISTORY_MEMORY_MB

    resource_limits = dict(getattr(getattr(cfg, "outputs", None), "resource_limits", {}) or {})
    config_limit = _positive_float_or_none(resource_limits.get("max_history_memory_mb"))
    if config_limit is None:
        return float(external_limit)
    return float(min(external_limit, config_limit))


def bytes_from_mb(value_mb: float) -> int:
    return int(float(value_mb) * 1024.0 * 1024.0)


def format_bytes_mb(value: int) -> str:
    return f"{float(value) / (1024.0 * 1024.0):.2f} MB"


def enforce_history_memory_budget(estimate: HistoryMemoryEstimate) -> None:
    if estimate.estimated_peak_bytes <= estimate.limit_bytes:
        return
    raise SimulationMemoryBudgetError(
        "Estimated simulation history memory exceeds the active budget: "
        f"estimated_peak={format_bytes_mb(estimate.estimated_peak_bytes)}, "
        f"limit={format_bytes_mb(estimate.limit_bytes)}, "
        f"samples={estimate.samples}, active_objects={estimate.active_objects}, "
        f"knowledge_pairs={estimate.knowledge_pairs}. "
        f"Raise the caller-controlled cap with {_ENV_MAX_HISTORY_MEMORY_MB} or "
        "--max-history-memory-mb, or reduce duration/dt/object count."
    )


def resource_profile(name: str | None) -> ResourceProfile:
    key = str(name or "config").strip().lower()
    if not key:
        key = "config"
    if key not in RESOURCE_PROFILES:
        available = ", ".join(sorted(RESOURCE_PROFILES))
        raise ValueError(f"Unknown resource profile {name!r}. Available: {available}")
    return RESOURCE_PROFILES[key]


def _dict_section(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if isinstance(value, dict):
        return dict(value)
    return {}


def apply_resource_profile_to_config_dict(root: dict[str, Any], profile_name: str | None) -> dict[str, Any]:
    """Return a config copy with conservative resource-profile rewrites applied."""
    profile = resource_profile(profile_name)
    out = deepcopy(dict(root))
    if profile.name == "off":
        simulator = _dict_section(out, "simulator")
        simulator["resource_profile"] = "off"
        out["simulator"] = simulator
        outputs = _dict_section(out, "outputs")
        resource_limits = _dict_section(outputs, "resource_limits")
        resource_limits["checkpoint_enabled"] = False
        resource_limits["throttle_enabled"] = False
        outputs["resource_limits"] = resource_limits
        mc_outputs = _dict_section(outputs, "monte_carlo")
        mc_outputs["checkpoint_enabled"] = False
        outputs["monte_carlo"] = mc_outputs
        out["outputs"] = outputs
        return out

    simulator = _dict_section(out, "simulator")
    simulator["resource_profile"] = profile.name
    out["simulator"] = simulator

    outputs = _dict_section(out, "outputs")
    resource_limits = _dict_section(outputs, "resource_limits")
    resource_limits.setdefault("checkpoint_enabled", bool(profile.checkpoint_enabled))
    resource_limits.setdefault("throttle_enabled", bool(profile.throttle_enabled))
    if profile.min_available_memory_mb is not None:
        resource_limits.setdefault("min_available_memory_mb", float(profile.min_available_memory_mb))
    if profile.hard_min_available_memory_mb is not None:
        resource_limits.setdefault("hard_min_available_memory_mb", float(profile.hard_min_available_memory_mb))
    if profile.max_load_per_cpu is not None:
        resource_limits.setdefault("max_load_per_cpu", float(profile.max_load_per_cpu))
    resource_limits.setdefault("resource_pause_seconds", float(profile.pause_seconds))
    resource_limits.setdefault("resource_max_wait_s", float(profile.max_wait_s))
    outputs["resource_limits"] = resource_limits

    if profile.disable_plots:
        plots = _dict_section(outputs, "plots")
        plots["enabled"] = False
        outputs["plots"] = plots

    mc_outputs = _dict_section(outputs, "monte_carlo")
    mc_outputs.setdefault("checkpoint_enabled", bool(profile.checkpoint_enabled))
    outputs["monte_carlo"] = mc_outputs
    out["outputs"] = outputs

    mc = _dict_section(out, "monte_carlo")
    if mc.get("enabled", False):
        if profile.force_serial:
            mc["parallel_enabled"] = False
            mc["parallel_workers"] = 1
        elif profile.max_parallel_workers is not None:
            current = int(mc.get("parallel_workers", 0) or 0)
            if current <= 0:
                mc["parallel_workers"] = int(profile.max_parallel_workers)
            else:
                mc["parallel_workers"] = int(max(1, min(current, profile.max_parallel_workers)))
        out["monte_carlo"] = mc

    analysis = _dict_section(out, "analysis")
    execution = _dict_section(analysis, "execution")
    if analysis.get("enabled", False):
        if profile.force_serial:
            execution["parallel_enabled"] = False
            execution["parallel_workers"] = 1
        elif profile.max_parallel_workers is not None:
            current = int(execution.get("parallel_workers", 0) or 0)
            if current <= 0:
                execution["parallel_workers"] = int(profile.max_parallel_workers)
            else:
                execution["parallel_workers"] = int(max(1, min(current, profile.max_parallel_workers)))
        analysis["execution"] = execution
        out["analysis"] = analysis
    return out


def _read_proc_mem_available_mb() -> tuple[float | None, float | None]:
    meminfo = "/proc/meminfo"
    if not os.path.exists(meminfo):
        return None, None
    values: dict[str, float] = {}
    try:
        with open(meminfo, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    values[parts[0].rstrip(":")] = float(parts[1]) / 1024.0
    except OSError:
        return None, None
    return values.get("MemTotal"), values.get("MemAvailable")


def _run_text(command: list[str], timeout_s: float = 1.0) -> str:
    try:
        proc = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_s)
    except Exception:
        return ""
    return proc.stdout.strip()


def _read_macos_memory_mb() -> tuple[float | None, float | None]:
    if platform.system() != "Darwin":
        return None, None
    total_mb: float | None = None
    total_text = _run_text(["sysctl", "-n", "hw.memsize"])
    try:
        total_mb = float(total_text) / (1024.0 * 1024.0) if total_text else None
    except ValueError:
        total_mb = None

    vm_text = _run_text(["vm_stat"])
    if not vm_text:
        return total_mb, None
    page_size = 4096.0
    available_pages = 0.0
    for line in vm_text.splitlines():
        if "page size of" in line:
            words = [w for w in line.replace(".", "").split() if w.isdigit()]
            if words:
                page_size = float(words[-1])
        label, _, raw_value = line.partition(":")
        key = label.strip()
        value_text = raw_value.strip().rstrip(".").replace(",", "")
        try:
            pages = float(value_text)
        except ValueError:
            continue
        if key in {"Pages free", "Pages speculative"}:
            available_pages += pages
    available_mb = (available_pages * page_size) / (1024.0 * 1024.0) if available_pages > 0 else None
    return total_mb, available_mb


def current_resource_snapshot() -> ResourceSnapshot:
    cpu_count = int(os.cpu_count() or 1)
    load_1m: float | None = None
    try:
        load_1m = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        load_1m = None
    total_mb, available_mb = _read_proc_mem_available_mb()
    if total_mb is None and available_mb is None:
        total_mb, available_mb = _read_macos_memory_mb()
    return ResourceSnapshot(
        timestamp_s=time.monotonic(),
        cpu_count=cpu_count,
        load_1m=load_1m,
        total_memory_mb=total_mb,
        available_memory_mb=available_mb,
    )


def _resource_limits_dict(cfg: Any) -> dict[str, Any]:
    limits = dict(getattr(getattr(cfg, "outputs", None), "resource_limits", {}) or {})
    simulator_profile = getattr(getattr(cfg, "simulator", None), "resource_profile", None)
    if simulator_profile not in (None, ""):
        limits["resource_profile"] = str(simulator_profile)
    return limits


def checkpoint_enabled(cfg: Any, mc_out_cfg: dict[str, Any] | None = None) -> bool:
    mc_out = dict(mc_out_cfg or {})
    if "checkpoint_enabled" in mc_out:
        return bool(mc_out.get("checkpoint_enabled"))
    limits = _resource_limits_dict(cfg)
    if "checkpoint_enabled" in limits:
        return bool(limits.get("checkpoint_enabled"))
    return True


class ResourceGovernor:
    def __init__(self, cfg: Any, *, emit: Any = None) -> None:
        limits = _resource_limits_dict(cfg)
        profile = resource_profile(limits.get("resource_profile", "config"))
        self.profile = profile
        self.enabled = bool(limits.get("throttle_enabled", profile.throttle_enabled))
        self.min_available_memory_mb = _optional_float(
            limits.get("min_available_memory_mb"), profile.min_available_memory_mb
        )
        self.hard_min_available_memory_mb = _optional_float(
            limits.get("hard_min_available_memory_mb"), profile.hard_min_available_memory_mb
        )
        self.max_load_per_cpu = _optional_float(limits.get("max_load_per_cpu"), profile.max_load_per_cpu)
        self.pause_seconds = float(limits.get("resource_pause_seconds", profile.pause_seconds) or profile.pause_seconds)
        self.max_wait_s = float(limits.get("resource_max_wait_s", profile.max_wait_s) or profile.max_wait_s)
        self.emit = emit

    def pressure_reasons(self, snapshot: ResourceSnapshot) -> list[str]:
        reasons: list[str] = []
        if self.min_available_memory_mb is not None and snapshot.available_memory_mb is not None:
            if snapshot.available_memory_mb < self.min_available_memory_mb:
                reasons.append(
                    f"available memory {snapshot.available_memory_mb:.0f} MB < {self.min_available_memory_mb:.0f} MB"
                )
        if self.max_load_per_cpu is not None and snapshot.load_per_cpu is not None:
            if snapshot.load_per_cpu > self.max_load_per_cpu:
                reasons.append(f"load/core {snapshot.load_per_cpu:.2f} > {self.max_load_per_cpu:.2f}")
        return reasons

    def assert_safe_to_start(self) -> ResourceSnapshot:
        snapshot = current_resource_snapshot()
        if not self.enabled:
            return snapshot
        if (
            self.hard_min_available_memory_mb is not None
            and snapshot.available_memory_mb is not None
            and snapshot.available_memory_mb < self.hard_min_available_memory_mb
        ):
            raise ResourcePressureError(
                "System memory is below the hard safety floor: "
                f"available={snapshot.available_memory_mb:.0f} MB, "
                f"floor={self.hard_min_available_memory_mb:.0f} MB."
            )
        return snapshot

    def wait_for_capacity(self, *, context: str = "batch run") -> ResourceSnapshot:
        snapshot = self.assert_safe_to_start()
        if not self.enabled:
            return snapshot
        started = time.monotonic()
        warned = False
        while True:
            reasons = self.pressure_reasons(snapshot)
            if not reasons:
                return snapshot
            if time.monotonic() - started >= self.max_wait_s:
                raise ResourcePressureError(
                    f"Resource pressure remained high before {context}: " + "; ".join(reasons)
                )
            if self.emit is not None and not warned:
                self.emit(
                    f"Throttling {context}: " + "; ".join(reasons) + f"; waiting {self.pause_seconds:.0f}s."
                )
                warned = True
            time.sleep(max(float(self.pause_seconds), 0.1))
            snapshot = self.assert_safe_to_start()


def _optional_float(value: Any, default: float | None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def _active_object_count(cfg: Any) -> int:
    object_ids = {str(object_id) for object_id, _agent in iter_object_sections(cfg, enabled_only=True)}
    return max(len(object_ids), 1)


def _study_type(cfg: Any) -> str:
    analysis = getattr(cfg, "analysis", None)
    if bool(getattr(analysis, "enabled", False)):
        return str(getattr(analysis, "study_type", "monte_carlo") or "monte_carlo").strip().lower()
    if bool(getattr(getattr(cfg, "monte_carlo", None), "enabled", False)):
        return "monte_carlo"
    return "single_run"


def estimate_resource_requirements(cfg: Any) -> ResourceEstimate:
    study_type = _study_type(cfg)
    steps = int(max(float(getattr(cfg.simulator, "duration_s", 0.0)) / max(float(getattr(cfg.simulator, "dt_s", 1.0)), 1e-9), 0))
    runs = int(getattr(cfg.monte_carlo, "iterations", 1) if study_type == "monte_carlo" else 1)
    parallel_enabled = bool(getattr(cfg.monte_carlo, "parallel_enabled", False)) if study_type == "monte_carlo" else False
    requested_workers = int(getattr(cfg.monte_carlo, "parallel_workers", 0) or 0) if parallel_enabled else 1
    if parallel_enabled and requested_workers <= 0:
        requested_workers = max(1, int(os.cpu_count() or 1) - 1)
    effective_workers = max(1, min(requested_workers, max(runs, 1)))
    limits = _resource_limits_dict(cfg)
    profile = resource_profile(limits.get("resource_profile", "config"))
    if profile.force_serial:
        effective_workers = 1
    elif profile.max_parallel_workers is not None:
        effective_workers = max(1, min(effective_workers, int(profile.max_parallel_workers)))
    active_objects = _active_object_count(cfg)
    estimated_history_mb_per_run = float(max(steps + 1, 1) * active_objects * 32 * 8) / (1024.0 * 1024.0)
    estimated_parallel_history_mb = estimated_history_mb_per_run * effective_workers
    plots_enabled = bool(getattr(getattr(cfg, "outputs", None), "plots", {}).get("enabled", True))
    snapshot = current_resource_snapshot()
    acceleration = acceleration_settings_from_config(cfg)
    min_available_mb = _optional_float(limits.get("min_available_memory_mb"), profile.min_available_memory_mb)
    hard_min_available_mb = _optional_float(
        limits.get("hard_min_available_memory_mb"), profile.hard_min_available_memory_mb
    )
    notes: list[str] = []
    risk = "safe"
    if runs >= 5 or steps >= 10000:
        risk = "moderate"
        notes.append("long campaign envelope")
    if effective_workers >= 3:
        risk = "heavy"
        notes.append("three or more concurrent workers")
    if plots_enabled and runs > 1:
        notes.append("plots enabled for a batch workflow")
    if snapshot.available_memory_mb is not None and estimated_parallel_history_mb > snapshot.available_memory_mb * 0.35:
        risk = "unsafe"
        notes.append("estimated parallel history memory is high relative to currently available memory")
    if hard_min_available_mb is not None and snapshot.available_memory_mb is not None:
        if snapshot.available_memory_mb < hard_min_available_mb:
            risk = "unsafe"
            notes.append("current available memory is below the hard safety floor")
    if min_available_mb is not None and snapshot.available_memory_mb is not None:
        if snapshot.available_memory_mb < min_available_mb and risk != "unsafe":
            risk = "heavy"
            notes.append("current available memory is below the profile's preferred start threshold")
    if snapshot.load_per_cpu is not None and snapshot.load_per_cpu > 2.0:
        risk = "heavy" if risk != "unsafe" else risk
        notes.append("current system load is already elevated")
    return ResourceEstimate(
        profile=str(_resource_limits_dict(cfg).get("resource_profile", "config") or "config"),
        study_type=study_type,
        runs=runs,
        steps_per_run=steps,
        requested_workers=requested_workers,
        effective_workers=effective_workers,
        active_objects=active_objects,
        plots_enabled=plots_enabled,
        checkpoint_enabled=checkpoint_enabled(cfg),
        estimated_history_mb_per_run=estimated_history_mb_per_run,
        estimated_parallel_history_mb=estimated_parallel_history_mb,
        current_available_memory_mb=snapshot.available_memory_mb,
        load_per_cpu=snapshot.load_per_cpu,
        acceleration_mode=acceleration.requested_mode,
        acceleration_backend=acceleration.effective_backend,
        risk=risk,
        notes=tuple(notes),
    )

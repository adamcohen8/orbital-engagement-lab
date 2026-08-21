"""Normalized ONP, OGP, and completed-run histories for orbital analysis."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from sim.analysis.directed_link import (
    DirectedLinkConfig,
    DirectedLinkResult,
    LinkEndpointHistory,
    evaluate_directed_link,
    evaluate_directed_link_sample,
    spacecraft_endpoint_history,
)
from sim.analysis.global_coverage import GlobalCoverageConfig, GlobalCoverageResult, evaluate_global_coverage
from sim.analysis.healpix import healpix_wgs84_centers
from sim.analysis.observer_target_geometry import evaluate_surface_targets_ecef
from sim.dynamics.orbit.frames import FrameContext, eci_to_ecef_rotation_context
from sim.utils.geodesy import WGS84_A_KM, WGS84_B_KM
from sim.utils.quaternion import normalize_quaternion, quaternion_to_dcm_bn


@dataclass(frozen=True)
class AnalysisState:
    time_s: float
    position_eci_km: np.ndarray
    velocity_eci_km_s: np.ndarray
    attitude_quat_bn: np.ndarray | None


@dataclass(frozen=True)
class AnalysisHistory:
    object_id: str
    product_kind: str
    state_provider_id: str
    frame: str
    initial_jd_utc: float
    times_s: np.ndarray
    position_eci_km: np.ndarray
    velocity_eci_km_s: np.ndarray
    attitude_quat_bn: np.ndarray | None = None
    attitude_source_kind: str = "not_required"
    attitude_provider_id: str | None = None
    refinement_source: str = "history_hermite_slerp"
    evaluator_at_time: Callable[[float], AnalysisState] | None = None

    def __post_init__(self) -> None:
        for name in ("object_id", "product_kind", "state_provider_id"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value)
        if str(self.frame).strip().lower() != "eci":
            raise ValueError("Orbital analysis histories must be expressed in canonical ECI.")
        object.__setattr__(self, "frame", "eci")
        epoch = float(self.initial_jd_utc)
        if not np.isfinite(epoch):
            raise ValueError("initial_jd_utc must be finite.")
        object.__setattr__(self, "initial_jd_utc", epoch)
        times = np.asarray(self.times_s, dtype=float).reshape(-1)
        position = np.asarray(self.position_eci_km, dtype=float)
        velocity = np.asarray(self.velocity_eci_km_s, dtype=float)
        if times.size < 1 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be finite and strictly increasing.")
        if position.shape != (times.size, 3) or velocity.shape != position.shape:
            raise ValueError("Position and velocity histories must have shape (samples, 3).")
        if np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity)):
            raise ValueError("Position and velocity histories must be finite.")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "position_eci_km", position)
        object.__setattr__(self, "velocity_eci_km_s", velocity)
        attitude = self.attitude_quat_bn
        source_kind = str(self.attitude_source_kind or "").strip().lower()
        provider_id = None if self.attitude_provider_id is None else str(self.attitude_provider_id).strip()
        if attitude is None:
            if source_kind != "not_required" or provider_id:
                raise ValueError("Missing attitude must be labeled not_required with no provider.")
        else:
            attitude = np.asarray(attitude, dtype=float)
            if attitude.shape != (times.size, 4) or np.any(~np.isfinite(attitude)):
                raise ValueError("Attitude history must be finite with shape (samples, 4).")
            norms = np.linalg.norm(attitude, axis=1)
            if np.any(np.abs(norms - 1.0) > 1.0e-10):
                raise ValueError("Attitude history must contain normalized quaternions.")
            if source_kind not in {"achieved", "replay", "analytic_ideal"} or not provider_id:
                raise ValueError("Attitude evidence requires a supported source kind and provider ID.")
            object.__setattr__(self, "attitude_quat_bn", attitude)
        object.__setattr__(self, "attitude_source_kind", source_kind)
        object.__setattr__(self, "attitude_provider_id", provider_id or None)
        refinement_source = str(self.refinement_source or "").strip()
        if not refinement_source:
            raise ValueError("refinement_source must be a non-empty string.")
        object.__setattr__(self, "refinement_source", refinement_source)

    def state_at(self, time_s: float) -> AnalysisState:
        value = float(time_s)
        if not np.isfinite(value) or value < self.times_s[0] or value > self.times_s[-1]:
            raise ValueError("Requested refinement epoch is outside the retained history.")
        if self.evaluator_at_time is not None:
            state = self.evaluator_at_time(value)
            return _validated_state(value, state, require_attitude=self.attitude_quat_bn is not None)
        index = int(np.searchsorted(self.times_s, value, side="right") - 1)
        index = min(max(index, 0), self.times_s.size - 1)
        if index == self.times_s.size - 1 or value == self.times_s[index]:
            attitude = None if self.attitude_quat_bn is None else self.attitude_quat_bn[index].copy()
            return AnalysisState(value, self.position_eci_km[index].copy(), self.velocity_eci_km_s[index].copy(), attitude)
        t0, t1 = float(self.times_s[index]), float(self.times_s[index + 1])
        dt = t1 - t0
        u = (value - t0) / dt
        p0, p1 = self.position_eci_km[index], self.position_eci_km[index + 1]
        v0, v1 = self.velocity_eci_km_s[index], self.velocity_eci_km_s[index + 1]
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        position = h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1
        dh00 = (6.0 * u**2 - 6.0 * u) / dt
        dh10 = 3.0 * u**2 - 4.0 * u + 1.0
        dh01 = (-6.0 * u**2 + 6.0 * u) / dt
        dh11 = 3.0 * u**2 - 2.0 * u
        velocity = dh00 * p0 + dh10 * v0 + dh01 * p1 + dh11 * v1
        attitude = None
        if self.attitude_quat_bn is not None:
            attitude = _slerp(self.attitude_quat_bn[index], self.attitude_quat_bn[index + 1], u)
        return AnalysisState(value, position, velocity, attitude)

    def with_attitude_replay(
        self,
        attitudes_quat_bn: np.ndarray,
        *,
        attitude_source_kind: str,
        attitude_provider_id: str,
    ) -> AnalysisHistory:
        """Attach explicit same-epoch replay or analytic-ideal attitude to an OGP history."""

        if self.attitude_quat_bn is not None:
            raise ValueError("Normalized history already contains attitude evidence.")
        return AnalysisHistory(
            object_id=self.object_id,
            product_kind=self.product_kind,
            state_provider_id=self.state_provider_id,
            frame=self.frame,
            initial_jd_utc=self.initial_jd_utc,
            times_s=self.times_s,
            position_eci_km=self.position_eci_km,
            velocity_eci_km_s=self.velocity_eci_km_s,
            attitude_quat_bn=attitudes_quat_bn,
            attitude_source_kind=attitude_source_kind,
            attitude_provider_id=attitude_provider_id,
            refinement_source=self.refinement_source,
        )

    def link_endpoint(self, *, require_attitude: bool) -> LinkEndpointHistory:
        if require_attitude and self.attitude_quat_bn is None:
            raise ValueError(f"Directional terminal on {self.object_id!r} requires explicit attitude evidence.")
        return spacecraft_endpoint_history(
            asset_id=self.object_id,
            state_provider_id=self.state_provider_id,
            times_s=self.times_s,
            positions_eci_km=self.position_eci_km,
            velocities_eci_km_s=self.velocity_eci_km_s,
            attitudes_quat_bn=self.attitude_quat_bn,
            attitude_source_kind=self.attitude_source_kind if self.attitude_quat_bn is not None else "not_required",
            attitude_provider_id=self.attitude_provider_id if self.attitude_quat_bn is not None else None,
        )


def _validated_state(time_s: float, state: AnalysisState, *, require_attitude: bool) -> AnalysisState:
    if not isinstance(state, AnalysisState) or abs(float(state.time_s) - time_s) > 1.0e-9:
        raise ValueError("Provider must return AnalysisState for the requested epoch.")
    position = np.asarray(state.position_eci_km, dtype=float).reshape(-1)
    velocity = np.asarray(state.velocity_eci_km_s, dtype=float).reshape(-1)
    if position.size != 3 or velocity.size != 3 or np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity)):
        raise ValueError("Provider state must contain finite three-vector position and velocity.")
    attitude = state.attitude_quat_bn
    if attitude is not None:
        attitude = np.asarray(attitude, dtype=float).reshape(-1)
        if attitude.size != 4 or np.any(~np.isfinite(attitude)):
            raise ValueError("Provider attitude must contain four finite scalar-first values.")
        norm = float(np.linalg.norm(attitude))
        if abs(norm - 1.0) > 1.0e-10:
            raise ValueError("Provider attitude quaternion must be normalized within 1e-10.")
    if require_attitude and attitude is None:
        raise ValueError("Provider omitted required attitude evidence.")
    return AnalysisState(time_s, position, velocity, attitude)


def _slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    q0 = normalize_quaternion(left)
    q1 = normalize_quaternion(right)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return normalize_quaternion(q0 + float(fraction) * (q1 - q0))
    angle = float(np.arccos(dot))
    return normalize_quaternion(
        np.sin((1.0 - fraction) * angle) / np.sin(angle) * q0
        + np.sin(fraction * angle) / np.sin(angle) * q1
    )


def history_from_single_run(
    *, object_id: str, times_s: np.ndarray, truth_state: np.ndarray,
    initial_jd_utc: float, attitude_enabled: bool, state_provider_id: str,
    product_kind: str = "onp_completed_run",
) -> AnalysisHistory:
    truth = np.asarray(truth_state, dtype=float)
    if truth.ndim != 2 or truth.shape[1] < 6:
        raise ValueError("Single-run truth state must contain position and velocity columns.")
    attitudes = None
    attitude_kind = "not_required"
    attitude_provider = None
    if attitude_enabled:
        if truth.shape[1] < 10:
            raise ValueError("Attitude-enabled truth state is missing quaternion columns.")
        attitudes = truth[:, 6:10]
        attitude_kind = "achieved"
        attitude_provider = f"{state_provider_id}:achieved_attitude"
    return AnalysisHistory(
        object_id=object_id, product_kind=product_kind, state_provider_id=state_provider_id,
        frame="eci", initial_jd_utc=initial_jd_utc, times_s=times_s,
        position_eci_km=truth[:, :3], velocity_eci_km_s=truth[:, 3:6],
        attitude_quat_bn=attitudes, attitude_source_kind=attitude_kind,
        attitude_provider_id=attitude_provider,
    )


def history_from_review_store(path: str | Path, *, object_id: str) -> AnalysisHistory:
    database = Path(path).expanduser().resolve()
    if database.is_dir():
        database = database / "review" / "run.sqlite"
    if not database.is_file():
        raise FileNotFoundError(f"Completed-run review store not found: {database}")
    with sqlite3.connect(database) as conn:
        metadata = conn.execute("SELECT config_json FROM run_metadata LIMIT 1").fetchone()
        frame_row = conn.execute("SELECT state_frame FROM object_state_frame WHERE object_id = ?", (object_id,)).fetchone()
        rows = conn.execute(
            "SELECT time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, "
            "vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s, quat_w, quat_x, quat_y, quat_z "
            "FROM object_state WHERE object_id = ? ORDER BY sample_index", (object_id,),
        ).fetchall()
    if not rows:
        raise ValueError(f"Review store contains no object_state rows for {object_id!r}.")
    if frame_row is None or str(frame_row[0]).lower() != "eci":
        raise ValueError("Review-store adapter requires an explicit canonical ECI object_state frame.")
    config = json.loads(str(metadata[0])) if metadata and metadata[0] else {}
    simulator = dict(config.get("simulator", {}) or {})
    if simulator.get("initial_jd_utc") is None:
        raise ValueError("Review-store orbital analysis requires simulator.initial_jd_utc provenance.")
    attitude = dict(dict(simulator.get("dynamics", {}) or {}).get("attitude", {}) or {})
    object_config = dict(dict(config.get("objects", {}) or {}).get(object_id, {}) or {})
    attitude_enabled = bool(attitude.get("enabled", True)) and str(
        object_config.get("runtime_profile", "") or ""
    ).strip().lower() != "trajectory_only"
    return history_from_single_run(
        object_id=object_id, times_s=np.asarray([row[0] for row in rows]),
        truth_state=np.asarray([row[1:] for row in rows]),
        initial_jd_utc=float(simulator.get("initial_jd_utc")),
        attitude_enabled=attitude_enabled,
        state_provider_id=f"review:{database.name}:{object_id}", product_kind="review_store",
    )


def history_from_ogp_product(product: Any) -> AnalysisHistory:
    if str(getattr(product, "status", "completed")) != "completed":
        raise ValueError("OGP propagation product must be completed.")
    if str(getattr(product, "output_frame", "")).lower() != "eci":
        raise ValueError("OGP adapter requires an ECI output product; TEME must be transformed upstream.")
    samples = list(getattr(product, "samples", []) or [])
    if not samples:
        raise ValueError("OGP propagation product contains no samples.")
    def value(row: Any, key: str) -> Any:
        return row.get(key) if isinstance(row, dict) else getattr(row, key)
    errors = [str(value(row, "error") or "") for row in samples]
    if any(errors):
        raise ValueError("OGP propagation product contains failed samples.")
    return AnalysisHistory(
        object_id=str(product.object_id), product_kind="ogp_product",
        state_provider_id=str(product.propagation_product_id), frame="eci",
        initial_jd_utc=float(product.start_jd_utc),
        times_s=np.asarray([value(row, "t_s") for row in samples]),
        position_eci_km=np.asarray([[value(row, key) for key in ("pos_x_km", "pos_y_km", "pos_z_km")] for row in samples]),
        velocity_eci_km_s=np.asarray([[value(row, key) for key in ("vel_x_km_s", "vel_y_km_s", "vel_z_km_s")] for row in samples]),
    )


def history_from_ogp_store(
    path: str | Path,
    *,
    propagation_product_id: str,
) -> AnalysisHistory:
    """Load one completed ECI OGP product and its samples from an OEL Scale store."""

    from sim.scale.store_propagation import load_completed_propagation_products, load_product_samples

    matches = [
        item
        for item in load_completed_propagation_products(path)
        if item.propagation_product_id == propagation_product_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one completed OGP product {propagation_product_id!r}; found {len(matches)}.")
    summary = matches[0]
    product = SimpleNamespace(
        propagation_product_id=summary.propagation_product_id,
        object_id=summary.object_id,
        output_frame=summary.output_frame,
        start_jd_utc=summary.start_jd_utc,
        status="completed",
        samples=load_product_samples(path, propagation_product_id=propagation_product_id),
    )
    return history_from_ogp_product(product)


def directed_link_refinement_evaluator(
    config: DirectedLinkConfig,
    *,
    tx_history: AnalysisHistory,
    rx_history: AnalysisHistory,
    frame_context: FrameContext,
) -> Callable[[float], tuple[bool, str]]:
    """Bind arbitrary-epoch endpoint providers to the authoritative scalar link kernel."""

    tx_directional = not config.tx_terminal.pattern.attitude_independent
    rx_directional = not config.rx_terminal.pattern.attitude_independent

    def endpoint(history: AnalysisHistory, time_s: float, require_attitude: bool) -> LinkEndpointHistory:
        state = history.state_at(time_s)
        if require_attitude and state.attitude_quat_bn is None:
            raise ValueError(f"Directional terminal on {history.object_id!r} requires attitude at refinement epoch.")
        attitude = None if state.attitude_quat_bn is None else np.asarray([state.attitude_quat_bn])
        return spacecraft_endpoint_history(
            asset_id=history.object_id,
            state_provider_id=history.state_provider_id,
            times_s=np.asarray([time_s]),
            positions_eci_km=np.asarray([state.position_eci_km]),
            velocities_eci_km_s=np.asarray([state.velocity_eci_km_s]),
            attitudes_quat_bn=attitude,
            attitude_source_kind=history.attitude_source_kind if attitude is not None else "not_required",
            attitude_provider_id=history.attitude_provider_id if attitude is not None else None,
        )

    def evaluate(time_s: float) -> tuple[bool, str]:
        result = evaluate_directed_link_sample(
            config,
            tx_history=endpoint(tx_history, time_s, tx_directional),
            rx_history=endpoint(rx_history, time_s, rx_directional),
            frame_context=frame_context,
        )
        return bool(result.samples.available[0]), str(result.samples.primary_reason[0])

    return evaluate


def global_coverage_refinement_evaluator(
    history: AnalysisHistory,
    *,
    order: int,
    half_angle_rad: float,
    quat_body_from_sensor: tuple[float, float, float, float],
    max_range_km: float | None,
    frame_context: FrameContext,
) -> Callable[[float, np.ndarray], tuple[np.ndarray, tuple[str, ...]]]:
    """Bind arbitrary-epoch history evidence to the authoritative coverage geometry kernel."""

    if history.attitude_quat_bn is None:
        raise ValueError("Directional global coverage requires explicit attitude evidence.")
    sensor_to_body = quaternion_to_dcm_bn(np.asarray(quat_body_from_sensor, dtype=float))
    boresight_body = sensor_to_body @ np.array([0.0, 0.0, 1.0], dtype=float)

    def evaluate(time_s: float, cell_indices: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
        state = history.state_at(time_s)
        if state.attitude_quat_bn is None:
            raise ValueError("Coverage refinement provider omitted required attitude evidence.")
        rotation = eci_to_ecef_rotation_context(float(time_s), frame_context)
        position_ecef = rotation @ state.position_eci_km
        ellipsoid_level = (
            (position_ecef[0] / WGS84_A_KM) ** 2
            + (position_ecef[1] / WGS84_A_KM) ** 2
            + (position_ecef[2] / WGS84_B_KM) ** 2
        )
        if ellipsoid_level <= 1.0:
            raise ValueError("Coverage refinement source is on or inside WGS84.")
        body_from_eci = quaternion_to_dcm_bn(state.attitude_quat_bn)
        boresight_ecef = rotation @ (body_from_eci.T @ boresight_body)
        cells = healpix_wgs84_centers(order, np.asarray(cell_indices, dtype=np.int64))
        geometry = evaluate_surface_targets_ecef(
            observer_ecef_km=position_ecef,
            target_ecef_km=cells.ecef_km,
            target_outward_normal_ecef=cells.outward_normal_ecef,
            boresight_ecef=boresight_ecef,
            half_angle_rad=half_angle_rad,
            max_range_km=max_range_km,
            angular_tolerance_rad=1.0e-12,
            range_tolerance_km=1.0e-9,
        )
        available = np.asarray(geometry.available, dtype=bool)
        reasons = tuple("available" if value else "not_covered" for value in available)
        return available, reasons

    return evaluate


def evaluate_history_global_coverage(
    config: GlobalCoverageConfig,
    *,
    history: AnalysisHistory,
    frame_context: FrameContext,
) -> GlobalCoverageResult:
    """Evaluate one normalized ONP/OGP history through batch and scalar coverage paths."""

    if config.source_asset_id != history.object_id:
        raise ValueError("Coverage config source asset does not match the normalized history.")
    if config.state_provider_id != history.state_provider_id:
        raise ValueError("Coverage config state provider does not match the normalized history.")
    if history.attitude_quat_bn is None:
        raise ValueError("Directional global coverage requires explicit attitude evidence.")
    if config.attitude_source_kind != history.attitude_source_kind:
        raise ValueError("Coverage config attitude source kind does not match the normalized history.")
    if config.attitude_provider_id != history.attitude_provider_id:
        raise ValueError("Coverage config attitude provider does not match the normalized history.")
    evaluator = global_coverage_refinement_evaluator(
        history,
        order=config.order,
        half_angle_rad=config.half_angle_rad,
        quat_body_from_sensor=config.quat_body_from_sensor,
        max_range_km=config.max_range_km,
        frame_context=frame_context,
    )
    return evaluate_global_coverage(
        config,
        times_s=history.times_s,
        positions_eci_km=history.position_eci_km,
        attitudes_quat_bn=history.attitude_quat_bn,
        frame_context=frame_context,
        evaluator_at_time=evaluator if config.transition_time_tolerance_s is not None else None,
        refinement_provider_id=(
            f"{history.state_provider_id}:{history.refinement_source}"
            if config.transition_time_tolerance_s is not None
            else None
        ),
    )


def evaluate_history_directed_link(
    config: DirectedLinkConfig,
    *,
    tx_history: AnalysisHistory,
    rx_history: AnalysisHistory,
    frame_context: FrameContext,
) -> DirectedLinkResult:
    """Evaluate two normalized histories through batch and scalar directed-link paths."""

    tx_directional = not config.tx_terminal.pattern.attitude_independent
    rx_directional = not config.rx_terminal.pattern.attitude_independent
    evaluator = directed_link_refinement_evaluator(
        config,
        tx_history=tx_history,
        rx_history=rx_history,
        frame_context=frame_context,
    )
    return evaluate_directed_link(
        config,
        tx_history=tx_history.link_endpoint(require_attitude=tx_directional),
        rx_history=rx_history.link_endpoint(require_attitude=rx_directional),
        frame_context=frame_context,
        evaluator_at_time=evaluator if config.transition_time_tolerance_s is not None else None,
        refinement_provider_id=(
            f"tx={tx_history.state_provider_id}:{tx_history.refinement_source};"
            f"rx={rx_history.state_provider_id}:{rx_history.refinement_source}"
            if config.transition_time_tolerance_s is not None
            else None
        ),
    )


__all__ = [
    "AnalysisHistory", "AnalysisState", "history_from_ogp_product",
    "history_from_ogp_store",
    "history_from_review_store", "history_from_single_run",
    "directed_link_refinement_evaluator",
    "evaluate_history_directed_link",
    "evaluate_history_global_coverage",
    "global_coverage_refinement_evaluator",
]

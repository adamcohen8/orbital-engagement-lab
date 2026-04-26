from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.api import SimulationSnapshot
from sim.utils.frames import eci_relative_to_ric_rect

EARTH_MU_KM3_S2 = 398600.4418


@dataclass(frozen=True)
class RPOTrainingConfig:
    enabled: bool = False
    scenario_id: str = ""
    learning_goal: str = ""
    target_object_id: str = "target"
    chaser_object_id: str = "chaser"
    keepout_radius_km: float | None = None
    goal_radius_km: float | None = None
    goal_relative_ric_km: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    goal_nmt_radial_amplitude_km: float | None = None
    goal_nmt_cross_track_amplitude_km: float = 0.0
    goal_nmt_cross_track_phase_deg: float = 0.0
    goal_nmt_center_ric_km: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    goal_nmt_tolerance_km: float | None = None
    goal_nmt_element_tolerance_km: float | None = None
    goal_nmt_velocity_tolerance_km_s: float | None = None
    max_time_s: float | None = None
    max_goal_speed_km_s: float | None = None
    max_delta_v_m_s: float | None = None

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "RPOTrainingConfig":
        game_cfg = dict(metadata.get("game", {}) or {})
        raw = dict(game_cfg.get("training", {}) or {})
        if not raw:
            return cls(enabled=False)
        goal = np.array(raw.get("goal_relative_ric_km", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
        if goal.size != 3:
            raise ValueError("metadata.game.training.goal_relative_ric_km must be length 3.")
        nmt_center = np.array(raw.get("goal_nmt_center_ric_km", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
        if nmt_center.size != 3:
            raise ValueError("metadata.game.training.goal_nmt_center_ric_km must be length 3.")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            scenario_id=str(raw.get("scenario_id", "") or ""),
            learning_goal=str(raw.get("learning_goal", "") or ""),
            target_object_id=str(raw.get("target_object_id", game_cfg.get("target_object_id", "target")) or "target"),
            chaser_object_id=str(raw.get("chaser_object_id", game_cfg.get("chaser_object_id", "chaser")) or "chaser"),
            keepout_radius_km=_optional_float(raw.get("keepout_radius_km")),
            goal_radius_km=_optional_float(raw.get("goal_radius_km")),
            goal_relative_ric_km=goal.astype(float),
            goal_nmt_radial_amplitude_km=_optional_float(raw.get("goal_nmt_radial_amplitude_km")),
            goal_nmt_cross_track_amplitude_km=float(raw.get("goal_nmt_cross_track_amplitude_km", 0.0) or 0.0),
            goal_nmt_cross_track_phase_deg=float(raw.get("goal_nmt_cross_track_phase_deg", 0.0) or 0.0),
            goal_nmt_center_ric_km=nmt_center.astype(float),
            goal_nmt_tolerance_km=_optional_float(raw.get("goal_nmt_tolerance_km")),
            goal_nmt_element_tolerance_km=_optional_float(raw.get("goal_nmt_element_tolerance_km", raw.get("goal_nmt_tolerance_km"))),
            goal_nmt_velocity_tolerance_km_s=_optional_float(raw.get("goal_nmt_velocity_tolerance_km_s")),
            max_time_s=_optional_float(raw.get("max_time_s")),
            max_goal_speed_km_s=_optional_float(raw.get("max_goal_speed_km_s")),
            max_delta_v_m_s=_optional_float(raw.get("max_delta_v_m_s")),
        )


@dataclass(frozen=True)
class RPOTrainingScore:
    scenario_id: str
    learning_goal: str
    samples: int
    elapsed_s: float
    closest_approach_km: float
    final_range_km: float
    final_goal_error_km: float
    final_relative_speed_km_s: float
    time_inside_keepout_s: float
    approximate_delta_v_m_s: float
    achieved_time_s: float | None
    min_goal_error_km: float
    final_nmt_radial_amplitude_km: float
    final_nmt_cross_track_amplitude_km: float
    final_nmt_radial_amplitude_error_km: float
    final_nmt_cross_track_amplitude_error_km: float
    final_nmt_drift_velocity_error_km_s: float
    goal_met: bool
    level_passed: bool
    level_failed: bool
    pass_fail_reasons: tuple[str, ...]
    keepout_violation: bool
    hints: tuple[str, ...]


class RPOTrainingTracker:
    def __init__(self, config: RPOTrainingConfig):
        self.config = config
        self.t_s: list[float] = []
        self.rel_ric_hist: list[np.ndarray] = []
        self.thrust_hist: list[np.ndarray] = []
        self.mean_motion_hist: list[float] = []

    def clear(self) -> None:
        self.t_s.clear()
        self.rel_ric_hist.clear()
        self.thrust_hist.clear()
        self.mean_motion_hist.clear()

    def record(self, snapshot: SimulationSnapshot) -> None:
        if not self.config.enabled:
            return
        target = snapshot.truth.get(self.config.target_object_id)
        chaser = snapshot.truth.get(self.config.chaser_object_id)
        if target is None or chaser is None:
            return
        rel = relative_ric_state_from_arrays(target, chaser)
        self.t_s.append(float(snapshot.time_s))
        self.rel_ric_hist.append(rel)
        target_arr = np.array(target, dtype=float).reshape(-1)
        n = float("nan")
        if target_arr.size >= 3:
            r_norm = float(np.linalg.norm(target_arr[:3]))
            if np.isfinite(r_norm) and r_norm > 0.0:
                n = float(np.sqrt(EARTH_MU_KM3_S2 / (r_norm**3)))
        self.mean_motion_hist.append(n)
        thrust = snapshot.applied_thrust.get(self.config.chaser_object_id, np.zeros(3, dtype=float))
        self.thrust_hist.append(np.array(thrust, dtype=float).reshape(3))

    def current_hint(self) -> str:
        if not self.rel_ric_hist:
            return ""
        rel = self.rel_ric_hist[-1]
        r = rel[:3]
        v = rel[3:]
        rng = float(np.linalg.norm(r))
        speed = float(np.linalg.norm(v))
        closing = float(np.dot(r, v)) < 0.0
        keepout = self.config.keepout_radius_km
        if keepout is not None and rng < float(keepout):
            return "Inside keepout: arrest closing motion and translate away from the target."
        if closing and speed > 0.01:
            return "Closing quickly: reduce relative speed before correcting position."
        if abs(float(r[1])) > max(abs(float(r[0])), abs(float(r[2])), 0.1):
            return "In-track error dominates: small along-track burns can create delayed radial effects."
        if speed < 0.001 and rng > 1.0:
            return "Mostly coasting: watch the natural relative drift before burning again."
        return "Use small pulses, then coast and observe the RIC trajectory."

    def score(self) -> RPOTrainingScore:
        if not self.rel_ric_hist:
            return RPOTrainingScore(
                scenario_id=self.config.scenario_id,
                learning_goal=self.config.learning_goal,
                samples=0,
                elapsed_s=0.0,
                closest_approach_km=float("nan"),
                final_range_km=float("nan"),
                final_goal_error_km=float("nan"),
                final_relative_speed_km_s=float("nan"),
                time_inside_keepout_s=0.0,
                approximate_delta_v_m_s=0.0,
                achieved_time_s=None,
                min_goal_error_km=float("nan"),
                final_nmt_radial_amplitude_km=float("nan"),
                final_nmt_cross_track_amplitude_km=float("nan"),
                final_nmt_radial_amplitude_error_km=float("nan"),
                final_nmt_cross_track_amplitude_error_km=float("nan"),
                final_nmt_drift_velocity_error_km_s=float("nan"),
                goal_met=False,
                level_passed=False,
                level_failed=False,
                pass_fail_reasons=("No samples recorded.",),
                keepout_violation=False,
                hints=(),
            )
        rel = np.vstack(self.rel_ric_hist)
        t = np.array(self.t_s, dtype=float)
        thrust = np.vstack(self.thrust_hist) if self.thrust_hist else np.zeros((rel.shape[0], 3), dtype=float)
        ranges = np.linalg.norm(rel[:, :3], axis=1)
        speeds = np.linalg.norm(rel[:, 3:], axis=1)
        n_hist = np.array(self.mean_motion_hist, dtype=float).reshape(-1)
        if self.config.goal_nmt_radial_amplitude_km is not None:
            goal_err = nmt_position_error_km(
                rel[:, :3],
                radial_amplitude_km=float(self.config.goal_nmt_radial_amplitude_km),
                cross_track_amplitude_km=float(self.config.goal_nmt_cross_track_amplitude_km),
                cross_track_phase_deg=float(self.config.goal_nmt_cross_track_phase_deg),
                center_ric_km=self.config.goal_nmt_center_ric_km,
            )
        else:
            goal_err = np.linalg.norm(rel[:, :3] - self.config.goal_relative_ric_km.reshape(1, 3), axis=1)
        element_errors = None
        if self.config.goal_nmt_radial_amplitude_km is not None and n_hist.size:
            element_errors = nmt_element_errors(
                rel,
                mean_motion_rad_s=n_hist[: rel.shape[0]],
                radial_amplitude_km=float(self.config.goal_nmt_radial_amplitude_km),
                cross_track_amplitude_km=float(self.config.goal_nmt_cross_track_amplitude_km),
                center_ric_km=self.config.goal_nmt_center_ric_km,
            )
        keepout_time = 0.0
        keepout_violation = False
        if self.config.keepout_radius_km is not None:
            inside = ranges < float(self.config.keepout_radius_km)
            keepout_violation = bool(np.any(inside))
            keepout_time = _sampled_dwell_time_s(inside, t)
        dv_m_s = _integrated_delta_v_m_s(thrust, t)
        goal_met_samples = np.ones(rel.shape[0], dtype=bool)
        if self.config.goal_radius_km is not None:
            goal_met_samples &= goal_err <= float(self.config.goal_radius_km)
        if self.config.goal_nmt_tolerance_km is not None:
            goal_met_samples &= goal_err <= float(self.config.goal_nmt_tolerance_km)
        if element_errors is not None and self.config.goal_nmt_element_tolerance_km is not None:
            tol = float(self.config.goal_nmt_element_tolerance_km)
            goal_met_samples &= element_errors["radial_amplitude_error_km"] <= tol
            goal_met_samples &= element_errors["cross_track_amplitude_error_km"] <= tol
        if element_errors is not None and self.config.goal_nmt_velocity_tolerance_km_s is not None:
            goal_met_samples &= element_errors["drift_velocity_error_km_s"] <= float(self.config.goal_nmt_velocity_tolerance_km_s)
        if self.config.max_time_s is not None:
            goal_met_samples &= (t - t[0]) <= float(self.config.max_time_s)
        if self.config.max_goal_speed_km_s is not None:
            goal_met_samples &= speeds <= float(self.config.max_goal_speed_km_s)
        achieved_idx = np.flatnonzero(goal_met_samples)
        achieved_time_s = float(t[int(achieved_idx[0])] - t[0]) if achieved_idx.size else None
        budget_ok = True
        reasons: list[str] = []
        if achieved_time_s is None:
            reasons.append("NMT target not achieved within tolerance.")
        time_failed = self.config.max_time_s is not None and achieved_time_s is None and float(t[-1] - t[0]) >= float(self.config.max_time_s)
        if time_failed:
            reasons.append(f"Time budget exceeded ({float(self.config.max_time_s):.0f} s).")
        dv_failed = False
        if self.config.max_delta_v_m_s is not None and dv_m_s > float(self.config.max_delta_v_m_s):
            budget_ok = False
            dv_failed = True
            reasons.append(f"Delta-v budget exceeded ({float(self.config.max_delta_v_m_s):.1f} m/s).")
        if self.config.keepout_radius_km is not None:
            budget_ok = budget_ok and not keepout_violation
            if keepout_violation:
                reasons.append("Keepout was violated.")
        level_passed = bool(achieved_time_s is not None and budget_ok)
        level_failed = bool((keepout_violation or dv_failed or time_failed) and not level_passed)
        goal_met = level_passed
        if level_passed:
            reasons.append("All pass criteria satisfied.")
        final_elements = _final_nmt_element_values(element_errors)
        hints = tuple(h for h in (self.current_hint(),) if h)
        return RPOTrainingScore(
            scenario_id=self.config.scenario_id,
            learning_goal=self.config.learning_goal,
            samples=int(rel.shape[0]),
            elapsed_s=float(t[-1] - t[0]) if t.size >= 2 else 0.0,
            closest_approach_km=float(np.min(ranges)),
            final_range_km=float(ranges[-1]),
            final_goal_error_km=float(goal_err[-1]),
            final_relative_speed_km_s=float(speeds[-1]),
            time_inside_keepout_s=float(keepout_time),
            approximate_delta_v_m_s=float(dv_m_s),
            achieved_time_s=achieved_time_s,
            min_goal_error_km=float(np.min(goal_err)),
            final_nmt_radial_amplitude_km=final_elements["radial_amplitude_km"],
            final_nmt_cross_track_amplitude_km=final_elements["cross_track_amplitude_km"],
            final_nmt_radial_amplitude_error_km=final_elements["radial_amplitude_error_km"],
            final_nmt_cross_track_amplitude_error_km=final_elements["cross_track_amplitude_error_km"],
            final_nmt_drift_velocity_error_km_s=final_elements["drift_velocity_error_km_s"],
            goal_met=bool(goal_met),
            level_passed=bool(level_passed),
            level_failed=bool(level_failed),
            pass_fail_reasons=tuple(reasons),
            keepout_violation=bool(keepout_violation),
            hints=hints,
        )

    def debrief_text(self) -> str:
        score = self.score()
        lines = [
            "",
            "=" * 72,
            "RPO TRAINER DEBRIEF",
            "=" * 72,
        ]
        if score.scenario_id:
            lines.append(f"Scenario      : {score.scenario_id}")
        if score.learning_goal:
            lines.append(f"Learning Goal : {score.learning_goal}")
        lines.extend(
            [
                f"Samples       : {score.samples}",
                f"Elapsed       : {score.elapsed_s:.1f} s",
                f"Closest App   : {score.closest_approach_km:.3f} km",
                f"Final Range   : {score.final_range_km:.3f} km",
                f"Goal Error    : {score.final_goal_error_km:.3f} km",
                f"Best Goal Err : {score.min_goal_error_km:.3f} km",
                f"Final Speed   : {score.final_relative_speed_km_s:.5f} km/s",
                f"Keepout Time  : {score.time_inside_keepout_s:.1f} s",
                f"Approx dV     : {score.approximate_delta_v_m_s:.2f} m/s",
                f"NMT Rad Amp   : {score.final_nmt_radial_amplitude_km:.3f} km",
                f"NMT Cross Amp : {score.final_nmt_cross_track_amplitude_km:.3f} km",
                f"NMT Drift Err : {score.final_nmt_drift_velocity_error_km_s:.6f} km/s",
                f"Achieved Time : {_format_optional_time(score.achieved_time_s)}",
                f"Level Passed  : {'yes' if score.level_passed else 'no'}",
            ]
        )
        for reason in score.pass_fail_reasons:
            lines.append(f"Pass/Fail     : {reason}")
        for hint in score.hints:
            lines.append(f"Coach Note    : {hint}")
        lines.append("=" * 72)
        return "\n".join(lines)


def relative_ric_state_from_arrays(target_truth: np.ndarray, chaser_truth: np.ndarray) -> np.ndarray:
    target = np.array(target_truth, dtype=float).reshape(-1)
    chaser = np.array(chaser_truth, dtype=float).reshape(-1)
    if target.size < 6 or chaser.size < 6:
        return np.full(6, np.nan, dtype=float)
    return eci_relative_to_ric_rect(chaser[:6], target[:6])


def nmt_position_error_km(
    relative_ric_km: np.ndarray,
    *,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float = 0.0,
    cross_track_phase_deg: float = 0.0,
    center_ric_km: np.ndarray,
) -> np.ndarray:
    pos = np.array(relative_ric_km, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    if pos.shape[1] < 3:
        raise ValueError("relative_ric_km must contain R, I, and C components.")
    center = np.array(center_ric_km, dtype=float).reshape(3)
    curve = nmt_curve_points_km(
        radial_amplitude_km=radial_amplitude_km,
        cross_track_amplitude_km=cross_track_amplitude_km,
        cross_track_phase_deg=cross_track_phase_deg,
        center_ric_km=center,
    )
    if curve.size == 0:
        return np.linalg.norm(pos[:, :3] - center.reshape(1, 3), axis=1)
    delta = pos[:, None, :3] - curve[None, :, :]
    return np.min(np.linalg.norm(delta, axis=2), axis=1)


def nmt_curve_points_km(
    *,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float = 0.0,
    cross_track_phase_deg: float = 0.0,
    center_ric_km: np.ndarray,
    samples: int = 721,
) -> np.ndarray:
    a_r = float(radial_amplitude_km)
    if not np.isfinite(a_r) or a_r <= 0.0:
        return np.empty((0, 3), dtype=float)
    a_c = float(cross_track_amplitude_km)
    if not np.isfinite(a_c):
        a_c = 0.0
    phase = np.deg2rad(float(cross_track_phase_deg))
    center = np.array(center_ric_km, dtype=float).reshape(3)
    theta = np.linspace(0.0, 2.0 * np.pi, max(int(samples), 8), endpoint=True)
    pts = np.zeros((theta.size, 3), dtype=float)
    pts[:, 0] = center[0] + a_r * np.cos(theta)
    pts[:, 1] = center[1] - 2.0 * a_r * np.sin(theta)
    pts[:, 2] = center[2] + a_c * np.cos(theta + phase)
    return pts


def nmt_velocity_error_km_s(
    relative_ric_state: np.ndarray,
    *,
    mean_motion_rad_s: float,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float = 0.0,
    cross_track_phase_deg: float = 0.0,
    center_ric_km: np.ndarray,
) -> float:
    rel = np.array(relative_ric_state, dtype=float).reshape(-1)
    if rel.size < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    center = np.array(center_ric_km, dtype=float).reshape(3)
    n = float(mean_motion_rad_s)
    curve = nmt_curve_points_km(
        radial_amplitude_km=radial_amplitude_km,
        cross_track_amplitude_km=cross_track_amplitude_km,
        cross_track_phase_deg=cross_track_phase_deg,
        center_ric_km=center,
    )
    if curve.size == 0 or not np.isfinite(n):
        return float(np.linalg.norm(rel[3:6]))
    idx = int(np.argmin(np.linalg.norm(curve - rel[:3].reshape(1, 3), axis=1)))
    theta = 2.0 * np.pi * idx / max(curve.shape[0] - 1, 1)
    a_r = float(radial_amplitude_km)
    a_c = float(cross_track_amplitude_km)
    phase = np.deg2rad(float(cross_track_phase_deg))
    expected = np.array(
        [
            -a_r * n * np.sin(theta),
            -2.0 * a_r * n * np.cos(theta),
            -a_c * n * np.sin(theta + phase),
        ],
        dtype=float,
    )
    return float(np.linalg.norm(rel[3:6] - expected))


def nmt_element_errors(
    relative_ric_state: np.ndarray,
    *,
    mean_motion_rad_s: np.ndarray | float,
    radial_amplitude_km: float,
    cross_track_amplitude_km: float,
    center_ric_km: np.ndarray,
) -> dict[str, np.ndarray]:
    rel = np.array(relative_ric_state, dtype=float)
    if rel.ndim == 1:
        rel = rel.reshape(1, -1)
    if rel.shape[1] < 6:
        raise ValueError("relative_ric_state must contain RIC position and velocity.")
    n_raw = np.array(mean_motion_rad_s, dtype=float).reshape(-1)
    if n_raw.size == 1:
        n = np.full(rel.shape[0], float(n_raw[0]), dtype=float)
    else:
        n = n_raw[: rel.shape[0]]
        if n.size < rel.shape[0]:
            n = np.pad(n, (0, rel.shape[0] - n.size), constant_values=np.nan)
    center = np.array(center_ric_km, dtype=float).reshape(3)
    pos = rel[:, :3] - center.reshape(1, 3)
    vel = rel[:, 3:6]
    valid_n = np.isfinite(n) & (np.abs(n) > 1.0e-12)
    radial_amp = np.full(rel.shape[0], np.nan, dtype=float)
    cross_amp = np.full(rel.shape[0], np.nan, dtype=float)
    drift_vel_err = np.full(rel.shape[0], np.nan, dtype=float)
    radial_amp[valid_n] = np.sqrt(pos[valid_n, 0] ** 2 + (vel[valid_n, 0] / n[valid_n]) ** 2)
    cross_amp[valid_n] = np.sqrt(pos[valid_n, 2] ** 2 + (vel[valid_n, 2] / n[valid_n]) ** 2)
    drift_vel_err[valid_n] = np.abs(vel[valid_n, 1] + 2.0 * n[valid_n] * pos[valid_n, 0])
    return {
        "radial_amplitude_km": radial_amp,
        "cross_track_amplitude_km": cross_amp,
        "radial_amplitude_error_km": np.abs(radial_amp - float(radial_amplitude_km)),
        "cross_track_amplitude_error_km": np.abs(cross_amp - float(cross_track_amplitude_km)),
        "drift_velocity_error_km_s": drift_vel_err,
    }


def _final_nmt_element_values(element_errors: dict[str, np.ndarray] | None) -> dict[str, float]:
    keys = (
        "radial_amplitude_km",
        "cross_track_amplitude_km",
        "radial_amplitude_error_km",
        "cross_track_amplitude_error_km",
        "drift_velocity_error_km_s",
    )
    if element_errors is None:
        return {k: float("nan") for k in keys}
    return {k: float(np.array(element_errors[k], dtype=float).reshape(-1)[-1]) for k in keys}


def _format_optional_time(value: float | None) -> str:
    if value is None:
        return "not achieved"
    return f"{float(value):.1f} s"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _sampled_dwell_time_s(mask: np.ndarray, time_s: np.ndarray) -> float:
    inside = np.array(mask, dtype=bool).reshape(-1)
    t = np.array(time_s, dtype=float).reshape(-1)
    n = min(inside.size, t.size)
    if n < 2:
        return 0.0
    dt = np.diff(t[:n])
    valid = np.isfinite(dt) & (dt > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(dt[valid] * inside[: n - 1][valid]))


def _integrated_delta_v_m_s(thrust_km_s2: np.ndarray, time_s: np.ndarray) -> float:
    thrust = np.array(thrust_km_s2, dtype=float)
    t = np.array(time_s, dtype=float).reshape(-1)
    n = min(thrust.shape[0], t.size)
    if n < 2:
        return 0.0
    accel = np.linalg.norm(thrust[: n - 1, :], axis=1)
    dt = np.diff(t[:n])
    valid = np.isfinite(accel) & np.isfinite(dt) & (dt > 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.sum(accel[valid] * dt[valid]) * 1.0e3)

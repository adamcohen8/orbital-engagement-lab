"""Deterministic short-horizon pursuit and evasion policies for RPO.

The policies in this module use HCW as an onboard prediction model only.  The
authoritative trajectory continues to be propagated by the scenario's normal
orbit dynamics through the flight-software actuation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite

import numpy as np

from sim.dynamics.orbit.relative_linear import RelativeLinearDynamics

MAX_PREDICTION_STEPS = 10_000
MAX_ACCELERATION_FRACTIONS = 8
MAX_PREDICTIVE_PROPAGATIONS = 2_000_000


@dataclass(frozen=True, slots=True)
class PredictiveAction:
    acceleration_ric_m_s2: tuple[float, float, float]
    predicted_closest_range_m: float
    predicted_closest_time_s: float
    predicted_capture_time_s: float | None
    phase: str


def select_intercept_action(
    state_ric_si: np.ndarray,
    *,
    mean_motion_rad_s: float,
    max_acceleration_m_s2: float,
    horizon_s: float,
    step_s: float,
    pulse_duration_s: float,
    capture_radius_m: float,
    capture_margin_m: float = 0.0,
    acceleration_fractions: tuple[float, ...] = (0.5, 1.0),
    target_acceleration_ric_m_s2: np.ndarray | None = None,
) -> PredictiveAction:
    """Select a coast-aware pursuer action against one target action.

    Coast is selected whenever the passive trajectory reaches the capture
    radius.  Otherwise the lowest-delta-v candidate that predicts capture is
    used; if none captures, the candidate with the best closest approach is
    selected.
    """

    state = _validated_state(state_ric_si)
    _validate_policy_inputs(
        mean_motion_rad_s=mean_motion_rad_s,
        max_acceleration_m_s2=max_acceleration_m_s2,
        horizon_s=horizon_s,
        step_s=step_s,
        pulse_duration_s=pulse_duration_s,
        capture_radius_m=capture_radius_m,
        capture_margin_m=capture_margin_m,
        acceleration_fractions=acceleration_fractions,
    )
    target_action = (
        np.zeros(3)
        if target_acceleration_ric_m_s2 is None
        else np.asarray(target_acceleration_ric_m_s2, dtype=float).reshape(3)
    )
    if not np.all(np.isfinite(target_action)):
        raise ValueError("target_acceleration_ric_m_s2 must be finite")

    actions = _candidate_actions(state, max_acceleration_m_s2, acceleration_fractions)
    _require_work_budget(len(actions), horizon_s=horizon_s, step_s=step_s)
    predictions = [
        _predict(
            state,
            action - target_action,
            mean_motion_rad_s=mean_motion_rad_s,
            horizon_s=horizon_s,
            step_s=step_s,
            pulse_duration_s=pulse_duration_s,
            capture_radius_m=capture_radius_m,
        )
        for action in actions
    ]
    passive = predictions[0]
    passive_threshold = max(capture_radius_m - capture_margin_m, 0.0)
    if passive.closest_range_m <= passive_threshold:
        return _result(actions[0], passive, "passive_intercept_coast")

    capturing = [
        (index, prediction)
        for index, prediction in enumerate(predictions)
        if prediction.capture_time_s is not None
    ]
    if capturing:
        index, prediction = min(
            capturing,
            key=lambda item: (
                float(np.linalg.norm(actions[item[0]])) * pulse_duration_s,
                float(item[1].capture_time_s or 0.0),
                item[1].closest_range_m,
            ),
        )
        phase = "intercept_burn" if np.linalg.norm(actions[index]) > 0.0 else "intercept_coast"
        return _result(actions[index], prediction, phase)

    index, prediction = min(
        enumerate(predictions),
        key=lambda item: (
            item[1].closest_range_m,
            item[1].closest_time_s,
            float(np.linalg.norm(actions[item[0]])) * pulse_duration_s,
        ),
    )
    phase = "intercept_search_burn" if np.linalg.norm(actions[index]) > 0.0 else "intercept_search_coast"
    return _result(actions[index], prediction, phase)


def select_evasion_action(
    state_ric_si: np.ndarray,
    *,
    mean_motion_rad_s: float,
    max_acceleration_m_s2: float,
    opponent_max_acceleration_m_s2: float,
    horizon_s: float,
    step_s: float,
    pulse_duration_s: float,
    capture_radius_m: float,
    capture_margin_m: float = 0.0,
    acceleration_fractions: tuple[float, ...] = (0.5, 1.0),
) -> PredictiveAction:
    """Choose an evasion action against the pursuer's best bounded response.

    For every target candidate the pursuer is allowed to select its own best
    action.  The target maximizes the resulting worst-case closest approach,
    preferring survival, later capture, and lower delta-v in that order.
    """

    state = _validated_state(state_ric_si)
    _validate_policy_inputs(
        mean_motion_rad_s=mean_motion_rad_s,
        max_acceleration_m_s2=max_acceleration_m_s2,
        horizon_s=horizon_s,
        step_s=step_s,
        pulse_duration_s=pulse_duration_s,
        capture_radius_m=capture_radius_m,
        capture_margin_m=capture_margin_m,
        acceleration_fractions=acceleration_fractions,
    )
    if not isfinite(opponent_max_acceleration_m_s2) or opponent_max_acceleration_m_s2 < 0.0:
        raise ValueError("opponent_max_acceleration_m_s2 must be finite and nonnegative")

    target_actions = _candidate_actions(state, max_acceleration_m_s2, acceleration_fractions)
    pursuer_actions = _candidate_actions(-state, opponent_max_acceleration_m_s2, acceleration_fractions)
    _require_work_budget(
        len(target_actions) * len(pursuer_actions),
        horizon_s=horizon_s,
        step_s=step_s,
    )
    outcomes: list[tuple[int, _Prediction]] = []
    for target_index, target_action in enumerate(target_actions):
        responses = [
            _predict(
                state,
                target_action - pursuer_action,
                mean_motion_rad_s=mean_motion_rad_s,
                horizon_s=horizon_s,
                step_s=step_s,
                pulse_duration_s=pulse_duration_s,
                capture_radius_m=capture_radius_m,
            )
            for pursuer_action in pursuer_actions
        ]
        pursuer_best = min(responses, key=_pursuer_outcome_key)
        outcomes.append((target_index, pursuer_best))

    target_index, prediction = max(
        outcomes,
        key=lambda item: _evader_outcome_key(
            item[1],
            action=target_actions[item[0]],
            pulse_duration_s=pulse_duration_s,
        ),
    )
    action = target_actions[target_index]
    phase = "predictive_evasion_burn" if np.linalg.norm(action) > 0.0 else "predictive_evasion_coast"
    return _result(action, prediction, phase)


@dataclass(frozen=True, slots=True)
class _Prediction:
    closest_range_m: float
    closest_time_s: float
    capture_time_s: float | None


def _predict(
    state: np.ndarray,
    relative_acceleration: np.ndarray,
    *,
    mean_motion_rad_s: float,
    horizon_s: float,
    step_s: float,
    pulse_duration_s: float,
    capture_radius_m: float,
) -> _Prediction:
    current = np.asarray(state, dtype=float).copy()
    elapsed = 0.0
    closest_range = float(np.linalg.norm(current[:3]))
    closest_time = 0.0
    capture_time = 0.0 if closest_range <= capture_radius_m else None
    while elapsed < horizon_s - 1.0e-12:
        dt = min(step_s, horizon_s - elapsed)
        if elapsed < pulse_duration_s < elapsed + dt:
            dt = pulse_duration_s - elapsed
        ad, bd = _discrete_matrices(float(mean_motion_rad_s), float(dt))
        action = relative_acceleration if elapsed < pulse_duration_s - 1.0e-12 else np.zeros(3)
        current = ad @ current + bd @ action
        elapsed += dt
        range_m = float(np.linalg.norm(current[:3]))
        if range_m < closest_range:
            closest_range = range_m
            closest_time = elapsed
        if capture_time is None and range_m <= capture_radius_m:
            capture_time = elapsed
    return _Prediction(closest_range, closest_time, capture_time)


@lru_cache(maxsize=64)
def _discrete_matrices(mean_motion_rad_s: float, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    return RelativeLinearDynamics.hcw(mean_motion_rad_s).discrete_matrices(dt_s)


def _candidate_actions(
    state: np.ndarray,
    maximum: float,
    fractions: tuple[float, ...],
) -> tuple[np.ndarray, ...]:
    actions: list[np.ndarray] = [np.zeros(3)]
    if maximum <= 0.0:
        return tuple(actions)
    directions = [
        np.array((1.0, 0.0, 0.0)),
        np.array((-1.0, 0.0, 0.0)),
        np.array((0.0, 1.0, 0.0)),
        np.array((0.0, -1.0, 0.0)),
        np.array((0.0, 0.0, 1.0)),
        np.array((0.0, 0.0, -1.0)),
    ]
    for vector in (state[:3], state[3:], state[:3] + state[3:] * 120.0):
        norm = float(np.linalg.norm(vector))
        if norm > 1.0e-12:
            unit = vector / norm
            directions.extend((unit, -unit))
    unique: list[np.ndarray] = []
    for direction in directions:
        if not any(float(np.dot(direction, existing)) > 1.0 - 1.0e-10 for existing in unique):
            unique.append(direction)
    for fraction in fractions:
        actions.extend(maximum * float(fraction) * direction for direction in unique)
    return tuple(actions)


def _pursuer_outcome_key(prediction: _Prediction) -> tuple[float, float, float]:
    captures = prediction.capture_time_s is not None
    return (
        0.0 if captures else 1.0,
        float(prediction.capture_time_s) if captures else prediction.closest_range_m,
        prediction.closest_range_m,
    )


def _evader_outcome_key(
    prediction: _Prediction,
    *,
    action: np.ndarray,
    pulse_duration_s: float,
) -> tuple[float, float, float, float]:
    survives = prediction.capture_time_s is None
    return (
        1.0 if survives else 0.0,
        prediction.closest_range_m if survives else float(prediction.capture_time_s or 0.0),
        prediction.closest_time_s,
        -float(np.linalg.norm(action)) * pulse_duration_s,
    )


def _result(action: np.ndarray, prediction: _Prediction, phase: str) -> PredictiveAction:
    return PredictiveAction(
        tuple(float(value) for value in action),
        prediction.closest_range_m,
        prediction.closest_time_s,
        prediction.capture_time_s,
        phase,
    )


def _validated_state(value: np.ndarray) -> np.ndarray:
    state = np.asarray(value, dtype=float).reshape(6)
    if not np.all(np.isfinite(state)):
        raise ValueError("state_ric_si must contain six finite values")
    return state


def _validate_policy_inputs(
    *,
    mean_motion_rad_s: float,
    max_acceleration_m_s2: float,
    horizon_s: float,
    step_s: float,
    pulse_duration_s: float,
    capture_radius_m: float,
    capture_margin_m: float,
    acceleration_fractions: tuple[float, ...],
) -> None:
    for name, value, positive in (
        ("mean_motion_rad_s", mean_motion_rad_s, True),
        ("max_acceleration_m_s2", max_acceleration_m_s2, False),
        ("horizon_s", horizon_s, True),
        ("step_s", step_s, True),
        ("pulse_duration_s", pulse_duration_s, True),
        ("capture_radius_m", capture_radius_m, True),
        ("capture_margin_m", capture_margin_m, False),
    ):
        if not isfinite(value) or (value <= 0.0 if positive else value < 0.0):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must be finite and {qualifier}")
    if step_s > pulse_duration_s:
        raise ValueError("step_s must be no greater than pulse_duration_s")
    if pulse_duration_s > horizon_s:
        raise ValueError("pulse_duration_s must be no greater than horizon_s")
    if capture_margin_m >= capture_radius_m:
        raise ValueError("capture_margin_m must be smaller than capture_radius_m")
    if not acceleration_fractions or any(
        not isfinite(value) or value <= 0.0 or value > 1.0 for value in acceleration_fractions
    ):
        raise ValueError("acceleration_fractions must contain values in (0, 1]")
    if len(acceleration_fractions) > MAX_ACCELERATION_FRACTIONS:
        raise ValueError(
            f"acceleration_fractions may contain at most {MAX_ACCELERATION_FRACTIONS} values"
        )
    steps = int(np.ceil(horizon_s / step_s)) + 1
    if steps > MAX_PREDICTION_STEPS:
        raise ValueError(f"predictive horizon requires more than {MAX_PREDICTION_STEPS} propagation steps")


def _require_work_budget(candidate_count: int, *, horizon_s: float, step_s: float) -> None:
    steps = int(np.ceil(float(horizon_s) / float(step_s))) + 1
    requested = int(candidate_count) * steps
    if requested > MAX_PREDICTIVE_PROPAGATIONS:
        raise ValueError(
            "predictive candidate search exceeds the bounded propagation budget "
            f"({requested} > {MAX_PREDICTIVE_PROPAGATIONS})"
        )

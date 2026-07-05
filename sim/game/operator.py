from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.dynamics.orbit.cr3bp import cr3bp_moon_state_km_s
from sim.utils.frames import ric_dcm_ir_from_rv

MAX_OPERATOR_BURN_DELTA_V_M_S = 5.0
MIN_OPERATOR_BURN_SPACING_S = 10.0
_TIME_RE = re.compile(r"\bT\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:s|sec|secs|second|seconds)?\b", re.IGNORECASE)
_COMPONENT_RE = re.compile(
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:m\s*/\s*s|mps|m\/s)?\s*([RIC])\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OperatorBurn:
    time_s: float
    delta_v_ric_m_s: tuple[float, float, float]

    @property
    def magnitude_m_s(self) -> float:
        return float(np.linalg.norm(np.asarray(self.delta_v_ric_m_s, dtype=float)))


@dataclass(frozen=True)
class OperatorBurnPlan:
    burns: tuple[OperatorBurn, ...] = ()

    @property
    def total_delta_v_m_s(self) -> float:
        return float(sum(burn.magnitude_m_s for burn in self.burns))


class OperatorBurnCommandProvider:
    """External game command provider that executes pre-scripted RIC impulses."""

    def __init__(
        self,
        plan: OperatorBurnPlan,
        *,
        controlled_object_id: str,
        reference_object_id: str,
        control_mode: str = "ric_translation",
        relative_frame: str = "ric",
        actuator_error_fraction: float = 0.0,
    ) -> None:
        self.plan = plan
        self.controlled_object_id = str(controlled_object_id)
        self.reference_object_id = str(reference_object_id)
        self.control_mode = str(control_mode or "ric_translation").strip().lower()
        self.relative_frame = str(relative_frame or "ric").strip().lower()
        self.actuator_error_fraction = max(float(actuator_error_fraction), 0.0)
        self._next_burn_index = 0
        self.executed_delta_v_m_s = 0.0
        self.last_executed_burn: OperatorBurn | None = None
        self.last_executed_delta_v_ric_m_s: tuple[float, float, float] | None = None

    def next_burn_time_s(self) -> float | None:
        if self._next_burn_index >= len(self.plan.burns):
            return None
        return float(self.plan.burns[self._next_burn_index].time_s)

    def next_burn(self) -> OperatorBurn | None:
        if self._next_burn_index >= len(self.plan.burns):
            return None
        return self.plan.burns[self._next_burn_index]

    def __call__(
        self,
        *,
        truth: Any,
        own_knowledge: dict[str, Any] | None = None,
        t_s: float,
        dt_s: float,
        object_id: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if object_id is not None and str(object_id) != self.controlled_object_id:
            return {}
        self.last_executed_burn = None
        self.last_executed_delta_v_ric_m_s = None
        if self._next_burn_index >= len(self.plan.burns):
            return self._idle_intent()

        step_start_s = float(t_s)
        step_stop_s = step_start_s + max(float(dt_s), 0.0)
        due_burns: list[OperatorBurn] = []
        while self._next_burn_index < len(self.plan.burns):
            candidate = self.plan.burns[self._next_burn_index]
            if candidate.time_s > step_stop_s + 1.0e-9:
                break
            due_burns.append(candidate)
            self._next_burn_index += 1
        if not due_burns:
            return self._idle_intent()

        reference_state = _operator_reference_state_for_frame(
            _state_from_knowledge(dict(own_knowledge or {}), self.reference_object_id),
            control_mode=self.control_mode,
            relative_frame=self.relative_frame,
        )
        dcm_ir = ric_dcm_ir_from_rv(reference_state[:3], reference_state[3:6])
        planned_delta_v_ric_m_s = np.sum(
            [np.asarray(burn.delta_v_ric_m_s, dtype=float) for burn in due_burns],
            axis=0,
        )
        delta_v_ric_m_s = planned_delta_v_ric_m_s * (1.0 + self.actuator_error_fraction)
        delta_v_eci_km_s = dcm_ir @ (delta_v_ric_m_s / 1000.0)
        thrust_eci_km_s2 = delta_v_eci_km_s / max(float(dt_s), 1.0e-9)
        self.executed_delta_v_m_s += float(np.linalg.norm(delta_v_ric_m_s))
        self.last_executed_burn = due_burns[-1]
        self.last_executed_delta_v_ric_m_s = tuple(float(v) for v in delta_v_ric_m_s)
        return {
            "thrust_eci_km_s2": thrust_eci_km_s2,
            "mission_bypass_orbital_command_latch": True,
            "command_mode_flags": {
                "operator_mode": True,
                "operator_burn_index": self._next_burn_index,
                "operator_burn_count": len(due_burns),
                "operator_burn_time_s": float(due_burns[-1].time_s),
                "operator_burn_planned_delta_v_ric_m_s": tuple(float(v) for v in planned_delta_v_ric_m_s),
                "operator_burn_delta_v_ric_m_s": tuple(float(v) for v in delta_v_ric_m_s),
                "operator_actuator_error_fraction": float(self.actuator_error_fraction),
            },
        }

    def _idle_intent(self) -> dict[str, Any]:
        return {
            "thrust_eci_km_s2": np.zeros(3, dtype=float),
            "mission_bypass_orbital_command_latch": True,
            "command_mode_flags": {"operator_mode": True},
        }


def parse_operator_burn_plan(text: str) -> OperatorBurnPlan:
    burns: list[OperatorBurn] = []
    for raw_line in str(text or "").replace(";", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        time_match = _TIME_RE.search(line)
        if time_match is None:
            raise ValueError(f"Missing burn time in: {line}")
        time_s = float(time_match.group(1))
        components = {"R": 0.0, "I": 0.0, "C": 0.0}
        found_component = False
        for match in _COMPONENT_RE.finditer(line):
            component = match.group(2).upper()
            components[component] = float(match.group(1))
            found_component = True
        if not found_component:
            raise ValueError(f"Missing R/I/C delta-v components in: {line}")
        burns.append(
            OperatorBurn(
                time_s=time_s,
                delta_v_ric_m_s=(components["R"], components["I"], components["C"]),
            )
        )
    return OperatorBurnPlan(burns=tuple(sorted(burns, key=lambda burn: burn.time_s)))


def validate_operator_burn_plan(
    plan: OperatorBurnPlan,
    *,
    total_delta_v_budget_m_s: float | None = None,
    max_burn_delta_v_m_s: float = MAX_OPERATOR_BURN_DELTA_V_M_S,
    min_burn_spacing_s: float = MIN_OPERATOR_BURN_SPACING_S,
    max_time_s: float | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    for idx, burn in enumerate(plan.burns, start=1):
        if not np.isfinite(burn.time_s) or burn.time_s < 0.0:
            errors.append(f"Burn {idx}: time must be non-negative.")
        if max_time_s is not None and burn.time_s > float(max_time_s):
            errors.append(f"Burn {idx}: time exceeds level time budget.")
        if not all(np.isfinite(value) for value in burn.delta_v_ric_m_s):
            errors.append(f"Burn {idx}: delta-v must be finite.")
        if burn.magnitude_m_s > float(max_burn_delta_v_m_s) + 1.0e-9:
            errors.append(f"Burn {idx}: delta-v exceeds {max_burn_delta_v_m_s:.1f} m/s.")
    indexed_burns = sorted(enumerate(plan.burns, start=1), key=lambda item: item[1].time_s)
    min_spacing = max(float(min_burn_spacing_s), 0.0)
    if min_spacing > 0.0:
        for (prev_idx, previous), (idx, burn) in zip(indexed_burns, indexed_burns[1:], strict=False):
            if not np.isfinite(previous.time_s) or not np.isfinite(burn.time_s):
                continue
            spacing_s = float(burn.time_s) - float(previous.time_s)
            if spacing_s < min_spacing - 1.0e-9:
                errors.append(
                    f"Burn {idx}: time must be at least {min_spacing:.0f} seconds after Burn {prev_idx}."
                )
    if total_delta_v_budget_m_s is not None and plan.total_delta_v_m_s > float(total_delta_v_budget_m_s) + 1.0e-9:
        errors.append(f"Plan total delta-v exceeds {float(total_delta_v_budget_m_s):.1f} m/s budget.")
    return tuple(errors)


def operator_plan_summary(plan: OperatorBurnPlan) -> tuple[str, ...]:
    if not plan.burns:
        return ("No scripted burns.",)
    lines = [
        f"{len(plan.burns)} burn{'s' if len(plan.burns) != 1 else ''}, total dV {plan.total_delta_v_m_s:.2f} m/s"
    ]
    for idx, burn in enumerate(plan.burns[:4], start=1):
        r, i, c = burn.delta_v_ric_m_s
        lines.append(f"{idx}. T={burn.time_s:.1f}s  R={r:.2f}  I={i:.2f}  C={c:.2f} m/s")
    if len(plan.burns) > 4:
        lines.append(f"... {len(plan.burns) - 4} more")
    return tuple(lines)


def _state_from_knowledge(own_knowledge: dict[str, Any], object_id: str) -> np.ndarray:
    if object_id in own_knowledge:
        candidate = own_knowledge[object_id]
    else:
        candidate = own_knowledge.get(str(object_id), {})
    if isinstance(candidate, dict):
        if "estimated_state_eci_km_s" in candidate:
            state = candidate["estimated_state_eci_km_s"]
        elif "state_eci_km_s" in candidate:
            state = candidate["state_eci_km_s"]
        elif "state" in candidate:
            state = candidate["state"]
        else:
            raise KeyError(f"No ECI state available for reference object {object_id!r}.")
    else:
        state = getattr(candidate, "state", candidate)
    arr = np.asarray(state, dtype=float).reshape(-1)
    if arr.size < 6:
        raise ValueError(f"Reference object {object_id!r} state must contain at least 6 values.")
    return arr[:6]


def _operator_reference_state_for_frame(reference_state_eci_km_s: np.ndarray, *, control_mode: str, relative_frame: str) -> np.ndarray:
    state = np.asarray(reference_state_eci_km_s, dtype=float).reshape(-1)[:6]
    frame_key = str(relative_frame or "").strip().lower()
    mode_key = str(control_mode or "").strip().lower()
    if frame_key in {"moon_ric", "lunar_ric", "target_moon_ric", "target_lunar_ric"} or mode_key in {
        "moon_ric",
        "moon_ric_translation",
        "lunar_ric",
        "lunar_ric_translation",
    }:
        return state - cr3bp_moon_state_km_s()
    return state

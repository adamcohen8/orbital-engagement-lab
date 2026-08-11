from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

MAX_OPERATOR_BURN_DELTA_V_M_S = 5.0
MIN_OPERATOR_BURN_SPACING_S = 10.0
OPERATOR_IMPULSE_DURATION_S = 1.0e-3
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

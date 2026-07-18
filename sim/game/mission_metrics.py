# ruff: noqa: F401,F403,F405,I001
from .runner_common import *
from .runner_models import *
from .runner_config import *

def _game_debrief_enabled(
    config: SimulationConfig,
    training_cfg: RPOTrainingConfig,
    *,
    arcade_enabled: bool | None = None,
) -> bool:
    if bool(arcade_enabled) or (arcade_enabled is None and _game_arcade_enabled(config)):
        return False
    if bool(getattr(training_cfg, "sandbox_mode", False)):
        return False
    return bool(getattr(training_cfg, "enabled", False))


def _mission_metrics(config: RPOTrainingConfig, score: Any) -> tuple[str, ...]:
    metrics: list[str] = []
    if config.max_time_s is not None:
        elapsed = float(getattr(score, "elapsed_s", 0.0))
        if config.survival_goal:
            ratio = elapsed / max(float(config.max_time_s), 1.0e-9)
            metrics.append(
                f"{_status_tag(elapsed >= float(config.max_time_s), ratio >= 0.5)} Survive {elapsed:4.0f}/{float(config.max_time_s):.0f}s"
            )
        else:
            remain = max(float(config.max_time_s) - elapsed, 0.0)
            ratio = remain / max(float(config.max_time_s), 1.0e-9)
            metrics.append(f"{_status_tag(remain > 0.0, ratio > 0.2)} Time {remain:4.0f}s")
    if config.max_delta_v_m_s is not None:
        remain = max(float(config.max_delta_v_m_s) - float(getattr(score, "approximate_delta_v_m_s", 0.0)), 0.0)
        ratio = remain / max(float(config.max_delta_v_m_s), 1.0e-9)
        if config.fail_on_delta_v_budget:
            tag = _status_tag(remain > 0.0, ratio > 0.2)
        else:
            tag = "OK"
        suffix = " Coast" if not config.fail_on_delta_v_budget and remain <= 0.0 else ""
        metrics.append(f"{tag} Chaser dV {format_speed_m_s(remain)}{suffix}")
    elif config.sandbox_mode:
        used = float(getattr(score, "approximate_delta_v_m_s", 0.0))
        metrics.append(f"INFO dV Used {format_speed_m_s(used)}")
    if config.max_target_delta_v_m_s is not None:
        remain = max(float(config.max_target_delta_v_m_s) - float(getattr(score, "target_delta_v_m_s", 0.0)), 0.0)
        ratio = remain / max(float(config.max_target_delta_v_m_s), 1.0e-9)
        metrics.append(f"{_status_tag(remain > 0.0, ratio > 0.2)} Target dV {format_speed_m_s(remain)}")
    if config.max_target_reference_range_km is not None:
        limit = float(config.max_target_reference_range_km)
        current = float(getattr(score, "final_target_reference_range_km", float("nan")))
        margin = limit - current
        metrics.append(f"{_status_tag(margin >= 0.0, margin > 0.1)} Mission {_fmt_distance(margin)}")
    if config.sun_angle_constraints:
        angle = float(getattr(score, "final_sun_angle_deg", float("nan")))
        metrics.append(f"INFO Sun {angle:.0f} deg")
    if config.required_burn_axes:
        satisfied = set(getattr(score, "burn_axes_satisfied", ()))
        parts = [
            f"{_burn_axis_short_label(axis)}{'+' if axis in satisfied else '-'}" for axis in config.required_burn_axes
        ]
        all_done = len(satisfied.intersection(config.required_burn_axes)) >= len(config.required_burn_axes)
        metrics.append(f"{'OK' if all_done else 'WARN'} Burns {'/'.join(parts)}")
    if config.required_phase_burns:
        satisfied = set(getattr(score, "phase_burns_satisfied", ()))
        done = len(satisfied.intersection(burn.name for burn in config.required_phase_burns))
        total = len(config.required_phase_burns)
        metrics.append(f"{'OK' if done >= total else 'WARN'} Phase {done}/{total}")
    if config.require_speed_multiplier_change:
        changed = bool(getattr(score, "speed_multiplier_changed", False))
        metrics.append(f"{'OK' if changed else 'WARN'} Speed X")
    if config.required_coast_after_burn_s is not None:
        coasted = bool(getattr(score, "coast_after_burn_satisfied", False))
        metrics.append(f"{'OK' if coasted else 'WARN'} Coast {float(config.required_coast_after_burn_s):.0f}s")
    if config.guided_tutorial_burns:
        done = len(getattr(score, "guided_tutorial_burns_satisfied", ()))
        total = int(getattr(score, "guided_tutorial_burns_total", len(config.guided_tutorial_burns)))
        if config.guided_tutorial_speed_step is not None:
            done += 1 if bool(getattr(score, "guided_tutorial_speed_satisfied", False)) else 0
            total += 1
        metrics.append(f"{'OK' if done >= total else 'WARN'} Tutor {done}/{total}")
    if config.goal_nmt_element_tolerance_km is not None:
        tol = float(config.goal_nmt_element_tolerance_km)
        r_err = float(getattr(score, "final_nmt_radial_amplitude_error_km", float("nan")))
        c_err = float(getattr(score, "final_nmt_cross_track_amplitude_error_km", float("nan")))
        metrics.append(
            f"{_status_tag(r_err <= tol, r_err <= 0.75 * tol)} R Amp {_fmt_distance(r_err)}/{_fmt_distance(tol)}"
        )
        metrics.append(
            f"{_status_tag(c_err <= tol, c_err <= 0.75 * tol)} C Amp {_fmt_distance(c_err)}/{_fmt_distance(tol)}"
        )
    if config.goal_nmt_velocity_tolerance_km_s is not None:
        tol = float(config.goal_nmt_velocity_tolerance_km_s)
        err = float(getattr(score, "final_nmt_drift_velocity_error_km_s", float("nan")))
        metrics.append(f"{_status_tag(err <= tol, err <= 0.75 * tol)} Drift {_fmt_speed(err)}/{_fmt_speed(tol)}")
    if config.max_cross_track_amplitude_km is not None:
        tol = float(config.max_cross_track_amplitude_km)
        amp = float(getattr(score, "final_nmt_cross_track_amplitude_km", float("nan")))
        metrics.append(f"{_status_tag(amp <= tol, amp <= 0.75 * tol)} C Amp {_fmt_distance(amp)}/{_fmt_distance(tol)}")
    if config.goal_nmt_radial_amplitude_km is None and config.goal_range_km is not None:
        err = float(getattr(score, "final_goal_error_km", float("nan")))
        tol = config.goal_range_tolerance_km
        if tol is not None:
            tol_float = float(tol)
            metrics.append(
                f"{_status_tag(err <= tol_float, err <= 0.75 * tol_float)} Range {_fmt_distance(err)}/{_fmt_distance(tol_float)}"
            )
        else:
            final_range = float(getattr(score, "final_range_km", float("nan")))
            target_range = float(config.goal_range_km)
            inside_range = final_range <= target_range
            metrics.append(
                f"{_status_tag(inside_range, inside_range)} Range {_fmt_distance(final_range)}/{_fmt_distance(target_range)}"
            )
    elif config.goal_nmt_radial_amplitude_km is None and config.goal_radius_km is not None:
        err = float(getattr(score, "final_goal_error_km", float("nan")))
        tol = float(config.goal_radius_km)
        metrics.append(f"{_status_tag(err <= tol, err <= 0.75 * tol)} Goal {_fmt_distance(err)}/{_fmt_distance(tol)}")
    if config.goal_nmt_radial_amplitude_km is None and config.max_goal_speed_km_s is not None:
        speed = float(getattr(score, "final_relative_speed_km_s", float("nan")))
        tol = float(config.max_goal_speed_km_s)
        metrics.append(f"{_status_tag(speed <= tol, speed <= 0.75 * tol)} Speed {_fmt_speed(speed)}/{_fmt_speed(tol)}")
    if config.hard_speed_limit_radius_km is not None and config.hard_speed_limit_km_s is not None:
        violated = bool(getattr(score, "hard_speed_limit_violation", False))
        metrics.append(
            f"{_status_tag(not violated, not violated)} Prox V <= {_fmt_speed(float(config.hard_speed_limit_km_s))}"
        )
    if config.goal_nmt_radial_amplitude_km is None and config.keepout_radius_km is not None:
        final_range = float(getattr(score, "final_range_km", float("nan")))
        margin = final_range - float(config.keepout_radius_km)
        metrics.append(f"{_status_tag(margin >= 0.0, margin > 0.1)} KO {_fmt_distance(margin)}")
    if config.forbidden_regions:
        clear = not bool(getattr(score, "forbidden_region_violation", False))
        metrics.append(f"{_status_tag(clear, clear)} FR {'Clear' if clear else 'Violated'}")
    if config.inspection_gates:
        total = int(getattr(score, "inspection_gates_total", len(config.inspection_gates)))
        satisfied = int(getattr(score, "inspection_gates_satisfied", 0))
        tag = "OK" if satisfied >= total else "WARN"
        metrics.append(f"{tag} Inspect {satisfied}/{total}")
    if config.approach_gates:
        total = int(getattr(score, "approach_gates_total", len(config.approach_gates)))
        satisfied = int(getattr(score, "approach_gates_satisfied", 0))
        required = any(gate.required for gate in config.approach_gates)
        if not required and not bool(getattr(score, "approach_gate_violation", False)):
            return tuple(metrics)
        if bool(getattr(score, "approach_gate_violation", False)):
            tag = "FAIL"
        elif satisfied >= total:
            tag = "OK"
        else:
            tag = "WARN"
        metrics.append(f"{tag} Gates {satisfied}/{total}")
    return tuple(metrics)


def _mission_checklist(config: RPOTrainingConfig, score: Any) -> tuple[str, ...]:
    checklist: list[str] = []
    if config.sandbox_mode:
        return ("INFO Experiment Freely",)
    if config.required_burn_axes:
        satisfied = set(getattr(score, "burn_axes_satisfied", ()))
        for axis in config.required_burn_axes:
            checklist.append(f"{'OK' if axis in satisfied else 'WARN'} {_burn_axis_display_label(axis)} burn")
    if config.required_phase_burns:
        satisfied = set(getattr(score, "phase_burns_satisfied", ()))
        for burn in config.required_phase_burns:
            checklist.append(f"{'OK' if burn.name in satisfied else 'WARN'} {burn.label}")
    if config.require_speed_multiplier_change:
        changed = bool(getattr(score, "speed_multiplier_changed", False))
        checklist.append(f"{'OK' if changed else 'WARN'} Change speed")
    if config.required_coast_after_burn_s is not None:
        coasted = bool(getattr(score, "coast_after_burn_satisfied", False))
        checklist.append(f"{'OK' if coasted else 'WARN'} Coast after burn")
    if config.guided_tutorial_burns:
        satisfied = set(getattr(score, "guided_tutorial_burns_satisfied", ()))
        for stage in config.guided_tutorial_burns:
            checklist.append(f"{'OK' if stage.name in satisfied else 'WARN'} {stage.display_label}")
    if config.guided_tutorial_speed_step is not None:
        satisfied = bool(getattr(score, "guided_tutorial_speed_satisfied", False))
        checklist.append(f"{'OK' if satisfied else 'WARN'} {config.guided_tutorial_speed_step.label}")
    if config.inspection_gates:
        total = int(getattr(score, "inspection_gates_total", len(config.inspection_gates)))
        satisfied = int(getattr(score, "inspection_gates_satisfied", 0))
        checklist.append(f"{'OK' if satisfied >= total else 'WARN'} Inspect gates {satisfied}/{total}")
    if config.survival_goal and config.max_time_s is not None:
        elapsed = float(getattr(score, "elapsed_s", 0.0))
        checklist.append(f"{'OK' if elapsed >= float(config.max_time_s) else 'WARN'} Survive timer")
    elif config.goal_range_km is not None:
        final_range = float(getattr(score, "final_range_km", float("nan")))
        target_range = float(config.goal_range_km)
        checklist.append(f"{'OK' if final_range <= target_range else 'WARN'} Reach range")
    elif config.goal_radius_km is not None:
        err = float(getattr(score, "final_goal_error_km", float("nan")))
        tol = float(config.goal_radius_km)
        checklist.append(f"{'OK' if err <= tol else 'WARN'} Reach goal")
    elif config.goal_nmt_radial_amplitude_km is not None:
        passed = bool(getattr(score, "level_passed", False))
        checklist.append(f"{'OK' if passed else 'WARN'} Match NMT")
    if config.max_cross_track_amplitude_km is not None:
        amp = float(getattr(score, "final_nmt_cross_track_amplitude_km", float("nan")))
        checklist.append(f"{'OK' if amp <= float(config.max_cross_track_amplitude_km) else 'WARN'} Damp C Amp")
    if config.keepout_radius_km is not None:
        clear = not bool(getattr(score, "keepout_violation", False))
        checklist.append(f"{'OK' if clear else 'FAIL'} Keepout clear")
    if config.max_delta_v_m_s is not None and config.fail_on_delta_v_budget:
        used = float(getattr(score, "approximate_delta_v_m_s", 0.0))
        checklist.append(f"{'OK' if used <= float(config.max_delta_v_m_s) else 'FAIL'} Chaser dV")
    if config.max_target_delta_v_m_s is not None:
        used = float(getattr(score, "target_delta_v_m_s", 0.0))
        checklist.append(f"{'OK' if used <= float(config.max_target_delta_v_m_s) else 'FAIL'} Target dV")
    if config.max_target_reference_range_km is not None:
        clear = not bool(getattr(score, "target_reference_range_violation", False))
        checklist.append(f"{'OK' if clear else 'FAIL'} Mission capable")
    return tuple(checklist[:5])


def _status_tag(ok: bool, strong: bool) -> str:
    if not bool(ok):
        return "FAIL"
    if not bool(strong):
        return "WARN"
    return "OK"


def _burn_axis_short_label(axis: str) -> str:
    labels = {"radial": "R", "in_track": "I", "cross_track": "C"}
    return labels.get(str(axis), str(axis)[:1].upper())


def _burn_axis_display_label(axis: str) -> str:
    labels = {"radial": "Radial", "in_track": "In-track", "cross_track": "Cross-track"}
    return labels.get(str(axis), str(axis).replace("_", " ").title())


def _score_debrief_lines(
    score: Any,
    *,
    config: RPOTrainingConfig | None = None,
    difficulty: str = "easy",
) -> tuple[str, ...]:
    if not (bool(getattr(score, "level_passed", False)) or bool(getattr(score, "level_failed", False))):
        return ()
    arcade_score = _arcade_score(config, score, difficulty=difficulty) if config is not None else 0
    lines = [
        f"Scenario      {str(getattr(score, 'scenario_id', '') or '--')}",
        f"Score         {arcade_score:,}" if arcade_score > 0 else "",
        f"Elapsed       {float(getattr(score, 'elapsed_s', float('nan'))):.1f} s",
        f"Closest App   {_fmt_distance(float(getattr(score, 'closest_approach_km', float('nan'))))}",
        f"Final Range   {_fmt_distance(float(getattr(score, 'final_range_km', float('nan'))))}",
        f"Goal Error    {_fmt_distance(float(getattr(score, 'final_goal_error_km', float('nan'))))}",
        f"Final Speed   {_fmt_speed(float(getattr(score, 'final_relative_speed_km_s', float('nan'))))}",
        f"Keepout Time  {float(getattr(score, 'time_inside_keepout_s', 0.0)):.1f} s",
        f"Approx dV     {format_speed_m_s(float(getattr(score, 'approximate_delta_v_m_s', 0.0)))}",
        f"Target dV     {format_speed_m_s(float(getattr(score, 'target_delta_v_m_s', 0.0)))}",
    ]
    lines = [line for line in lines if line]
    for reason in tuple(getattr(score, "pass_fail_reasons", ()) or ())[:3]:
        lines.append(f"Result        {reason}")
    return tuple(lines)


def _fmt_distance(value_km: float) -> str:
    return format_distance_km(value_km)


def _fmt_speed(value_km_s: float) -> str:
    return format_speed_km_s(value_km_s)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)

__all__ = [name for name in globals() if not name.startswith("__")]

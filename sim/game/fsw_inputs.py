"""Game UI/script adapters that publish typed, timestamped FSW inputs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from sim.flight_software import (
    ClockScale,
    ClockTag,
    ControlAxisSample,
    GroundCommandKind,
    GroundCommandPayload,
    InputEvent,
    InputKind,
    PacketId,
    PilotInputPayload,
    Quality,
    TelemetryField,
)
from sim.flight_software.game_stacks import GamePilotInputProfile, GamePilotMode

from .operator import OPERATOR_IMPULSE_DURATION_S


@dataclass(frozen=True, slots=True)
class GameInputTimelineEntry:
    simulation_time_ns: int
    event: InputEvent


class GameSimulationClock:
    """Advance simulation/onboard time independently of render cadence."""

    def __init__(self, *, clock_id: str, tick_period_ns: int = 1_000_000, start_ticks: int = 0) -> None:
        if not clock_id.strip() or tick_period_ns <= 0 or start_ticks < 0:
            raise ValueError("game clock identity, period, and start must be valid")
        self.clock_id = clock_id
        self.tick_period_ns = int(tick_period_ns)
        self.ticks = int(start_ticks)
        self.paused = False
        self.speed_multiplier = 1.0
        self._fractional_ns = 0.0

    @property
    def tag(self) -> ClockTag:
        return ClockTag(self.clock_id, self.ticks, self.tick_period_ns, ClockScale.ONBOARD)

    @property
    def time_ns(self) -> int:
        return self.ticks * self.tick_period_ns

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def set_speed_multiplier(self, multiplier: float) -> None:
        value = float(multiplier)
        if not isfinite(value) or value <= 0.0:
            raise ValueError("speed multiplier must be finite and positive")
        self.speed_multiplier = value

    def advance_wall_time(self, wall_dt_s: float) -> int:
        elapsed = float(wall_dt_s)
        if not isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("wall_dt_s must be finite and nonnegative")
        if self.paused:
            return 0
        total_ns = self._fractional_ns + elapsed * self.speed_multiplier * 1.0e9
        ticks = int(total_ns // self.tick_period_ns)
        self._fractional_ns = total_ns - ticks * self.tick_period_ns
        self.ticks += ticks
        return ticks


class GamePilotInputAdapter:
    def __init__(
        self,
        profile: GamePilotInputProfile,
        *,
        source_id: str,
        boot_id: str,
        aerodynamic_axis_sources: tuple[tuple[str, str], ...] = (("deployment", "pitch"), ("bank", "roll")),
    ) -> None:
        if not source_id.strip() or not boot_id.strip():
            raise ValueError("source_id and boot_id must be non-empty")
        self.profile = profile
        self.source_id = source_id
        self.boot_id = boot_id
        self.aerodynamic_axis_sources = aerodynamic_axis_sources
        self._sequence = 0
        self._firing = False
        self._last_sample_signature: tuple[tuple[ControlAxisSample, ...], bool] | None = None
        self._last_live_control_signature: tuple[tuple[ControlAxisSample, ...], bool] | None = None
        self.timeline: list[GameInputTimelineEntry] = []
        self.control_mode = profile.mode.value
        self._physical_runtime: object | None = None
        self.ballistic_coefficient_min_kg_m2 = 40.0
        self.ballistic_coefficient_max_kg_m2 = 200.0
        self.aerodynamic_drag_coefficient = 2.2
        self.aerodynamic_lift_coefficient = 0.0
        self.aerodynamic_lift_area_m2 = 0.0

    def bind_physical_runtime(
        self,
        runtime: object,
        *,
        ballistic_coefficient_min_kg_m2: float = 40.0,
        ballistic_coefficient_max_kg_m2: float = 200.0,
        drag_coefficient: float = 2.2,
        lift_coefficient: float = 0.0,
        lift_area_m2: float = 0.0,
    ) -> None:
        self._physical_runtime = runtime
        self.ballistic_coefficient_min_kg_m2 = float(ballistic_coefficient_min_kg_m2)
        self.ballistic_coefficient_max_kg_m2 = float(ballistic_coefficient_max_kg_m2)
        self.aerodynamic_drag_coefficient = float(drag_coefficient)
        self.aerodynamic_lift_coefficient = float(lift_coefficient)
        self.aerodynamic_lift_area_m2 = float(lift_area_m2)

    @property
    def ballistic_coefficient_kg_m2(self) -> float:
        deployment = self._device_position("deployment", default=0.5)
        return self.ballistic_coefficient_min_kg_m2 + deployment * (
            self.ballistic_coefficient_max_kg_m2 - self.ballistic_coefficient_min_kg_m2
        )

    @property
    def lift_bank_angle_deg(self) -> float:
        return self._device_position("bank", default=0.0) * 180.0 / 3.141592653589793

    @property
    def drag_area_m2(self) -> float:
        mass = 0.0
        runtime = self._physical_runtime
        stack = getattr(runtime, "stack", None)
        config = getattr(stack, "config", None)
        if config is not None:
            mass = float(getattr(config, "assumed_mass_kg", 0.0))
        denominator = max(self.aerodynamic_drag_coefficient * self.ballistic_coefficient_kg_m2, 1.0e-12)
        return mass / denominator

    def _device_position(self, actuator_id: str, *, default: float) -> float:
        runtime = self._physical_runtime
        hardware = getattr(runtime, "hardware", {}).get(actuator_id) if runtime is not None else None
        return float(getattr(hardware, "position", default))

    def sample(self, command_state: object, *, at: ClockTag) -> InputEvent:
        axes = self._axes(command_state)
        firing = bool(getattr(command_state, "firing", False))
        return self._emit_sample(axes, firing=firing, at=at)

    def sample_if_changed(self, command_state: object, *, at: ClockTag) -> InputEvent | None:
        """Publish the initial control state and subsequent input transitions."""

        axes = self._axes(command_state)
        firing = bool(getattr(command_state, "firing", False))
        return self._emit_if_changed(axes, firing=firing, at=at)

    def live_control_state_changed(self, command_state: object) -> bool:
        """Observe raw UI control transitions without consuming timed input."""

        signature = (
            self._axes(command_state),
            bool(getattr(command_state, "firing", False)),
        )
        if signature == self._last_live_control_signature:
            return False
        self._last_live_control_signature = signature
        return True

    def sample_control_interval_if_changed(
        self,
        command_state: object,
        *,
        at: ClockTag,
        control_interval_s: float,
    ) -> InputEvent | None:
        """Publish one interval-averaged live control state when timing is enabled."""

        interval_s = float(control_interval_s)
        if not isfinite(interval_s) or interval_s <= 0.0:
            raise ValueError("control_interval_s must be finite and positive")
        if not bool(getattr(command_state, "use_timing_accumulator", False)):
            return self.sample_if_changed(command_state, at=at)

        if self.profile.mode is GamePilotMode.TRANSLATION:
            consume = getattr(command_state, "consume_ric_duty_cycle", None)
            if not callable(consume):
                raise TypeError("timed translation input requires consume_ric_duty_cycle")
            duty_cycle = tuple(
                0.0 if abs(float(value)) <= 1.0e-12 else float(value)
                for value in consume(interval_s)
            )
            if len(duty_cycle) != 3 or not all(isfinite(value) for value in duty_cycle):
                raise ValueError("timed translation duty cycle must contain three finite values")
            held_axes = tuple(
                float(getattr(command_state, name, 0.0))
                for name in ("pitch", "yaw", "roll")
            )
            effective_axes = tuple(
                held if abs(held) > 1.0e-12 else accumulated
                for held, accumulated in zip(held_axes, duty_cycle, strict=True)
            )
            axes = self._axes(command_state, translation_axes=effective_axes)
            firing = bool(getattr(command_state, "firing", False))
        elif self.profile.mode is GamePilotMode.ATTITUDE_THRUST:
            consume = getattr(command_state, "consume_firing_duty_cycle", None)
            if not callable(consume):
                raise TypeError("timed attitude-thrust input requires consume_firing_duty_cycle")
            firing_duty = float(consume(interval_s))
            if not isfinite(firing_duty):
                raise ValueError("timed firing duty cycle must be finite")
            firing_duty = 0.0 if firing_duty <= 1.0e-12 else max(0.0, min(1.0, firing_duty))
            if bool(getattr(command_state, "firing", False)):
                firing_duty = 1.0
            throttle = max(0.0, min(1.0, float(getattr(command_state, "throttle", 1.0))))
            axes = self._axes(command_state, throttle_fraction=throttle * firing_duty)
            firing = firing_duty > 0.0
        else:
            return self.sample_if_changed(command_state, at=at)

        return self._emit_if_changed(axes, firing=firing, at=at)

    def _emit_if_changed(
        self,
        axes: tuple[ControlAxisSample, ...],
        *,
        firing: bool,
        at: ClockTag,
    ) -> InputEvent | None:
        signature = (axes, firing)
        if signature == self._last_sample_signature:
            return None
        return self._emit_sample(axes, firing=firing, at=at)

    def _emit_sample(
        self,
        axes: tuple[ControlAxisSample, ...],
        *,
        firing: bool,
        at: ClockTag,
    ) -> InputEvent:
        pressed = (self.profile.firing_action,) if firing and not self._firing else ()
        released = (self.profile.firing_action,) if self._firing and not firing else ()
        self._firing = firing
        self._last_sample_signature = (axes, firing)
        event = InputEvent(
            PacketId(self.source_id, self.boot_id, self._sequence),
            InputKind.PILOT_INPUT,
            at,
            at,
            Quality(),
            PilotInputPayload(self.profile.profile_id, axes, pressed, released),
        )
        self._sequence += 1
        self.timeline.append(GameInputTimelineEntry(at.ticks * at.tick_period_ns, event))
        return event

    def _axes(
        self,
        state: object,
        *,
        translation_axes: tuple[float, float, float] | None = None,
        throttle_fraction: float | None = None,
    ) -> tuple[ControlAxisSample, ...]:
        throttle_value = (
            float(getattr(state, "throttle", 1.0))
            if throttle_fraction is None
            else float(throttle_fraction)
        )
        throttle = 2.0 * throttle_value - 1.0
        if self.profile.mode is GamePilotMode.TRANSLATION:
            values = (
                (
                    float(getattr(state, "pitch", 0.0)),
                    float(getattr(state, "yaw", 0.0)),
                    float(getattr(state, "roll", 0.0)),
                )
                if translation_axes is None
                else translation_axes
            )
            return tuple(
                ControlAxisSample(control_id, float(max(-1.0, min(1.0, value))))
                for control_id, value in (
                    (self.profile.radial_axis, values[0]),
                    (self.profile.in_track_axis, values[1]),
                    (self.profile.cross_track_axis, values[2]),
                    (self.profile.throttle_axis, throttle),
                )
            )
        if self.profile.mode is GamePilotMode.ATTITUDE_THRUST:
            pairs = (
                (self.profile.roll_axis, "roll"),
                (self.profile.pitch_axis, "pitch"),
                (self.profile.yaw_axis, "yaw"),
                (self.profile.throttle_axis, None),
            )
        else:
            pairs = self.aerodynamic_axis_sources
        return tuple(
            ControlAxisSample(control_id, float(max(-1.0, min(1.0, throttle if attribute is None else getattr(state, attribute, 0.0)))))
            for control_id, attribute in pairs
        )


class GameOperatorInputAdapter:
    def __init__(self, *, source_id: str, boot_id: str) -> None:
        if not source_id.strip() or not boot_id.strip():
            raise ValueError("source_id and boot_id must be non-empty")
        self.source_id = source_id
        self.boot_id = boot_id
        self._sequence = 0

    def scheduled_burn_events(
        self,
        burns: Iterable[object],
        *,
        clock_id: str,
        tick_period_ns: int,
        impulse_duration_s: float = OPERATOR_IMPULSE_DURATION_S,
        actuator_error_fraction: float = 0.0,
    ) -> tuple[InputEvent, ...]:
        if not isfinite(float(impulse_duration_s)) or impulse_duration_s <= 0.0:
            raise ValueError("operator impulse duration must be finite and positive")
        if not isfinite(float(actuator_error_fraction)) or actuator_error_fraction < 0.0:
            raise ValueError("operator actuator error fraction must be finite and nonnegative")
        duration_ticks = max(1, int(round(float(impulse_duration_s) * 1.0e9 / tick_period_ns)))
        realized_duration_s = duration_ticks * tick_period_ns * 1.0e-9
        events: list[InputEvent] = []
        for index, burn in enumerate(burns):
            time_s = float(burn.time_s)
            planned_delta_v = tuple(float(value) for value in burn.delta_v_ric_m_s)
            delta_v = tuple(value * (1.0 + float(actuator_error_fraction)) for value in planned_delta_v)
            if (
                len(delta_v) != 3
                or not all(isfinite(value) for value in planned_delta_v)
                or not isfinite(time_s)
                or time_s < 0.0
            ):
                raise ValueError("operator burns require a nonnegative time and three finite RIC components")
            ticks = int(round(time_s * 1.0e9 / tick_period_ns))
            tag = ClockTag(clock_id, ticks, tick_period_ns, ClockScale.ONBOARD)
            payload = GroundCommandPayload(
                f"operator-burn-{index}",
                GroundCommandKind.ACTION_REQUEST,
                (
                    TelemetryField("delta_v_r_m_s", delta_v[0], "m/s"),
                    TelemetryField("delta_v_i_m_s", delta_v[1], "m/s"),
                    TelemetryField("delta_v_c_m_s", delta_v[2], "m/s"),
                    TelemetryField("planned_delta_v_r_m_s", planned_delta_v[0], "m/s"),
                    TelemetryField("planned_delta_v_i_m_s", planned_delta_v[1], "m/s"),
                    TelemetryField("planned_delta_v_c_m_s", planned_delta_v[2], "m/s"),
                    TelemetryField("impulse_duration_s", realized_duration_s, "s"),
                    TelemetryField("actuator_error_fraction", float(actuator_error_fraction)),
                ),
                execute_at=tag,
            )
            events.append(
                InputEvent(
                    PacketId(self.source_id, self.boot_id, self._sequence),
                    InputKind.GROUND_COMMAND,
                    tag,
                    tag,
                    Quality(),
                    payload,
                )
            )
            self._sequence += 1
        return tuple(events)

    def goal_update(self, goal_id: str, parameters: tuple[TelemetryField, ...], *, at: ClockTag) -> InputEvent:
        payload = GroundCommandPayload(goal_id, GroundCommandKind.GOAL_UPDATE, parameters, execute_at=at)
        event = InputEvent(
            PacketId(self.source_id, self.boot_id, self._sequence),
            InputKind.GROUND_COMMAND,
            at,
            at,
            Quality(),
            payload,
        )
        self._sequence += 1
        return event


class GameOperatorController:
    """Presentation-side state for an operator plan delivered as typed events."""

    def __init__(
        self,
        plan: object,
        adapter: GameOperatorInputAdapter,
        *,
        impulse_duration_s: float = OPERATOR_IMPULSE_DURATION_S,
        actuator_error_fraction: float = 0.0,
    ) -> None:
        self.plan = plan
        self.adapter = adapter
        self.impulse_duration_s = max(float(impulse_duration_s), 1.0e-9)
        self.actuator_error_fraction = max(float(actuator_error_fraction), 0.0)
        self._next_burn_index = 0
        self.executed_delta_v_m_s = 0.0
        self.last_executed_burn: object | None = None
        self.last_executed_delta_v_ric_m_s: tuple[float, float, float] | None = None

    def observe_time(self, time_s: float) -> None:
        self.last_executed_burn = None
        self.last_executed_delta_v_ric_m_s = None
        burns = tuple(getattr(self.plan, "burns", ()) or ())
        while self._next_burn_index < len(burns):
            burn = burns[self._next_burn_index]
            if float(burn.time_s) > float(time_s) + 1.0e-9:
                break
            self.last_executed_burn = burn
            delta_v = tuple(
                float(value) * (1.0 + self.actuator_error_fraction)
                for value in burn.delta_v_ric_m_s
            )
            self.last_executed_delta_v_ric_m_s = delta_v
            self.executed_delta_v_m_s += sum(value * value for value in delta_v) ** 0.5
            self._next_burn_index += 1

    def next_burn_time_s(self) -> float | None:
        burns = tuple(getattr(self.plan, "burns", ()) or ())
        if self._next_burn_index >= len(burns):
            return None
        return float(burns[self._next_burn_index].time_s)

    def next_burn(self) -> object | None:
        burns = tuple(getattr(self.plan, "burns", ()) or ())
        return None if self._next_burn_index >= len(burns) else burns[self._next_burn_index]

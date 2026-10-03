"""Parallel Rust Trainer stacks with the existing typed lifecycle and policy."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import numpy as np

from sim.flight_software.contracts import (
    ActuatorCommand,
    AerodynamicEffectorPositionCommand,
    IdealWrenchCommand,
    ValidityInterval,
)
from sim.flight_software.game_stacks import GamePilotMode, GamePilotReferenceFlightSoftwareStack, _add_ticks, _throttle
from sim.flight_software.reference_stacks import RpoReferenceFlightSoftwareStack
from sim.flight_software.rust_game_backend import (
    RustOrbitNavigator,
    RustTranslationAllocator,
    extension,
    implementation_digest,
    native_torque_controller,
)
from sim.flight_software.rust_game_control import RustTranslationController


def _bind_native_identity(stack):
    # Retain the advertised contract/stack id, but never inherit Python's
    # implementation qualification or allow Python checkpoints to masquerade.
    digest = sha256(
        (stack.identity.implementation_hash + ":rust-game-v1:" + implementation_digest()).encode()
    ).hexdigest()
    stack._identity = replace(stack.identity, implementation_hash=digest)


class RustGamePilotFlightSoftwareStack(GamePilotReferenceFlightSoftwareStack):
    numeric_backend = "rust"

    def __init__(self, config, **kwargs):
        extension()
        super().__init__(
            replace(config, attitude_controller=native_torque_controller(config.attitude_controller)), **kwargs
        )
        self._navigator = RustOrbitNavigator.from_navigator(self._navigator)
        self._translation_allocator = RustTranslationAllocator(config.translation_allocator)
        _bind_native_identity(self)

    def _commit_restored_stack_state(self, state):
        super()._commit_restored_stack_state(state)
        self._navigator = RustOrbitNavigator.from_navigator(self._navigator)

    def _requested_force(self, solution):
        if not solution.own_state_valid:
            return None
        c, profile = self.config, self.config.profile
        mass = solution.mass_kg if solution.mass_kg is not None else c.assumed_mass_kg
        impulse = self._pending_delta_v_ric_m_s is not None
        if impulse:
            duration = self._pending_impulse_duration_s or c.operator_impulse_duration_s
            axes = tuple(x / duration for x in self._pending_delta_v_ric_m_s)
            self._pending_delta_v_ric_m_s = self._pending_impulse_duration_s = None
            ticks = max(1, int(round(duration / (solution.generated_at.tick_period_ns * 1e-9))))
            mode = 0
        else:
            axes = tuple(
                self._axes.get(axis, 0.0)
                for axis in (profile.radial_axis, profile.in_track_axis, profile.cross_track_axis)
            )
            ticks = c.validity_ticks
            mode = {GamePilotMode.TRANSLATION: 0, GamePilotMode.DIRECT_ECI: 1, GamePilotMode.ATTITUDE_THRUST: 2}[
                profile.mode
            ]
            if mode == 2 and (
                profile.firing_action not in self._held_actions or solution.attitude.attitude_quat_bn is None
            ):
                return None
        reference = np.asarray(
            self._reference_state_eci_m_m_s
            if self._reference_state_eci_m_m_s is not None
            else (*solution.position_eci_m, *solution.velocity_eci_m_s)
        )
        if c.translation_reference_origin_state_eci_m_m_s is not None:
            reference = reference - c.translation_reference_origin_state_eci_m_m_s
        q = solution.attitude.attitude_quat_bn or (1.0, 0.0, 0.0, 0.0)
        force = extension().pilot_force(
            mode,
            axes,
            _throttle(self._axes.get(profile.throttle_axis)),
            c.max_acceleration_m_s2,
            mass,
            tuple(reference),
            q,
            impulse,
        )
        return np.asarray(force), ticks

    def _advance_desired_attitude(self, body_rate, dt_s):
        return tuple(extension().advance_attitude(self._desired_attitude, tuple(body_rate), dt_s))

    def _translation_commands(self, effort, solution):
        if not self._live_command_fast_path:
            return self._translation_allocator.allocate(
                effort, solution, next_command_id=self._next_command_id
            ).proposed_commands
        if effort.force_n is None or solution.attitude.attitude_quat_bn is None:
            return ()
        allocator = self.config.translation_allocator
        body, _, _ = extension().ideal_wrench(effort.force_n, solution.attitude.attitude_quat_bn, allocator.max_force_n)
        return (
            ActuatorCommand(
                self._next_command_id(),
                allocator.satellite_id,
                allocator.actuator_id,
                effort.generated_at,
                effort.validity,
                allocator.actuator_frame,
                IdealWrenchCommand(tuple(body), (0.0, 0.0, 0.0)),
            ),
        )

    def _aerodynamic_commands(self, now):
        bindings = self.config.effectors
        positions = extension().effector_positions(
            [self._axes.get(b.control_id, 0.0) for b in bindings], [(b.minimum, b.maximum, b.neutral) for b in bindings]
        )
        validity = ValidityInterval(now, _add_ticks(now, self.config.validity_ticks))
        return tuple(
            ActuatorCommand(
                self._next_command_id(),
                self.config.satellite_id,
                b.actuator_id,
                now,
                validity,
                b.actuator_frame,
                AerodynamicEffectorPositionCommand(b.coordinate_id, value, b.unit),
            )
            for b, value in zip(bindings, positions, strict=True)
        )


class RustRpoFlightSoftwareStack(RpoReferenceFlightSoftwareStack):
    numeric_backend = "rust"

    def __init__(self, config, **kwargs):
        extension()
        super().__init__(
            replace(config, attitude_controller=native_torque_controller(config.attitude_controller)), **kwargs
        )
        self._allocator = RustTranslationAllocator(config.allocator)
        _bind_native_identity(self)

    def _new_navigator(self):
        return RustOrbitNavigator.from_navigator(super()._new_navigator())

    def _new_controller(self, config):
        return RustTranslationController(config)


def game_stack_type(stack_type, backend):
    if backend == "python":
        return stack_type
    if backend != "rust":
        raise ValueError("metadata.game.backend must be python or rust")
    if stack_type is GamePilotReferenceFlightSoftwareStack:
        return RustGamePilotFlightSoftwareStack
    if stack_type is RpoReferenceFlightSoftwareStack:
        return RustRpoFlightSoftwareStack
    raise ValueError(f"Rust game backend does not support {stack_type.stack_id}")

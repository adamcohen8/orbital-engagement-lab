"""Native control laws for the optional Trainer flight software backend."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from sim.control.orbit import predictive_engagement as prediction
from sim.control.orbit.lqr import HCWLQRController
from sim.control.orbit.ric_pd import RICPDTransferController, RICPDTransferGuidanceResult
from sim.flight_software.rust_game_backend import extension
from sim.gnc.orbit_v2 import TranslationControlLaw, TranslationController, TranslationMode, _clock_ns


@lru_cache(maxsize=32)
def _prediction_schedule(n, horizon, step, pulse):
    elapsed = 0.0
    matrices, times, active = [], [], []
    while elapsed < horizon - 1e-12:
        dt = min(step, horizon - elapsed)
        if elapsed < pulse < elapsed + dt:
            dt = pulse - elapsed
        ad, bd = prediction._discrete_matrices(n, dt)
        matrices.append((*ad.ravel(), *bd.ravel()))
        active.append(elapsed < pulse - 1e-12)
        elapsed += dt
        times.append(elapsed)
    return matrices, times, active


def native_predictive_action(state, *, evasion=False, opponent_max_acceleration_m_s2=0.0, **params):
    state = prediction._validated_state(state)
    prediction._validate_policy_inputs(**params)
    maximum = params["max_acceleration_m_s2"]
    fractions = params["acceleration_fractions"]
    actions = prediction._candidate_actions(state, maximum, fractions)
    schedule = _prediction_schedule(
        params["mean_motion_rad_s"], params["horizon_s"], params["step_s"], params["pulse_duration_s"]
    )
    if evasion:
        if not np.isfinite(opponent_max_acceleration_m_s2) or opponent_max_acceleration_m_s2 < 0.0:
            raise ValueError("opponent_max_acceleration_m_s2 must be finite and nonnegative")
        opponents = prediction._candidate_actions(-state, opponent_max_acceleration_m_s2, fractions)
        candidates = [tuple(a - b) for a in actions for b in opponents]
    else:
        opponents = ()
        candidates = [tuple(a) for a in actions]
    prediction._require_work_budget(len(candidates), horizon_s=params["horizon_s"], step_s=params["step_s"])
    outcomes = [
        prediction._Prediction(*p)
        for p in extension().predict_batch(
            tuple(state),
            candidates,
            *schedule,
            params["capture_radius_m"],
        )
    ]
    pulse = params["pulse_duration_s"]
    if evasion:
        worst = [
            min(outcomes[i : i + len(opponents)], key=prediction._pursuer_outcome_key)
            for i in range(0, len(outcomes), len(opponents))
        ]
        # max/min keep the first candidate on ties, matching the Python policy.
        index, outcome = max(
            enumerate(worst),
            key=lambda item: prediction._evader_outcome_key(item[1], action=actions[item[0]], pulse_duration_s=pulse),
        )
        phase = "predictive_evasion_burn" if np.linalg.norm(actions[index]) > 0.0 else "predictive_evasion_coast"
    elif outcomes[0].closest_range_m <= max(params["capture_radius_m"] - params["capture_margin_m"], 0.0):
        return prediction._result(actions[0], outcomes[0], "passive_intercept_coast")
    else:
        capturing = [(i, p) for i, p in enumerate(outcomes) if p.capture_time_s is not None]
        if capturing:
            index, outcome = min(
                capturing,
                key=lambda item: (
                    float(np.linalg.norm(actions[item[0]])) * pulse,
                    float(item[1].capture_time_s or 0.0),
                    item[1].closest_range_m,
                ),
            )
            phase = "intercept_burn" if np.linalg.norm(actions[index]) > 0.0 else "intercept_coast"
        else:
            index, outcome = min(
                enumerate(outcomes),
                key=lambda item: (
                    item[1].closest_range_m,
                    item[1].closest_time_s,
                    float(np.linalg.norm(actions[item[0]])) * pulse,
                ),
            )
            phase = "intercept_search_burn" if np.linalg.norm(actions[index]) > 0.0 else "intercept_search_coast"
    return prediction._result(actions[index], outcome, phase)


class RustRICPDTransferController(RICPDTransferController):
    """Shared transfer planning and snapshot state, native per-release feedback."""

    def guide_relative_state(self, relative_state_ric_rect_km, chief_position_eci_km, chief_velocity_eci_km_s, *, t_s):
        state = np.asarray(relative_state_ric_rect_km, dtype=float).reshape(6)
        chief = np.concatenate((chief_position_eci_km, chief_velocity_eci_km_s))
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(chief)) or not np.isfinite(t_s):
            raise ValueError("RIC PD transfer state and time must be finite")
        if np.linalg.norm(chief[:3]) <= 0.0:
            raise ValueError("RIC PD transfer chief position must be nonzero")
        if self._arrival_t_s is None:
            self._arrival_t_s = float(t_s) + self.transfer_time_s
        effective = self.state_signs * (state - self.desired_state_ric)
        signed = self._arrival_t_s - float(t_s)
        remaining = max(signed, 1.0)
        distance = float(np.linalg.norm(effective[:3]))
        brake = 0.0 < signed <= self.final_brake_start_s
        terminal = (
            signed <= 0.0 or remaining <= self.terminal_start_s or distance <= self.terminal_range_km
        ) and not brake
        target = effective[3:] if self._target_velocity_ric_km_s is None else self._target_velocity_ric_km_s
        failed = False
        if (
            not terminal
            and not brake
            and (self._target_velocity_ric_km_s is None or t_s >= self._next_correction_t_s - 1e-12)
        ):
            try:
                self._refresh_guidance(effective, float(t_s), remaining)
            except ValueError:
                failed = True
            else:
                target = self._target_velocity_ric_km_s
        if not hasattr(self, "_native_transfer_matrices"):
            self._native_transfer_matrices = (
                self._terminal_gain.ravel().tolist(),
                self._relative_dynamics.system_matrix()[3:, :].ravel().tolist(),
            )
        accel, pre, target, error, scale, phase_index = extension().transfer_control(
            tuple(effective),
            tuple(target),
            tuple(self._planned_arrival_velocity_ric_km_s),
            *self._native_transfer_matrices,
            signed,
            terminal,
            brake,
            failed,
            self.final_brake_start_s,
            self.burn_time_constant_s,
            self.max_accel_km_s2,
            self.velocity_deadband_m_s,
        )
        acceleration_eci = extension().ric_to_eci(accel, tuple(chief))
        phase = ("terminal_cleanup", "final_brake", "guided_burn", "coast")[phase_index]
        return RICPDTransferGuidanceResult(
            np.asarray(acceleration_eci),
            {
                "mode": "ric_pd_transfer",
                "dynamics_model": self.dynamics_model,
                "dynamics_metadata": self._relative_dynamics.metadata(),
                "phase": phase,
                "ric_curv_state_slice": list(self.ric_curv_state_slice),
                "chief_eci_state_slice": list(self.chief_eci_state_slice),
                "desired_state_ric": self.desired_state_ric.tolist(),
                "remaining_s": remaining,
                "signed_remaining_s": signed,
                "range_km": distance,
                "target_velocity_ric_km_s": list(target),
                "velocity_error_ric_km_s": list(error),
                "ideal_delta_v_ric_km_s": self._last_ideal_delta_v_ric_km_s.tolist(),
                "planned_arrival_velocity_ric_km_s": self._planned_arrival_velocity_ric_km_s.tolist(),
                "accel_ric_km_s2": list(accel),
                "limit_scale": scale,
                "linear_feedback_debug": HCWLQRController._linear_feedback_debug_payload(
                    control_axes=["R", "I", "C"],
                    k_gain=self._terminal_gain,
                    x_rect=state,
                    x_effective=effective,
                    control_pre_limit=np.asarray(pre),
                    control_post_limit=np.asarray(accel),
                    limit_scale=scale,
                    state_signs=self.state_signs,
                ),
            },
        )


class RustTranslationController(TranslationController):
    def __init__(self, config):
        if config.control_law is not TranslationControlLaw.REFERENCE_PD:
            raise ValueError("Rust Trainer RPO stacks currently support reference_pd control")
        super().__init__(config)

    def _relative_feedback_acceleration_ric(self, state, target, solution, track):
        c = self.config
        return np.asarray(
            extension().pd_feedback(
                tuple(state), tuple(target), c.kp_position_s2, c.kd_velocity_s_inv, c.mean_motion_rad_s
            )
        ), None

    def _ric_to_eci(self, vector_ric, solution, track=None):
        if not solution.own_state_valid:
            return None
        position = (
            track.chief_position_eci_m
            if track is not None and track.chief_position_eci_m is not None
            else solution.position_eci_m
        )
        velocity = (
            track.chief_velocity_eci_m_s
            if track is not None and track.chief_velocity_eci_m_s is not None
            else solution.velocity_eci_m_s
        )
        return np.asarray(extension().ric_to_eci(tuple(vector_ric), (*position, *velocity)))

    def _relative(self, mode, solution, track):
        if mode is not TranslationMode.PASSIVE_RETREAT or track is None:
            return super()._relative(mode, solution, track)
        c = self.config
        a, position_error, velocity_error, coast = extension().retreat(
            (*track.position_m, *track.velocity_m_s), c.kd_velocity_s_inv, c.retreat_speed_m_s, c.retreat_coast_range_m
        )
        return (
            self._ric_to_eci(a, solution, track),
            position_error,
            velocity_error,
            "passive_coast" if coast else "retreat_burn",
        )

    def _new_ric_pd_transfer(self, mean_motion_rad_s):
        original = super()._new_ric_pd_transfer(mean_motion_rad_s)
        native = RustRICPDTransferController.__new__(RustRICPDTransferController)
        native.__dict__.update(original.__dict__)
        return native

    def _predictive_engagement_guidance(self, mode, solution, track):
        state = np.asarray((*track.position_m, *track.velocity_m_s))
        distance, speed = float(np.linalg.norm(state[:3])), float(np.linalg.norm(state[3:]))
        if mode is TranslationMode.INTERCEPT_COAST and (
            self._predictive_capture_latched or distance <= self.config.capture_radius_m
        ):
            self._predictive_capture_latched = True
            self._predictive_action_ric_m_s2 = np.zeros(3)
            self._predictive_phase = "capture_complete"
            return self._ric_to_eci(np.zeros(3), solution, track), distance, speed, self._predictive_phase
        if mode is TranslationMode.PREDICTIVE_EVASION and distance <= self.config.capture_radius_m:
            self._predictive_action_ric_m_s2 = np.zeros(3)
            self._predictive_phase = "evasion_captured"
            return self._ric_to_eci(np.zeros(3), solution, track), distance, speed, self._predictive_phase
        now = _clock_ns(solution.generated_at)
        c = self.config
        if self._predictive_next_plan_ns is None or now >= self._predictive_next_plan_ns:
            decision = native_predictive_action(
                state,
                evasion=mode is TranslationMode.PREDICTIVE_EVASION,
                opponent_max_acceleration_m_s2=c.opponent_max_acceleration_m_s2,
                mean_motion_rad_s=self._relative_mean_motion(solution, track),
                max_acceleration_m_s2=c.max_acceleration_m_s2,
                horizon_s=c.prediction_horizon_s,
                step_s=c.prediction_step_s,
                pulse_duration_s=c.prediction_pulse_duration_s,
                capture_radius_m=c.capture_radius_m,
                capture_margin_m=c.capture_margin_m,
                acceleration_fractions=c.prediction_acceleration_fractions,
            )
            self._predictive_action_ric_m_s2 = np.asarray(decision.acceleration_ric_m_s2)
            self._predictive_phase = decision.phase
            self._predictive_action_until_ns = now + int(round(c.prediction_pulse_duration_s * 1e9))
            self._predictive_next_plan_ns = now + int(round(c.prediction_decision_interval_s * 1e9))
        if self._predictive_action_until_ns is not None and now < self._predictive_action_until_ns:
            action, phase = self._predictive_action_ric_m_s2, self._predictive_phase
        else:
            action = np.zeros(3)
            phase = (
                "intercept_replan_coast"
                if mode is TranslationMode.INTERCEPT_COAST
                else "predictive_evasion_replan_coast"
            )
        return self._ric_to_eci(action, solution, track), distance, speed, phase

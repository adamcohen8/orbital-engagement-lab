# ruff: noqa: F401,F403,F405,I001
from ..strategies.base import *

@dataclass
class ControllerPointingExecution:
    align_thruster_to_thrust: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    require_attitude_alignment: bool = True
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    use_strategy_fallback_thrust: bool = True
    detumble_enter_rate_rad_s: float | None = None
    detumble_exit_rate_rad_s: float | None = None
    detumble_mode_name: str = "detumble"
    nominal_mode_name: str = "nominal"
    _detumble_latched: bool = False

    def _maybe_update_mode(
        self, truth: StateTruth, att_belief: StateBelief | None, attitude_controller: Any | None
    ) -> None:
        if (
            self.detumble_enter_rate_rad_s is None
            or attitude_controller is None
            or not hasattr(attitude_controller, "set_mode")
        ):
            return
        if att_belief is not None and att_belief.state.size >= 13:
            w = np.array(att_belief.state[10:13], dtype=float)
        else:
            w = np.array(truth.angular_rate_body_rad_s, dtype=float)
        w_norm = float(np.linalg.norm(w))
        enter = float(max(self.detumble_enter_rate_rad_s, 0.0))
        exit_rate = float(
            max(self.detumble_exit_rate_rad_s if self.detumble_exit_rate_rad_s is not None else enter, 0.0)
        )
        if self._detumble_latched:
            if w_norm <= exit_rate:
                self._detumble_latched = False
        elif w_norm >= enter:
            self._detumble_latched = True
        mode = self.detumble_mode_name if self._detumble_latched else self.nominal_mode_name
        try:
            attitude_controller.set_mode(mode)
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Unable to set attitude controller mode '%s': %s", mode, exc)

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        t_s: float,
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        orb_belief: StateBelief | None = None,
        att_belief: StateBelief | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._maybe_update_mode(truth=truth, att_belief=att_belief, attitude_controller=attitude_controller)
        _apply_orbit_controller_intent(orbit_controller, intent)
        c_orb = (
            orbit_controller.act(orb_belief, t_s, 2.0)
            if (orbit_controller is not None and orb_belief is not None)
            else Command.zero()
        )
        thrust_cmd = np.array(c_orb.thrust_eci_km_s2, dtype=float).reshape(3)
        thrust_norm = float(np.sqrt(np.dot(thrust_cmd, thrust_cmd)))
        if self.use_strategy_fallback_thrust and thrust_norm <= 1e-15 and "fallback_thrust_eci_km_s2" in intent:
            thrust_cmd = np.array(intent.get("fallback_thrust_eci_km_s2"), dtype=float).reshape(3)
            thrust_norm = float(np.sqrt(np.dot(thrust_cmd, thrust_cmd)))

        q_des = None
        if "desired_attitude_quat_bn" in intent:
            q_des = np.array(intent.get("desired_attitude_quat_bn"), dtype=float).reshape(-1)
        elif bool(intent.get("align_to_thrust", self.align_thruster_to_thrust)) and thrust_norm > 1e-15:
            q_des = _desired_attitude_for_thrust(
                truth=truth,
                thrust_eci_km_s2=thrust_cmd,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
            )
        if (
            q_des is not None
            and q_des.size == 4
            and attitude_controller is not None
            and hasattr(attitude_controller, "set_target")
        ):
            try:
                attitude_controller.set_target(q_des)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set mission execution target quaternion: %s", exc)

        c_att = (
            attitude_controller.act(att_belief, t_s, 2.0)
            if (attitude_controller is not None and att_belief is not None)
            else Command.zero()
        )

        tol_rad = _resolve_angle_tolerance_rad(self.alignment_tolerance_rad, self.alignment_tolerance_deg)
        alignment_error_rad = float("nan")
        if thrust_norm > 1e-15:
            b_dir = _unit(np.array(self.thruster_direction_body, dtype=float))
            b_to_eci = quaternion_to_dcm_bn(np.array(truth.attitude_quat_bn, dtype=float)).T
            force_axis_eci = -_unit(b_to_eci @ b_dir)
            thrust_dir = _unit(thrust_cmd)
            alignment_error_rad = float(np.arccos(np.clip(np.dot(force_axis_eci, thrust_dir), -1.0, 1.0)))
            if self.require_attitude_alignment and alignment_error_rad > tol_rad:
                thrust_cmd = np.zeros(3, dtype=float)

        mode_flags = dict(c_orb.mode_flags or {})
        if np.isfinite(alignment_error_rad):
            mode_flags["alignment_error_rad"] = float(alignment_error_rad)
            mode_flags["attitude_alignment_satisfied"] = bool(alignment_error_rad <= tol_rad)
        mode_flags["execution"] = "controller_pointing"
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": thrust_cmd,
            "torque_body_nm": np.array(c_att.torque_body_nm, dtype=float).reshape(3),
            "command_mode_flags": mode_flags,
            "desired_attitude_quat_bn": (
                np.array(q_des, dtype=float).reshape(4)
                if q_des is not None and q_des.size == 4
                else intent.get("desired_attitude_quat_bn")
            ),
            "mission_mode": {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "controller_pointing",
            },
        }

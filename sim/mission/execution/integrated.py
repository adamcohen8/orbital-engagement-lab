# ruff: noqa: F401,F403,F405,I001
from ..strategies.base import *
from .burns import PredictiveBurnExecution

@dataclass
class DirectIntegratedExecution:
    align_thruster_to_thrust: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    use_strategy_fallback_thrust: bool = True
    use_orbit_controller: bool = False
    orbit_controller_budget_ms: float = 2.0
    attitude_controller_budget_ms: float = 2.0

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
        _apply_orbit_controller_intent(orbit_controller, intent)
        thrust_cmd = np.array(intent.get("command_thrust_eci_km_s2", np.zeros(3)), dtype=float).reshape(3)
        if (
            float(np.linalg.norm(thrust_cmd)) <= 1e-15
            and self.use_orbit_controller
            and orbit_controller is not None
            and orb_belief is not None
        ):
            c_orb = orbit_controller.act(orb_belief, float(t_s), float(max(self.orbit_controller_budget_ms, 1e-9)))
            thrust_cmd = np.array(c_orb.thrust_eci_km_s2, dtype=float).reshape(3)
        if (
            float(np.linalg.norm(thrust_cmd)) <= 1e-15
            and self.use_strategy_fallback_thrust
            and "fallback_thrust_eci_km_s2" in intent
        ):
            thrust_cmd = np.array(intent.get("fallback_thrust_eci_km_s2"), dtype=float).reshape(3)

        q_des = intent.get("desired_attitude_quat_bn")
        if (
            q_des is None
            and bool(intent.get("align_to_thrust", self.align_thruster_to_thrust))
            and float(np.linalg.norm(thrust_cmd)) > 1e-15
        ):
            q_des = _desired_attitude_for_thrust(
                truth=truth,
                thrust_eci_km_s2=thrust_cmd,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
            )
        q_des_arr = None if q_des is None else np.array(q_des, dtype=float).reshape(4)

        if q_des_arr is not None and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(q_des_arr)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set direct execution attitude target: %s", exc)
        c_att = (
            attitude_controller.act(att_belief, float(t_s), float(max(self.attitude_controller_budget_ms, 1e-9)))
            if (attitude_controller is not None and att_belief is not None)
            else Command.zero()
        )

        torque_cmd = np.array(intent.get("command_torque_body_nm", c_att.torque_body_nm), dtype=float).reshape(3)
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": thrust_cmd,
            "torque_body_nm": torque_cmd,
            "desired_attitude_quat_bn": q_des_arr,
            "command_mode_flags": {"execution": "direct_integrated"},
            "mission_mode": {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "direct_integrated",
            },
        }


@dataclass
class IntegratedCommandExecution:
    require_attitude_alignment: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    min_burn_accel_km_s2: float = 1e-12
    orbit_controller_budget_ms: float = 2.0
    attitude_controller_budget_ms: float = 2.0

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(
            self.alignment_tolerance_rad, self.alignment_tolerance_deg
        )

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        orb_belief: StateBelief | None = None,
        att_belief: StateBelief | None = None,
        t_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        _apply_orbit_controller_intent(orbit_controller, intent)

        c_orb = (
            orbit_controller.act(orb_belief, float(t_s), float(max(self.orbit_controller_budget_ms, 1e-9)))
            if (orbit_controller is not None and orb_belief is not None)
            else Command.zero()
        )
        thrust_cmd = np.array(c_orb.thrust_eci_km_s2, dtype=float).reshape(3)
        burn_requested = float(np.linalg.norm(thrust_cmd)) > float(max(self.min_burn_accel_km_s2, 0.0))

        align_ok = True
        align_angle = 0.0
        required_q = np.array(truth.attitude_quat_bn, dtype=float)
        if burn_requested and self.require_attitude_alignment and (not attitude_disabled):
            align_ok, align_angle = PredictiveBurnExecution._alignment(self, truth=truth, accel_eci_km_s2=thrust_cmd)
            required_q = _desired_attitude_for_thrust(
                truth=truth,
                thrust_eci_km_s2=thrust_cmd,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
            )

        if (not attitude_disabled) and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(np.array(required_q, dtype=float))
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set attitude target in IntegratedCommandExecution: %s", exc)
        c_att = (
            attitude_controller.act(att_belief, float(t_s), float(max(self.attitude_controller_budget_ms, 1e-9)))
            if (not attitude_disabled) and attitude_controller is not None and att_belief is not None
            else Command.zero()
        )

        if burn_requested and align_ok:
            out["thrust_eci_km_s2"] = thrust_cmd
            phase = "burn"
        else:
            out["thrust_eci_km_s2"] = np.zeros(3, dtype=float)
            phase = "slew" if burn_requested else "hold"
        out["torque_body_nm"] = np.array(c_att.torque_body_nm, dtype=float).reshape(3)
        out["desired_attitude_quat_bn"] = np.array(required_q, dtype=float)
        out["mission_use_integrated_command"] = True
        out["mission_mode"] = {
            **dict(intent.get("mission_mode", {}) or {}),
            "execution": "integrated_command",
            "phase": phase,
            "burn_requested": bool(burn_requested),
            "alignment_ok": bool(align_ok),
            "alignment_angle_rad": float(align_angle),
        }
        return out

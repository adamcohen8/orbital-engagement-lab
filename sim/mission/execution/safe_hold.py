# ruff: noqa: F401,F403,F405,I001
from ..strategies.base import *

@dataclass
class SafeHoldExecution:
    attitude_controller_budget_ms: float = 2.0

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        t_s: float,
        attitude_controller: Any | None = None,
        att_belief: StateBelief | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        q_des = intent.get("desired_attitude_quat_bn", np.array(truth.attitude_quat_bn, dtype=float))
        q_des_arr = np.array(q_des, dtype=float).reshape(4)
        if attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(q_des_arr)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set safe-hold attitude target: %s", exc)
        c_att = (
            attitude_controller.act(att_belief, float(t_s), float(max(self.attitude_controller_budget_ms, 1e-9)))
            if (attitude_controller is not None and att_belief is not None)
            else Command.zero()
        )
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": np.zeros(3, dtype=float),
            "torque_body_nm": np.array(c_att.torque_body_nm, dtype=float).reshape(3),
            "desired_attitude_quat_bn": q_des_arr,
            "command_mode_flags": {
                **dict(c_att.mode_flags or {}),
                "execution": "safe_hold",
            },
            "mission_mode": {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "safe_hold",
            },
        }

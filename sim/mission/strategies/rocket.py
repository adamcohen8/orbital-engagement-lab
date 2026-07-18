# ruff: noqa: F401,F403,F405,I001
from .base import *

@dataclass
class RocketPursuitMissionStrategy:
    target_id: str | None = None
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        **kwargs: Any,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "strategy_name": "rocket_pursuit",
            "orbital_goal": "pursuit",
            "mission_mode": {"strategy": "rocket_pursuit", "orbital_goal": "pursuit"},
            "align_to_thrust": bool(self.align_to_thrust),
        }
        if self.target_id:
            out["target_id"] = str(self.target_id)
        return out


@dataclass
class RocketPredefinedOrbitMissionStrategy:
    predef_target_alt_km: float = 400.0
    predef_target_ecc: float = 0.02
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "strategy_name": "rocket_predefined_orbit",
            "orbital_goal": "predefined_orbit",
            "predefined_orbit_goal": {
                "target_alt_km": float(self.predef_target_alt_km),
                "target_ecc": float(self.predef_target_ecc),
            },
            "mission_mode": {"strategy": "rocket_predefined_orbit", "orbital_goal": "predefined_orbit"},
            "align_to_thrust": bool(self.align_to_thrust),
        }


@dataclass
class RocketGoNowExecution:
    def update(
        self,
        *,
        intent: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        mission_mode = dict(intent.get("mission_mode", {}) or {})
        mission_mode["launch"] = "go_now"
        return {
            "launch_authorized": True,
            "mission_mode": mission_mode,
        }


@dataclass
class RocketGoWhenPossibleExecution:
    go_when_possible_margin_m_s: float = 0.0
    target_id: str | None = None

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief] | None = None,
        rocket_state: RocketState | None = None,
        rocket_vehicle_cfg: RocketVehicleConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        target_id = self.target_id or intent.get("target_id")
        target_state = _resolve_target_state(
            target_id=(None if target_id is None else str(target_id)),
            use_knowledge_for_targeting=True,
            own_knowledge=dict(own_knowledge or {}),
        )
        dv_needed = (
            np.inf
            if target_state is None
            else float(
                np.linalg.norm(np.array(target_state[1], dtype=float) - np.array(truth.velocity_eci_km_s, dtype=float))
                * 1e3
            )
        )
        dv_avail = (
            _compat_estimate_stack_delta_v_m_s(rocket_state, rocket_vehicle_cfg)
            if (rocket_state is not None and rocket_vehicle_cfg is not None)
            else np.inf
        )
        launch_authorized = bool(
            np.isfinite(dv_avail) and dv_avail >= (dv_needed + float(self.go_when_possible_margin_m_s))
        )
        mission_mode = dict(intent.get("mission_mode", {}) or {})
        mission_mode["launch"] = "go_when_possible"
        return {
            "launch_authorized": launch_authorized,
            "mission_mode": mission_mode,
        }


@dataclass
class RocketWaitOptimalExecution:
    window_period_s: float = 5400.0
    window_open_duration_s: float = 300.0

    def update(
        self,
        *,
        intent: dict[str, Any],
        t_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        period = max(float(self.window_period_s), 1.0)
        open_dt = float(np.clip(self.window_open_duration_s, 0.0, period))
        launch_authorized = (float(t_s) % period) <= open_dt
        mission_mode = dict(intent.get("mission_mode", {}) or {})
        mission_mode["launch"] = "wait_optimal_window"
        return {
            "launch_authorized": bool(launch_authorized),
            "mission_mode": mission_mode,
        }


@dataclass
class RocketMissionStrategy:
    launch_mode: str = "go_now"  # go_now|go_when_possible|wait_optimal_window
    orbital_goal: str = "pursuit"  # pursuit|predefined_orbit
    target_id: str | None = None
    go_when_possible_margin_m_s: float = 0.0
    window_period_s: float = 5400.0
    window_open_duration_s: float = 300.0
    predef_target_alt_km: float = 400.0
    predef_target_ecc: float = 0.02

    def update(
        self,
        *,
        truth: StateTruth,
        t_s: float,
        rocket_state: RocketState | None = None,
        rocket_vehicle_cfg: RocketVehicleConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Compatibility wrapper for older configs that combined goal selection and launch timing.
        if self.orbital_goal == "predefined_orbit":
            out = RocketPredefinedOrbitMissionStrategy(
                predef_target_alt_km=self.predef_target_alt_km,
                predef_target_ecc=self.predef_target_ecc,
            ).update(truth=truth)
        else:
            out = RocketPursuitMissionStrategy(target_id=self.target_id).update(truth=truth)
        if self.launch_mode == "wait_optimal_window":
            out.update(
                RocketWaitOptimalExecution(
                    window_period_s=self.window_period_s,
                    window_open_duration_s=self.window_open_duration_s,
                ).update(intent=out, t_s=t_s)
            )
        elif self.launch_mode == "go_when_possible":
            out.update(
                RocketGoWhenPossibleExecution(
                    go_when_possible_margin_m_s=self.go_when_possible_margin_m_s,
                    target_id=self.target_id,
                ).update(
                    intent=out,
                    truth=truth,
                    rocket_state=rocket_state,
                    rocket_vehicle_cfg=rocket_vehicle_cfg,
                )
            )
        else:
            out.update(RocketGoNowExecution().update(intent=out))
        return out

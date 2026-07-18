# ruff: noqa: F401,F403,F405,I001
from .base import *

@dataclass
class PursuitMissionStrategy:
    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    max_accel_km_s2: float = 0.0
    blind_direction_eci: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        **kwargs: Any,
    ) -> dict[str, Any]:
        tgt = _resolve_target_state(
            target_id=self.target_id,
            use_knowledge_for_targeting=self.use_knowledge_for_targeting,
            own_knowledge=own_knowledge,
        )
        if tgt is None:
            direction = _unit(np.array(self.blind_direction_eci, dtype=float))
        else:
            direction = _unit(tgt[0] - np.array(truth.position_eci_km, dtype=float))
        return {
            "strategy_name": "pursuit",
            "target_id": self.target_id,
            "fallback_thrust_eci_km_s2": float(max(self.max_accel_km_s2, 0.0)) * direction,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {"strategy": "pursuit"},
        }


@dataclass
class EvadeMissionStrategy:
    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    max_accel_km_s2: float = 0.0
    blind_direction_eci: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        **kwargs: Any,
    ) -> dict[str, Any]:
        tgt = _resolve_target_state(
            target_id=self.target_id,
            use_knowledge_for_targeting=self.use_knowledge_for_targeting,
            own_knowledge=own_knowledge,
        )
        if tgt is None:
            direction = _unit(np.array(self.blind_direction_eci, dtype=float))
        else:
            direction = -_unit(tgt[0] - np.array(truth.position_eci_km, dtype=float))
        return {
            "strategy_name": "evade",
            "target_id": self.target_id,
            "fallback_thrust_eci_km_s2": float(max(self.max_accel_km_s2, 0.0)) * direction,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {"strategy": "evade"},
        }


@dataclass
class HoldMissionStrategy:
    attitude_mode: str = "hold_eci"  # hold_eci|hold_ric|sun_track|spotlight|sensing
    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    hold_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    hold_quat_br: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    spotlight_lat_deg: float = 0.0
    spotlight_lon_deg: float = 0.0
    spotlight_alt_km: float = 0.0
    spotlight_ric_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        env: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        mode = str(self.attitude_mode).lower()
        if mode == "hold_eci":
            q_cmd = normalize_quaternion(np.array(self.hold_quat_bn, dtype=float))
        elif mode == "hold_ric":
            c_ir = ric_dcm_ir_from_rv(truth.position_eci_km, truth.velocity_eci_km_s)
            c_br = quaternion_to_dcm_bn(np.array(self.hold_quat_br, dtype=float))
            q_cmd = dcm_to_quaternion_bn(c_br @ c_ir.T)
        elif mode == "sun_track":
            sun_dir = np.array(env.get("sun_dir_eci", np.array([1.0, 0.0, 0.0])), dtype=float)
            q_cmd = PoseCommandGenerator.sun_track(
                truth=truth,
                sun_dir_eci=sun_dir,
                panel_normal_body=np.array(self.boresight_body, dtype=float),
            )
        elif mode == "spotlight":
            q_cmd = PoseCommandGenerator.spotlight_latlon(
                truth=truth,
                latitude_deg=float(self.spotlight_lat_deg),
                longitude_deg=float(self.spotlight_lon_deg),
                altitude_km=float(self.spotlight_alt_km),
                boresight_body=np.array(self.boresight_body, dtype=float),
            )
        elif mode == "sensing":
            q_cmd = PoseCommandGenerator.spotlight_ric_direction(
                truth=truth,
                ric_direction=np.array(self.spotlight_ric_direction, dtype=float),
                boresight_body=np.array(self.boresight_body, dtype=float),
            )
        else:
            tgt = _resolve_target_state(
                target_id=self.target_id,
                use_knowledge_for_targeting=self.use_knowledge_for_targeting,
                own_knowledge=own_knowledge,
            )
            if tgt is None:
                q_cmd = normalize_quaternion(np.array(truth.attitude_quat_bn, dtype=float))
            else:
                q_cmd = PoseCommandGenerator.sun_track(
                    truth=truth,
                    sun_dir_eci=_unit(tgt[0] - np.array(truth.position_eci_km, dtype=float)),
                    panel_normal_body=np.array(self.boresight_body, dtype=float),
                )
        return {
            "strategy_name": "hold",
            "desired_attitude_quat_bn": np.array(q_cmd, dtype=float),
            "mission_mode": {"strategy": "hold", "attitude": self.attitude_mode},
        }


@dataclass
class StationKeepMissionStrategy:
    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    desired_relative_ric_rect: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    kp_pos: float = 1.0e-5
    kd_vel: float = 5.0e-4
    max_accel_km_s2: float = 5.0e-5
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        **kwargs: Any,
    ) -> dict[str, Any]:
        desired_rel = np.array(self.desired_relative_ric_rect, dtype=float).reshape(6)
        tgt = _resolve_target_state(
            target_id=self.target_id,
            use_knowledge_for_targeting=self.use_knowledge_for_targeting,
            own_knowledge=own_knowledge,
        )
        desired_state_eci = None
        fallback_accel = np.zeros(3, dtype=float)
        if tgt is not None:
            desired_state_eci = ric_rect_state_to_eci(desired_rel, tgt[0], tgt[1])
            fallback_accel = _relative_pd_accel_eci(
                truth=truth,
                target_state_eci=tgt,
                desired_relative_ric_rect=desired_rel,
                kp_pos=float(self.kp_pos),
                kd_vel=float(self.kd_vel),
                max_accel_km_s2=float(self.max_accel_km_s2),
            )
        return {
            "strategy_name": "stationkeep",
            "target_id": self.target_id,
            "use_knowledge_for_targeting": bool(self.use_knowledge_for_targeting),
            "desired_relative_ric_rect_6": desired_rel,
            "desired_state_eci_6": desired_state_eci,
            "fallback_thrust_eci_km_s2": fallback_accel,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {"strategy": "stationkeep"},
        }


@dataclass
class OrbitalElementsStationKeepMissionStrategy:
    target_coes: dict[str, Any] = field(default_factory=dict)
    kp_pos: float = 1.0e-6
    kd_vel: float = 2.0e-3
    max_accel_km_s2: float = 5.0e-6
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        **kwargs: Any,
    ) -> dict[str, Any]:
        current_state = np.hstack(
            (np.array(truth.position_eci_km, dtype=float), np.array(truth.velocity_eci_km_s, dtype=float))
        )
        desired_state = coes_target_state_at_current_true_anomaly(dict(self.target_coes or {}), current_state)
        fallback_accel = _absolute_pd_accel_eci(
            truth=truth,
            desired_state_eci_6=desired_state,
            kp_pos=float(self.kp_pos),
            kd_vel=float(self.kd_vel),
            max_accel_km_s2=float(self.max_accel_km_s2),
        )
        return {
            "strategy_name": "orbital_elements_stationkeep",
            "desired_state_eci_6": desired_state,
            "fallback_thrust_eci_km_s2": fallback_accel,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {
                "strategy": "orbital_elements_stationkeep",
                "phase_mode": "current_true_anomaly",
            },
        }


@dataclass
class OrbitalElementsTrackingMissionStrategy:
    target_coes: dict[str, Any] = field(default_factory=dict)
    controlled_elements: tuple[str, ...] | list[str] | str = ("a", "ecc", "inc", "raan", "argp")
    energy_gain_per_s: float = 1.0e-3
    eccentricity_gain_per_s: float = 5.0e-4
    plane_gain_per_s: float = 5.0e-4
    max_accel_km_s2: float = 5.0e-5
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        **kwargs: Any,
    ) -> dict[str, Any]:
        current_state = np.hstack(
            (np.array(truth.position_eci_km, dtype=float), np.array(truth.velocity_eci_km_s, dtype=float))
        )
        result = orbital_element_feedback_accel(
            current_state,
            dict(self.target_coes or {}),
            controlled_elements=self.controlled_elements,
            energy_gain_per_s=float(self.energy_gain_per_s),
            eccentricity_gain_per_s=float(self.eccentricity_gain_per_s),
            plane_gain_per_s=float(self.plane_gain_per_s),
            max_accel_km_s2=float(self.max_accel_km_s2),
        )
        coes = result.current_coes
        controlled_elements = (
            [self.controlled_elements] if isinstance(self.controlled_elements, str) else list(self.controlled_elements)
        )
        return {
            "strategy_name": "orbital_elements_tracking",
            "fallback_thrust_eci_km_s2": np.array(result.accel_eci_km_s2, dtype=float),
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {
                "strategy": "orbital_elements_tracking",
                "controlled_elements": controlled_elements,
                "current_a_km": float(coes.a_km),
                "current_ecc": float(coes.ecc),
                "current_inc_deg": float(coes.inc_deg),
                "current_raan_deg": float(coes.raan_deg),
                "current_argp_deg": float(coes.argp_deg),
                "energy_error_km2_s2": float(result.energy_error_km2_s2),
                "eccentricity_vector_error_norm": float(np.linalg.norm(result.eccentricity_vector_error)),
                "hhat_error_norm": float(np.linalg.norm(result.hhat_error)),
            },
        }


@dataclass
class InspectMissionStrategy:
    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    desired_relative_ric_rect: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    )
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    kp_pos: float = 1.0e-5
    kd_vel: float = 5.0e-4
    max_accel_km_s2: float = 5.0e-5
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        **kwargs: Any,
    ) -> dict[str, Any]:
        desired_rel = np.array(self.desired_relative_ric_rect, dtype=float).reshape(6)
        tgt = _resolve_target_state(
            target_id=self.target_id,
            use_knowledge_for_targeting=self.use_knowledge_for_targeting,
            own_knowledge=own_knowledge,
        )
        desired_state_eci = None
        desired_attitude = np.array(truth.attitude_quat_bn, dtype=float)
        fallback_accel = np.zeros(3, dtype=float)
        if tgt is not None:
            desired_state_eci = ric_rect_state_to_eci(desired_rel, tgt[0], tgt[1])
            fallback_accel = _relative_pd_accel_eci(
                truth=truth,
                target_state_eci=tgt,
                desired_relative_ric_rect=desired_rel,
                kp_pos=float(self.kp_pos),
                kd_vel=float(self.kd_vel),
                max_accel_km_s2=float(self.max_accel_km_s2),
            )
            los_eci = _unit(tgt[0] - np.array(truth.position_eci_km, dtype=float))
            if float(np.linalg.norm(los_eci)) > 0.0:
                desired_attitude = PoseCommandGenerator.sun_track(
                    truth=truth,
                    sun_dir_eci=los_eci,
                    panel_normal_body=np.array(self.boresight_body, dtype=float),
                )
        return {
            "strategy_name": "inspect",
            "target_id": self.target_id,
            "use_knowledge_for_targeting": bool(self.use_knowledge_for_targeting),
            "desired_relative_ric_rect_6": desired_rel,
            "desired_state_eci_6": desired_state_eci,
            "desired_attitude_quat_bn": np.array(desired_attitude, dtype=float),
            "fallback_thrust_eci_km_s2": fallback_accel,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {"strategy": "inspect"},
        }


@dataclass
class SafeHoldMissionStrategy:
    attitude_mode: str = "hold_current"  # hold_current|hold_eci|sun_track
    hold_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))

    def update(
        self,
        *,
        truth: StateTruth,
        env: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        mode = str(self.attitude_mode).lower()
        if mode == "hold_eci":
            q_cmd = normalize_quaternion(np.array(self.hold_quat_bn, dtype=float))
        elif mode == "sun_track":
            sun_dir = np.array(env.get("sun_dir_eci", np.array([1.0, 0.0, 0.0])), dtype=float)
            q_cmd = PoseCommandGenerator.sun_track(
                truth=truth,
                sun_dir_eci=sun_dir,
                panel_normal_body=np.array(self.boresight_body, dtype=float),
            )
        else:
            q_cmd = normalize_quaternion(np.array(truth.attitude_quat_bn, dtype=float))
        return {
            "strategy_name": "safe_hold",
            "desired_attitude_quat_bn": np.array(q_cmd, dtype=float),
            "fallback_thrust_eci_km_s2": np.zeros(3, dtype=float),
            "command_torque_body_nm": np.zeros(3, dtype=float),
            "mission_mode": {"strategy": "safe_hold", "attitude": mode},
        }


@dataclass
class DesiredStateMissionStrategy:
    target_id: str | None = None
    desired_state_source: str = "target"  # target|explicit
    use_knowledge_for_targeting: bool = True
    desired_position_eci_km: np.ndarray | None = None
    desired_velocity_eci_km_s: np.ndarray | None = None
    align_to_thrust: bool = True

    def update(
        self,
        *,
        own_knowledge: dict[str, StateBelief],
        **kwargs: Any,
    ) -> dict[str, Any]:
        desired = _resolve_desired_state_from_inputs(
            target_id=self.target_id,
            desired_state_source=self.desired_state_source,
            use_knowledge_for_targeting=self.use_knowledge_for_targeting,
            desired_position_eci_km=self.desired_position_eci_km,
            desired_velocity_eci_km_s=self.desired_velocity_eci_km_s,
            own_knowledge=own_knowledge,
        )
        if desired is None:
            return {
                "strategy_name": "desired_state",
                "mission_mode": {"strategy": "desired_state", "phase": "hold_no_target"},
            }
        x_des = np.hstack((desired[0], desired[1]))
        return {
            "strategy_name": "desired_state",
            "target_id": self.target_id,
            "desired_state_eci_6": x_des,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {"strategy": "desired_state", "source": str(self.desired_state_source)},
        }


@dataclass
class DefensiveMissionStrategy:
    chaser_id: str = "chaser"
    defense_mode: str = "fixed_ric_axis"  # fixed_ric_axis|away_from_chaser
    axis_mode: str = "+R"  # +R|-R|+I|-I|+C|-C
    burn_accel_km_s2: float = 2e-6
    require_finite_knowledge: bool = True
    align_to_thrust: bool = True

    def _resolve_chaser_state(
        self,
        *,
        own_knowledge: dict[str, StateBelief],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        kb = own_knowledge.get(self.chaser_id)
        if kb is not None and kb.state.size >= 6:
            x = np.array(kb.state[:6], dtype=float)
            if (not self.require_finite_knowledge) or bool(np.all(np.isfinite(x))):
                return np.array(x[:3], dtype=float), np.array(x[3:6], dtype=float)
        return None

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        **kwargs: Any,
    ) -> dict[str, Any]:
        chaser_state = self._resolve_chaser_state(own_knowledge=own_knowledge)
        thrust_cmd = np.zeros(3, dtype=float)
        direction_source = "none"
        if chaser_state is not None and float(max(self.burn_accel_km_s2, 0.0)) > 0.0:
            mode = str(self.defense_mode).strip().lower()
            if mode == "away_from_chaser":
                direction_eci = -_unit(chaser_state[0] - np.array(truth.position_eci_km, dtype=float))
                direction_source = "away_from_chaser"
            else:
                direction_eci = ric_dcm_ir_from_rv(
                    np.array(truth.position_eci_km, dtype=float),
                    np.array(truth.velocity_eci_km_s, dtype=float),
                ) @ _axis_unit_ric(self.axis_mode)
                direction_source = "fixed_ric_axis"
            thrust_cmd = float(max(self.burn_accel_km_s2, 0.0)) * _unit(direction_eci)
        return {
            "strategy_name": "defensive",
            "target_id": self.chaser_id,
            "fallback_thrust_eci_km_s2": thrust_cmd,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {
                "strategy": "defensive",
                "defense_mode": str(self.defense_mode),
                "axis_mode": str(self.axis_mode),
                "has_chaser_knowledge": bool(chaser_state is not None),
                "direction_source": direction_source,
                "triggered": bool(float(np.linalg.norm(thrust_cmd)) > 0.0),
            },
        }

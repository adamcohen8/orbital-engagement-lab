# ruff: noqa: F401,F403,F405,I001
from .strategies.base import *

@dataclass
class MissionExecutiveStrategy:
    initial_mode: str | None = None
    modes: list[dict[str, Any]] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    strict_validation: bool = True
    _modes: dict[str, _MissionExecutiveMode] = field(default_factory=dict, init=False, repr=False)
    _active_mode: str | None = field(default=None, init=False, repr=False)
    _last_transition: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _active_mode_enter_t_s: float | None = field(default=None, init=False, repr=False)
    _transition_armed: dict[int, bool] = field(default_factory=dict, init=False, repr=False)
    _ordered_transitions: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        built: dict[str, _MissionExecutiveMode] = {}
        for raw in list(self.modes or []):
            if not isinstance(raw, dict):
                if self.strict_validation:
                    raise ValueError("Every mission executive mode must be a mapping.")
                continue
            name = str(raw.get("name", "") or "").strip()
            if not name:
                if self.strict_validation:
                    raise ValueError("Every mission executive mode must have a non-empty name.")
                continue
            if name in built:
                raise ValueError(f"Duplicate mission executive mode name {name!r}.")
            built[name] = _MissionExecutiveMode(
                name=name,
                strategy=_pointer_dict_to_obj(dict(raw.get("mission_strategy", {}) or {})),
                execution=_pointer_dict_to_obj(dict(raw.get("mission_execution", {}) or {})),
            )
        self._modes = built
        if self.initial_mode is not None:
            initial = str(self.initial_mode).strip()
            if initial not in self._modes:
                if self.strict_validation:
                    raise ValueError(f"initial_mode {initial!r} is not present in modes.")
            else:
                self._active_mode = initial
        elif self._modes:
            self._active_mode = next(iter(self._modes.keys()))
        self._ordered_transitions = self._validate_and_order_transitions(list(self.transitions or []))
        self._transition_armed = {i: True for i, _ in enumerate(self._ordered_transitions)}

    def _validate_and_order_transitions(self, transitions: list[Any]) -> list[dict[str, Any]]:
        allowed = {
            "range_lt",
            "range_gt",
            "fuel_below_kg",
            "fuel_below_fraction",
            "time_gte",
            "mode_elapsed_gte",
        }
        normalized: list[tuple[int, int, dict[str, Any]]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for index, raw in enumerate(transitions):
            if not isinstance(raw, dict):
                if self.strict_validation:
                    raise ValueError("Every mission executive transition must be a mapping.")
                continue
            transition = dict(raw)
            trigger = str(transition.get("trigger", "") or "").strip().lower()
            if trigger not in allowed:
                if self.strict_validation:
                    raise ValueError(f"Unsupported mission executive trigger {trigger!r}; expected one of {sorted(allowed)}.")
                continue
            to_mode = str(transition.get("to_mode", "") or "").strip()
            if to_mode not in self._modes:
                if self.strict_validation:
                    raise ValueError(f"Transition to_mode {to_mode!r} is not present in modes.")
                continue
            raw_from = transition.get("from_mode", "*")
            from_modes = list(raw_from) if isinstance(raw_from, (list, tuple)) else [raw_from]
            for from_mode_raw in from_modes:
                from_mode = str(from_mode_raw or "*").strip()
                if from_mode not in {"", "*"} and from_mode not in self._modes:
                    if self.strict_validation:
                        raise ValueError(f"Transition from_mode {from_mode!r} is not present in modes.")
            threshold = transition.get(
                {
                    "range_lt": "threshold_km",
                    "range_gt": "threshold_km",
                    "fuel_below_kg": "threshold_kg",
                    "fuel_below_fraction": "threshold_fraction",
                    "time_gte": "threshold_s",
                    "mode_elapsed_gte": "threshold_s",
                }[trigger],
                transition.get("threshold"),
            )
            if threshold is None or not np.isfinite(float(threshold)):
                raise ValueError(f"Transition trigger {trigger!r} requires a finite threshold.")
            signature = (repr(raw_from), to_mode, trigger, str(transition.get("target_id", "") or ""))
            if signature in seen:
                raise ValueError(f"Duplicate mission executive transition {signature!r}.")
            seen.add(signature)
            priority = int(transition.get("priority", 0) or 0)
            transition["priority"] = priority
            normalized.append((-priority, index, transition))
        normalized.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in normalized]

    @staticmethod
    def _metric_suffix(trigger: str) -> str | None:
        t = str(trigger).strip().lower()
        if t.startswith("range_"):
            return "km"
        if t == "fuel_below_kg":
            return "kg"
        if t == "fuel_below_fraction":
            return "fraction"
        if t in {"time_gte", "mode_elapsed_gte"}:
            return "s"
        return None

    def _fuel_metrics(
        self,
        *,
        truth: StateTruth,
        dry_mass_kg: float | None,
        fuel_capacity_kg: float | None,
        rocket_state: RocketState | None,
        rocket_vehicle_cfg: RocketVehicleConfig | None,
    ) -> tuple[float | None, float | None]:
        if rocket_state is not None:
            fuel_kg = float(np.sum(np.clip(np.array(rocket_state.stage_prop_remaining_kg, dtype=float), 0.0, np.inf)))
            fuel0_kg = None
            if rocket_vehicle_cfg is not None:
                fuel0_kg = float(sum(float(s.propellant_mass_kg) for s in rocket_vehicle_cfg.stack.stages))
            fuel_frac = None
            if fuel0_kg is not None and fuel0_kg > 0.0:
                fuel_frac = float(np.clip(fuel_kg / fuel0_kg, 0.0, 1.0))
            return fuel_kg, fuel_frac
        if dry_mass_kg is None or not np.isfinite(float(dry_mass_kg)):
            return None, None
        fuel_kg = float(max(float(truth.mass_kg) - float(dry_mass_kg), 0.0))
        fuel_frac = None
        if fuel_capacity_kg is not None and np.isfinite(float(fuel_capacity_kg)) and float(fuel_capacity_kg) > 0.0:
            fuel_frac = float(np.clip(fuel_kg / float(fuel_capacity_kg), 0.0, 1.0))
        elif fuel_kg <= 0.0:
            fuel_frac = 0.0
        return fuel_kg, fuel_frac

    def _range_km(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        target_id: str | None,
        use_knowledge: bool,
    ) -> float | None:
        tgt = _resolve_target_state(
            target_id=target_id,
            use_knowledge_for_targeting=use_knowledge,
            own_knowledge=own_knowledge,
        )
        if tgt is None:
            return None
        return float(np.linalg.norm(np.array(tgt[0], dtype=float) - np.array(truth.position_eci_km, dtype=float)))

    @staticmethod
    def _transition_applies_to_mode(transition: dict[str, Any], active_mode: str) -> bool:
        raw = transition.get("from_mode", "*")
        if raw is None:
            return True
        if isinstance(raw, (list, tuple)):
            return active_mode in {str(x).strip() for x in raw}
        token = str(raw).strip()
        return token in {"", "*"} or token == active_mode

    def _evaluate_transition(
        self,
        *,
        transition: dict[str, Any],
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        dry_mass_kg: float | None,
        rocket_state: RocketState | None,
        rocket_vehicle_cfg: RocketVehicleConfig | None,
        fuel_capacity_kg: float | None,
        t_s: float | None,
    ) -> tuple[bool, str]:
        trigger = str(transition.get("trigger", "") or "").strip().lower()
        if trigger in {"range_lt", "range_gt"}:
            range_km = self._range_km(
                truth=truth,
                own_knowledge=own_knowledge,
                target_id=(None if transition.get("target_id") is None else str(transition.get("target_id"))),
                use_knowledge=bool(transition.get("use_knowledge_for_targeting", True)),
            )
            if range_km is None:
                return False, "range_unavailable"
            threshold_km = float(transition.get("threshold_km", transition.get("threshold", 0.0)) or 0.0)
            if trigger == "range_lt":
                return bool(range_km < threshold_km), f"range_km={range_km:.6f}<threshold_km={threshold_km:.6f}"
            return bool(range_km > threshold_km), f"range_km={range_km:.6f}>threshold_km={threshold_km:.6f}"
        fuel_kg, fuel_frac = self._fuel_metrics(
            truth=truth,
            dry_mass_kg=dry_mass_kg,
            fuel_capacity_kg=fuel_capacity_kg,
            rocket_state=rocket_state,
            rocket_vehicle_cfg=rocket_vehicle_cfg,
        )
        if trigger == "fuel_below_kg":
            if fuel_kg is None:
                return False, "fuel_kg_unavailable"
            threshold_kg = float(transition.get("threshold_kg", transition.get("threshold", 0.0)) or 0.0)
            return bool(fuel_kg < threshold_kg), f"fuel_kg={fuel_kg:.6f}<threshold_kg={threshold_kg:.6f}"
        if trigger == "fuel_below_fraction":
            if fuel_frac is None:
                return False, "fuel_fraction_unavailable"
            threshold = float(transition.get("threshold_fraction", transition.get("threshold", 0.0)) or 0.0)
            return bool(fuel_frac < threshold), f"fuel_fraction={fuel_frac:.6f}<threshold={threshold:.6f}"
        if trigger == "time_gte":
            if t_s is None:
                return False, "time_unavailable"
            threshold_s = float(transition.get("threshold_s", transition.get("threshold")))
            return bool(t_s >= threshold_s), f"time_s={t_s:.6f}>=threshold_s={threshold_s:.6f}"
        if trigger == "mode_elapsed_gte":
            if t_s is None or self._active_mode_enter_t_s is None:
                return False, "mode_elapsed_unavailable"
            elapsed_s = float(t_s) - float(self._active_mode_enter_t_s)
            threshold_s = float(transition.get("threshold_s", transition.get("threshold")))
            return bool(elapsed_s >= threshold_s), f"mode_elapsed_s={elapsed_s:.6f}>=threshold_s={threshold_s:.6f}"
        return False, f"unsupported_trigger={trigger}"

    def _metric_value_for_transition(
        self,
        *,
        transition: dict[str, Any],
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        dry_mass_kg: float | None,
        rocket_state: RocketState | None,
        rocket_vehicle_cfg: RocketVehicleConfig | None,
        fuel_capacity_kg: float | None,
        t_s: float | None,
    ) -> float | None:
        trigger = str(transition.get("trigger", "") or "").strip().lower()
        if trigger in {"range_lt", "range_gt"}:
            return self._range_km(
                truth=truth,
                own_knowledge=own_knowledge,
                target_id=(None if transition.get("target_id") is None else str(transition.get("target_id"))),
                use_knowledge=bool(transition.get("use_knowledge_for_targeting", True)),
            )
        fuel_kg, fuel_frac = self._fuel_metrics(
            truth=truth,
            dry_mass_kg=dry_mass_kg,
            fuel_capacity_kg=fuel_capacity_kg,
            rocket_state=rocket_state,
            rocket_vehicle_cfg=rocket_vehicle_cfg,
        )
        if trigger == "fuel_below_kg":
            return fuel_kg
        if trigger == "fuel_below_fraction":
            return fuel_frac
        if trigger == "time_gte":
            return t_s
        if trigger == "mode_elapsed_gte":
            if t_s is None or self._active_mode_enter_t_s is None:
                return None
            return float(t_s) - float(self._active_mode_enter_t_s)
        return None

    def _rearm_condition_met(self, *, transition: dict[str, Any], metric_value: float | None) -> bool:
        if metric_value is None or not np.isfinite(float(metric_value)):
            return False
        trigger = str(transition.get("trigger", "") or "").strip().lower()
        suffix = self._metric_suffix(trigger)
        reset_value = None
        if suffix is not None:
            reset_value = transition.get(f"reset_threshold_{suffix}")
        if reset_value is None:
            reset_value = transition.get("reset_threshold")
        if reset_value is None:
            return True
        threshold = float(reset_value)
        if trigger == "range_lt":
            return bool(metric_value > threshold)
        if trigger == "range_gt":
            return bool(metric_value < threshold)
        if trigger in {"fuel_below_kg", "fuel_below_fraction"}:
            return bool(metric_value > threshold)
        return True

    def _maybe_transition(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        dry_mass_kg: float | None,
        rocket_state: RocketState | None,
        rocket_vehicle_cfg: RocketVehicleConfig | None,
        fuel_capacity_kg: float | None,
        t_s: float | None,
    ) -> None:
        active_mode = self._active_mode
        if active_mode is None:
            return
        self._last_transition = None
        for idx, transition in enumerate(self._ordered_transitions):
            if not isinstance(transition, dict):
                continue
            if not self._transition_applies_to_mode(transition, active_mode):
                continue
            to_mode = str(transition.get("to_mode", "") or "").strip()
            if not to_mode or to_mode not in self._modes:
                continue
            metric_value = self._metric_value_for_transition(
                transition=transition,
                truth=truth,
                own_knowledge=own_knowledge,
                dry_mass_kg=dry_mass_kg,
                rocket_state=rocket_state,
                rocket_vehicle_cfg=rocket_vehicle_cfg,
                fuel_capacity_kg=fuel_capacity_kg,
                t_s=t_s,
            )
            if not self._transition_armed.get(idx, True):
                if self._rearm_condition_met(transition=transition, metric_value=metric_value):
                    self._transition_armed[idx] = True
                else:
                    continue
            min_mode_duration_s = float(max(float(transition.get("min_mode_duration_s", 0.0) or 0.0), 0.0))
            if (
                min_mode_duration_s > 0.0
                and self._active_mode_enter_t_s is not None
                and t_s is not None
                and (float(t_s) - float(self._active_mode_enter_t_s)) < (min_mode_duration_s - 1e-12)
            ):
                continue
            fired, detail = self._evaluate_transition(
                transition=transition,
                truth=truth,
                own_knowledge=own_knowledge,
                dry_mass_kg=dry_mass_kg,
                rocket_state=rocket_state,
                rocket_vehicle_cfg=rocket_vehicle_cfg,
                fuel_capacity_kg=fuel_capacity_kg,
                t_s=t_s,
            )
            if not fired:
                continue
            self._active_mode = to_mode
            self._active_mode_enter_t_s = None if t_s is None else float(t_s)
            self._transition_armed[idx] = False
            self._last_transition = {
                "from_mode": active_mode,
                "to_mode": to_mode,
                "trigger": str(transition.get("trigger", "") or ""),
                "detail": detail,
                "min_mode_duration_s": float(max(float(transition.get("min_mode_duration_s", 0.0) or 0.0), 0.0)),
                "priority": int(transition.get("priority", 0) or 0),
            }
            return

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        dry_mass_kg: float | None = None,
        fuel_capacity_kg: float | None = None,
        rocket_state: RocketState | None = None,
        rocket_vehicle_cfg: RocketVehicleConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._modes:
            return {}
        if self._active_mode not in self._modes:
            self._active_mode = next(iter(self._modes.keys()))
        if self._active_mode_enter_t_s is None:
            t_now = kwargs.get("t_s")
            self._active_mode_enter_t_s = None if t_now is None else float(t_now)
        self._maybe_transition(
            truth=truth,
            own_knowledge=own_knowledge,
            dry_mass_kg=dry_mass_kg,
            rocket_state=rocket_state,
            rocket_vehicle_cfg=rocket_vehicle_cfg,
            fuel_capacity_kg=fuel_capacity_kg,
            t_s=(None if kwargs.get("t_s") is None else float(kwargs.get("t_s"))),
        )
        mode = self._modes.get(self._active_mode or "")
        if mode is None:
            return {}
        strategy_out = _call_plugin_method(
            mode.strategy,
            ("update", "plan", "decide"),
            {
                "truth": truth,
                "own_knowledge": own_knowledge,
                "env": dict(kwargs.get("env", {}) or {}),
                "dry_mass_kg": dry_mass_kg,
                "fuel_capacity_kg": fuel_capacity_kg,
                "rocket_state": rocket_state,
                "rocket_vehicle_cfg": rocket_vehicle_cfg,
                **dict(kwargs or {}),
            },
        )
        out = dict(strategy_out or {})
        if mode.execution is not None:
            out["_mission_execution_override"] = mode.execution
        mission_mode = dict(out.get("mission_mode", {}) or {})
        mission_mode["executive_mode"] = str(mode.name)
        if self._last_transition is not None:
            mission_mode["executive_transition"] = dict(self._last_transition)
        out["mission_mode"] = mission_mode
        return out

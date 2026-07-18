# ruff: noqa: F401,F403,F405,I001
from .orbit_od_common import *
from .orbit_od_artifacts import *
from .orbit_od_parameters import *
from .orbit_od_quality import *
from .orbit_od_maneuvers import *

def solve_dynamics_orbit_determination(
    packet: ObservationPacket | Mapping[str, Any],
    *,
    output_dir: str | Path,
    object_id: str | None = None,
    role: str | None = None,
    scenario_name: str = "dynamics_orbit_determination",
    fit_duration_s: float | None = None,
    holdout_duration_s: float | None = None,
    dt_s: float | None = None,
    estimate: str = "state",
    dynamics_model: str = "two_body",
    j2: bool = False,
    drag: bool = False,
    srp: bool = False,
    atmosphere_model: str | None = None,
    orbit_force_model: Mapping[str, Any] | None = None,
    base_specs: Mapping[str, Any] | None = None,
    attitude_source: str = "none",
    attitude_mode: str = "sun_track",
    attitude_body_axis: Sequence[float] | None = None,
    attitude_controller: str = "surrogate_snap",
    attitude_history: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    max_nfev: int = 24,
    robust_loss: str = "linear",
    robust_f_scale: float = 1.0,
    sigma_clip_threshold: float | None = None,
    prior_mean_native: Sequence[float] | None = None,
    prior_covariance_native: Sequence[Sequence[float]] | None = None,
    detect_maneuvers: bool = False,
    maneuver_frame: str = "ric",
    max_maneuver_dv_m_s: float = 25.0,
    maneuver_min_delta_v_m_s: float = 0.05,
    maneuver_min_improvement_ratio: float = 0.35,
    maneuver_burn_duration_s: float | None = None,
    maneuver_guard_observations: int = 3,
    maneuver_max_candidates: int = 12,
    maneuver_max_nfev: int | None = None,
) -> dict[str, Any]:
    packet_obj = _coerce_observation_packet(packet)
    observations = packet_obj.observations
    if len(observations) < 2:
        raise ValueError("dynamics OD requires at least two observations.")
    oid = str(object_id or packet_obj.to_dict().get("object_id") or "target")
    out_role = str(role or oid)
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    orbit_force_model_cfg = _normalize_orbit_force_model(
        orbit_force_model,
        dynamics_model=dynamics_model,
        j2=j2,
        drag=drag,
        srp=srp,
        atmosphere_model=atmosphere_model,
    )

    times = np.array([float(row["time_s"]) for row in observations], dtype=float)
    times = times - float(times[0])
    positions = np.array([row["position_eci_km"] for row in observations], dtype=float)
    position_sigmas = np.array([float(row.get("position_sigma_km") or 1.0) for row in observations], dtype=float)
    has_velocity = all("velocity_eci_km_s" in row for row in observations)
    velocities = np.array([row["velocity_eci_km_s"] for row in observations], dtype=float) if has_velocity else None
    velocity_sigmas = (
        np.array([float(row.get("velocity_sigma_km_s") or 1.0e-3) for row in observations], dtype=float)
        if has_velocity
        else None
    )

    partition = partition_time_arc(
        times,
        fit_duration_s=fit_duration_s,
        holdout_duration_s=holdout_duration_s,
        allow_repeated_epochs=True,
    )
    fit_duration = partition.fit_duration_s
    holdout_duration = partition.holdout_duration_s
    total_duration = max(fit_duration + holdout_duration, fit_duration)
    fit_mask = partition.fit_mask
    holdout_mask = partition.holdout_mask
    if dt_s is None:
        dt = _default_dt_from_times(times)
    else:
        dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError("dt_s must be positive.")

    preliminary_guess = fit_state_from_position_observations(packet_obj, object_id=oid, role=out_role, epoch="first")
    x0, initial_guess_method = _local_initial_state(observations)
    epoch_jd_utc = preliminary_guess.get("epoch_jd_utc")
    selected = selected_orbit_od_parameters(estimate)
    parameters = build_orbit_od_parameter_set(selected)
    attitude_source_key = _normalize_attitude_source(attitude_source)
    attitude_history_rows = _coerce_attitude_history(
        attitude_history if attitude_history is not None else packet_obj.observations
    )
    if attitude_source_key == "observed_history" and not attitude_history_rows:
        raise ValueError("attitude_source='observed_history' requires attitude_quat_bn samples.")
    if (
        attitude_source_key in {"modeled_inline", "modeled_replay"}
        and str(attitude_mode or "").strip().lower() != "sun_track"
    ):
        raise ValueError("Only attitude_mode='sun_track' is currently supported for modeled attitude OD.")
    attitude_body_axis_vec = _attitude_body_axis(attitude_body_axis)
    if attitude_source_key == "modeled_replay":
        attitude_history_rows = _build_modeled_attitude_history(
            object_id=oid,
            role=out_role,
            state=x0,
            epoch_jd_utc=None if epoch_jd_utc is None else float(epoch_jd_utc),
            source_packet=packet_obj,
            base_specs=base_specs,
            scenario_name=scenario_name,
            output_root=output_root,
            duration_s=total_duration,
            dt_s=dt,
            dynamics_model=dynamics_model,
            j2=j2,
            drag=drag,
            srp=srp,
            atmosphere_model=atmosphere_model,
            orbit_force_model=orbit_force_model_cfg,
            attitude_mode=attitude_mode,
            attitude_body_axis=attitude_body_axis_vec,
            attitude_controller=attitude_controller,
        )
    eval_counter = {"count": 0}

    def evaluate(
        native_values: np.ndarray,
        *,
        duration_s: float,
        scratch_name: str,
        parameters_override: ParameterSet | None = None,
        maneuver: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, MissionInputPacket, ScenarioArtifact]:
        eval_counter["count"] += 1
        active_parameters = parameters if parameters_override is None else parameters_override
        parameter_values = active_parameters.mapping(native_values)
        state = _state_from_parameters(x0, parameter_values)
        candidate_packet = _candidate_packet(
            object_id=oid,
            role=out_role,
            state=state,
            epoch_jd_utc=None if epoch_jd_utc is None else float(epoch_jd_utc),
            source_packet=packet_obj,
            base_specs=base_specs,
            parameter_values=parameter_values,
            attitude_history_rows=attitude_history_rows,
        )
        artifact = _candidate_artifact(
            candidate_packet,
            object_id=oid,
            scenario_name=f"{scenario_name}_{scratch_name}",
            output_dir=output_root / "_od_eval_scratch" / f"{scratch_name}_{eval_counter['count']:04d}",
            duration_s=duration_s,
            dt_s=dt,
            dynamics_model=dynamics_model,
            j2=j2,
            drag=drag,
            srp=srp,
            atmosphere_model=atmosphere_model,
            orbit_force_model=orbit_force_model_cfg,
            attitude_source=attitude_source_key,
            attitude_mode=attitude_mode,
            attitude_body_axis=attitude_body_axis_vec,
            attitude_controller=attitude_controller,
            attitude_history_rows=attitude_history_rows,
            maneuver=maneuver,
        )
        requested_epochs = np.unique(times[times <= float(duration_s) + 1.0e-9])
        sim_t, sim_x = evaluate_artifact_at_epochs(
            artifact,
            object_id=oid,
            epochs_s=requested_epochs,
        )
        return sim_t, sim_x, candidate_packet, artifact

    def residual(native_values: np.ndarray) -> np.ndarray:
        sim_t, sim_x, _packet, _artifact = evaluate(native_values, duration_s=fit_duration, scratch_name="fit")
        sim_on_obs = _states_at_epochs(sim_t, sim_x, times[fit_mask])
        return _whiten_state_observation_residuals(
            sim_on_obs,
            observations=[observations[index] for index in np.flatnonzero(fit_mask)],
            reference_positions=positions[fit_mask],
            reference_velocities=velocities[fit_mask] if velocities is not None else None,
            position_sigmas=position_sigmas[fit_mask],
            velocity_sigmas=velocity_sigmas[fit_mask] if velocity_sigmas is not None else None,
        )

    prefit_t, prefit_x, _prefit_packet, _prefit_artifact = evaluate(
        parameters.initial_native(),
        duration_s=fit_duration,
        scratch_name="prefit_metrics",
    )
    prefit_on_obs = _states_at_epochs(prefit_t, prefit_x, times[fit_mask])
    prefit_metrics = _state_error_metrics(
        prefit_on_obs,
        positions[fit_mask],
        velocities[fit_mask] if velocities is not None else None,
    )
    solve = solve_batch_least_squares(
        parameters,
        residual,
        max_nfev=max_nfev,
        robust_loss=robust_loss,
        robust_f_scale=robust_f_scale,
        sigma_clip_threshold=sigma_clip_threshold,
        prior_mean_native=None if prior_mean_native is None else np.asarray(prior_mean_native, dtype=float),
        prior_covariance_native=None
        if prior_covariance_native is None
        else np.asarray(prior_covariance_native, dtype=float),
    )
    fit_t, fit_x, fitted_packet, fitted_artifact = evaluate(
        solve.x_native, duration_s=fit_duration, scratch_name="postfit"
    )
    fit_on_obs = _states_at_epochs(fit_t, fit_x, times[fit_mask])
    fit_metrics = _state_error_metrics(
        fit_on_obs, positions[fit_mask], velocities[fit_mask] if velocities is not None else None
    )

    pred_t, pred_x, prediction_packet, prediction_artifact = evaluate(
        solve.x_native,
        duration_s=total_duration,
        scratch_name="prediction",
    )
    holdout_t_obs = times[holdout_mask]
    holdout_ref_pos = positions[holdout_mask]
    holdout_ref_vel = velocities[holdout_mask] if velocities is not None else None
    pred_on_holdout = _states_at_epochs(pred_t, pred_x, holdout_t_obs)
    holdout_metrics = _state_error_metrics(pred_on_holdout, holdout_ref_pos, holdout_ref_vel)

    fit_whitened_residual = _whiten_state_observation_residuals(
        fit_on_obs,
        observations=[observations[index] for index in np.flatnonzero(fit_mask)],
        reference_positions=positions[fit_mask],
        reference_velocities=velocities[fit_mask] if velocities is not None else None,
        position_sigmas=position_sigmas[fit_mask],
        velocity_sigmas=velocity_sigmas[fit_mask] if velocity_sigmas is not None else None,
    )
    holdout_whitened_residual = _whiten_state_observation_residuals(
        pred_on_holdout,
        observations=[observations[index] for index in np.flatnonzero(holdout_mask)],
        reference_positions=holdout_ref_pos,
        reference_velocities=holdout_ref_vel,
        position_sigmas=position_sigmas[holdout_mask],
        velocity_sigmas=velocity_sigmas[holdout_mask] if velocity_sigmas is not None else None,
    )

    estimated_parameters = parameters.metadata(solve.x_native)
    solver_summary = {
        "success": bool(solve.success),
        "message": solve.message,
        "nfev": int(solve.nfev),
        "initial_cost": float(solve.initial_cost),
        "final_cost": float(solve.cost),
        "rms_weighted_residual": float(solve.rms_residual),
        "diagnostics": solve.diagnostics,
        "decision_records": _label_state_residual_decisions(
            solve.decision_records,
            observations=[observations[index] for index in np.flatnonzero(fit_mask)],
            include_velocity=velocities is not None,
        ),
    }
    state_component_count = 6 if velocities is not None else 3
    residual_audit = build_residual_audit(
        [
            *residual_records_from_vectors(
                fit_whitened_residual,
                partition="fit",
                measurement_type="eci_position_velocity" if velocities is not None else "eci_position",
                component_count=state_component_count,
                arc_id=scenario_name,
            ),
            *residual_records_from_vectors(
                holdout_whitened_residual,
                partition="holdout",
                measurement_type="eci_position_velocity" if velocities is not None else "eci_position",
                component_count=state_component_count,
                arc_id=scenario_name,
            ),
        ],
        decision_records=solver_summary["decision_records"],
    )
    quality_gates = build_dynamics_od_quality_gates(
        solver=solver_summary,
        parameter_metadata=estimated_parameters,
        covariance=solve.covariance_native,
        prefit_position_rms_m=float(prefit_metrics["position_rms_m"]),
        fit_metrics=fit_metrics,
        holdout_metrics=holdout_metrics,
        selected_parameters=selected,
        attitude_source=attitude_source_key,
        attitude_history_rows=attitude_history_rows,
        base_specs=base_specs,
    )

    fitted_packet_path = output_root / "fitted_mission_input_packet.json"
    estimated_parameters_path = output_root / "estimated_parameters.json"
    materialized_fit_config_path = output_root / "materialized_fit_config.yaml"
    materialized_prediction_config_path = output_root / "materialized_prediction_config.yaml"
    fit_residuals_csv = output_root / "fit_residuals.csv"
    holdout_errors_csv = output_root / "holdout_errors.csv"
    report_json_path = output_root / "od_fit_report.json"
    report_md_path = output_root / "od_fit_report.md"
    fit_plot_path = output_root / "fit_position_residuals.png"
    holdout_plot_path = output_root / "holdout_position_errors.png"

    write_json(str(fitted_packet_path), fitted_packet.to_dict())
    write_json(
        str(estimated_parameters_path),
        {
            "parameters": estimated_parameters,
            "covariance": None if solve.covariance_native is None else solve.covariance_native.tolist(),
        },
    )
    fitted_artifact.write(materialized_fit_config_path)
    prediction_artifact.write(materialized_prediction_config_path)
    _write_error_csv(
        fit_residuals_csv,
        times[fit_mask],
        fit_on_obs,
        positions[fit_mask],
        velocities[fit_mask] if velocities is not None else None,
        observations=[observations[idx] for idx in np.flatnonzero(fit_mask)],
        partition="fit",
    )
    _write_error_csv(
        holdout_errors_csv,
        holdout_t_obs,
        pred_on_holdout,
        holdout_ref_pos,
        holdout_ref_vel,
        observations=[observations[idx] for idx in np.flatnonzero(holdout_mask)],
        partition="holdout",
    )
    _write_error_plot(
        fit_plot_path,
        times[fit_mask],
        fit_on_obs[:, :3] - positions[fit_mask],
        title="Dynamics OD Fit Position Residuals",
    )
    _write_error_plot(
        holdout_plot_path,
        holdout_t_obs - fit_duration,
        pred_on_holdout[:, :3] - holdout_ref_pos,
        title="Dynamics OD Holdout Position Errors",
    )

    result = {
        "method": "dynamics_orbit_least_squares",
        "observation_packet_source": dict(packet_obj.to_dict().get("source", {}) or {}),
        "object_id": oid,
        "scenario_name": scenario_name,
        "dynamics_model": dynamics_model,
        "j2": bool(j2),
        "drag": bool(drag),
        "srp": bool(srp),
        "atmosphere_model": atmosphere_model or "",
        "orbit_force_model": orbit_force_model_cfg,
        "attitude_source": attitude_source_key,
        "attitude_mode": str(attitude_mode or ""),
        "attitude_body_axis": attitude_body_axis_vec.tolist(),
        "attitude_history_sample_count": len(attitude_history_rows),
        "fit_duration_s": fit_duration,
        "holdout_duration_s": holdout_duration,
        "dt_s": dt,
        "observation_partition": partition.summary,
        "epoch_evaluation": exact_epoch_provenance(times[fit_mask | holdout_mask]),
        "estimate_spec": estimate,
        "estimation_policy": {
            "robust_loss": str(robust_loss),
            "robust_f_scale": float(robust_f_scale),
            "sigma_clip_threshold": sigma_clip_threshold,
            "prior_enabled": prior_mean_native is not None,
        },
        "selected_parameters": selected,
        "observation_count": len(observations),
        "fit_observation_count": int(np.count_nonzero(fit_mask)),
        "holdout_observation_count": int(np.count_nonzero(holdout_mask)),
        "initial_guess": {
            "method": initial_guess_method,
            "state_eci_km_s": x0.tolist(),
            "preliminary_batch_position_fit": {
                "state_eci_km_s": preliminary_guess.get("state_eci_km_s"),
                "diagnostics": dict(preliminary_guess.get("diagnostics", {}) or {}),
            },
        },
        "estimated_parameters": estimated_parameters,
        "derived_parameters": _derived_estimated_parameters(estimated_parameters, base_specs=base_specs),
        "solver": solver_summary,
        "quality_gates": quality_gates,
        "verdict": _dynamics_od_verdict(quality_gates),
        "prefit_metrics": prefit_metrics,
        "fit_metrics": fit_metrics,
        "holdout_metrics": holdout_metrics,
        "residual_audit": residual_audit,
        "fitted_mission_input_packet_path": str(fitted_packet_path),
        "estimated_parameters_path": str(estimated_parameters_path),
        "fit_residuals_csv": str(fit_residuals_csv),
        "holdout_errors_csv": str(holdout_errors_csv),
        "fit_plot_path": str(fit_plot_path),
        "holdout_plot_path": str(holdout_plot_path),
        "materialized_fit_config_path": str(materialized_fit_config_path),
        "materialized_prediction_config_path": str(materialized_prediction_config_path),
        "report_json_path": str(report_json_path),
        "report_md_path": str(report_md_path),
    }
    if detect_maneuvers:
        maneuver_report = _detect_single_maneuver(
            output_root=output_root,
            object_id=oid,
            selected_parameters=selected,
            base_parameters=parameters,
            baseline_native_values=solve.x_native,
            baseline_fit_metrics=fit_metrics,
            baseline_holdout_metrics=holdout_metrics,
            baseline_covariance=solve.covariance_native,
            evaluate=evaluate,
            times=times,
            fit_mask=fit_mask,
            holdout_mask=holdout_mask,
            observations=observations,
            positions=positions,
            velocities=velocities,
            position_sigmas=position_sigmas,
            velocity_sigmas=velocity_sigmas,
            fit_duration_s=fit_duration,
            total_duration_s=total_duration,
            dt_s=dt,
            frame=maneuver_frame,
            max_delta_v_m_s=max_maneuver_dv_m_s,
            min_delta_v_m_s=maneuver_min_delta_v_m_s,
            min_improvement_ratio=maneuver_min_improvement_ratio,
            burn_duration_s=maneuver_burn_duration_s,
            guard_observations=maneuver_guard_observations,
            max_candidates=maneuver_max_candidates,
            max_nfev=max_nfev if maneuver_max_nfev is None else int(maneuver_max_nfev),
            robust_loss=robust_loss,
            robust_f_scale=robust_f_scale,
            sigma_clip_threshold=sigma_clip_threshold,
        )
        result["maneuver_detection"] = maneuver_report
        for key, value in dict(maneuver_report.get("artifacts", {}) or {}).items():
            result[f"maneuver_{key}"] = value
    write_json(str(report_json_path), result)
    _write_report_md(report_md_path, result)
    return result

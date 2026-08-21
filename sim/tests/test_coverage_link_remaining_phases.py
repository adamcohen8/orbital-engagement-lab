from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sim.analysis.communications_coverage import (
    CommunicationsCoverageConfig,
    EarthTerminalProfile,
    evaluate_communications_coverage,
    write_communications_coverage_artifacts,
)
from sim.analysis.coverage_aggregation import (
    ConstellationCoverageConfig,
    evaluate_constellation_coverage,
    write_constellation_coverage_artifacts,
)
from sim.analysis.coverage_queries import CoverageRegionMask, evaluate_coverage_queries
from sim.analysis.coverage_sensitivity import (
    CoverageSensitivityCriteria,
    evaluate_coverage_sensitivity,
    write_coverage_sensitivity_evidence,
)
from sim.analysis.coverage_tasking import (
    CoverageTaskingConfig,
    TaskingConstraints,
    TaskOpportunity,
    optimize_coverage_tasking,
    write_coverage_tasking_artifacts,
)
from sim.analysis.directed_link import (
    BOLTZMANN_J_K,
    SPEED_OF_LIGHT_M_S,
    DirectedLinkConfig,
    LinkEndpointHistory,
    LinkTerminal,
    TerminalPattern,
    evaluate_directed_link,
    evaluate_directed_link_sample,
    fixed_wgs84_site_history,
    free_space_link_ledger,
    spacecraft_endpoint_history,
    write_directed_link_artifacts,
)
from sim.analysis.directed_link_runtime import (
    AuthorizedLinkRuntimeMonitor,
    RuntimeLinkEvaluation,
    RuntimeLinkMonitorConfig,
)
from sim.analysis.event_refinement import availability_intervals, refine_availability_transitions
from sim.analysis.global_coverage import summarize_sampled_coverage_mask
from sim.analysis.healpix import healpix_npix, healpix_wgs84_centers
from sim.dynamics.orbit.frames import (
    FrameContext,
    eci_to_ecef_rotation_context,
    transform_state,
)
from sim.utils.geodesy import WGS84_A_KM
from sim.utils.quaternion import dcm_to_quaternion_bn


def _identity_terminal(
    terminal_id: str,
    asset_id: str,
    *,
    parent_frame: str = "body",
    pattern: TerminalPattern | None = None,
    mounting: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> LinkTerminal:
    return LinkTerminal(
        terminal_id=terminal_id,
        asset_id=asset_id,
        parent_frame=parent_frame,
        quat_parent_from_terminal=mounting,
        pattern=pattern or TerminalPattern("constant", 0.0),
    )


def _link_config(
    tx: LinkTerminal,
    rx: LinkTerminal,
    *,
    required_eb_n0_db: float = -100.0,
    min_elevation_rad: float | None = None,
) -> DirectedLinkConfig:
    return DirectedLinkConfig(
        analysis_id="link_fixture",
        link_id="tx_to_rx",
        tx_terminal=tx,
        rx_terminal=rx,
        carrier_frequency_hz=2.2e9,
        tx_power_w=10.0,
        data_rate_bps=1.0e6,
        system_noise_temperature_k=500.0,
        required_eb_n0_db=required_eb_n0_db,
        tx_line_loss_db=1.0,
        rx_line_loss_db=2.0,
        misc_loss_db=3.0,
        min_fixed_site_elevation_rad=min_elevation_rad,
    )


def _spacecraft_history(
    asset_id: str,
    positions: np.ndarray,
    *,
    attitudes: np.ndarray | None = None,
) -> object:
    times = np.array([0.0, 10.0, 20.0])
    return spacecraft_endpoint_history(
        asset_id=asset_id,
        state_provider_id=f"{asset_id}.truth",
        times_s=times,
        positions_eci_km=np.broadcast_to(positions, (times.size, 3)).copy(),
        velocities_eci_km_s=np.zeros((times.size, 3)),
        attitudes_quat_bn=attitudes,
        attitude_source_kind="not_required" if attitudes is None else "achieved",
        attitude_provider_id=None if attitudes is None else f"{asset_id}.attitude",
    )


def _attitude_for_boresight_eci(boresight_eci: np.ndarray) -> np.ndarray:
    body_z = np.asarray(boresight_eci, dtype=float)
    body_z /= np.linalg.norm(body_z)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, body_z))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    body_x = np.cross(reference, body_z)
    body_x /= np.linalg.norm(body_x)
    body_y = np.cross(body_z, body_x)
    return dcm_to_quaternion_bn(np.vstack((body_x, body_y, body_z)))


def _fixed_ecef_spacecraft_evidence(
    times: np.ndarray,
    *,
    position_ecef: np.ndarray,
    boresight_ecef: np.ndarray,
) -> tuple[FrameContext, np.ndarray, np.ndarray]:
    context = FrameContext(jd_utc_start=2451545.0)
    positions = []
    attitudes = []
    for time_s in times:
        rotation = eci_to_ecef_rotation_context(float(time_s), context)
        positions.append(rotation.T @ position_ecef)
        attitudes.append(_attitude_for_boresight_eci(rotation.T @ boresight_ecef))
    return context, np.asarray(positions), np.asarray(attitudes)


def test_free_space_ledger_matches_independent_hand_calculation_and_scaling() -> None:
    ranges = np.array([1000.0, 2000.0])
    ledger = free_space_link_ledger(
        ranges,
        carrier_frequency_hz=2.2e9,
        tx_power_w=10.0,
        tx_gain_dbi=6.0,
        rx_gain_dbi=12.0,
        data_rate_bps=1.0e6,
        system_noise_temperature_k=500.0,
        required_eb_n0_db=9.6,
        tx_line_loss_db=1.0,
        rx_line_loss_db=2.0,
        misc_loss_db=3.0,
    )
    path_loss = 20.0 * math.log10(4.0 * math.pi * 1.0e6 * 2.2e9 / SPEED_OF_LIGHT_M_S)
    tx_power_dbw = 10.0 * math.log10(10.0)
    eirp = tx_power_dbw + 6.0 - 1.0
    received = eirp + 12.0 - path_loss - 2.0 - 3.0
    noise_density = 10.0 * math.log10(BOLTZMANN_J_K * 500.0)
    expected_margin = received - noise_density - 10.0 * math.log10(1.0e6) - 9.6
    assert ledger.margin_db[0] == pytest.approx(expected_margin, abs=1.0e-12)
    assert ledger.free_space_path_loss_db[1] - ledger.free_space_path_loss_db[0] == pytest.approx(
        20.0 * math.log10(2.0),
        abs=1.0e-12,
    )
    zero_margin = free_space_link_ledger(
        np.array([1000.0]),
        carrier_frequency_hz=2.2e9,
        tx_power_w=10.0,
        tx_gain_dbi=6.0,
        rx_gain_dbi=12.0,
        data_rate_bps=1.0e6,
        system_noise_temperature_k=500.0,
        required_eb_n0_db=float(ledger.eb_n0_db[0]),
        tx_line_loss_db=1.0,
        rx_line_loss_db=2.0,
        misc_loss_db=3.0,
    )
    assert zero_margin.margin_db[0] == pytest.approx(0.0, abs=1.0e-12)
    assert bool(zero_margin.margin_pass[0]) is True


def test_free_space_ledger_has_expected_power_loss_temperature_and_rate_monotonicity() -> None:
    common = dict(
        range_km=np.array([1000.0]),
        carrier_frequency_hz=2.2e9,
        tx_power_w=10.0,
        tx_gain_dbi=6.0,
        rx_gain_dbi=12.0,
        data_rate_bps=1.0e6,
        system_noise_temperature_k=500.0,
        required_eb_n0_db=9.6,
    )
    baseline = free_space_link_ledger(**common).margin_db[0]
    assert free_space_link_ledger(**{**common, "tx_power_w": 20.0}).margin_db[0] > baseline
    assert free_space_link_ledger(**{**common, "tx_gain_dbi": 7.0}).margin_db[0] > baseline
    assert free_space_link_ledger(**{**common, "misc_loss_db": 1.0}).margin_db[0] < baseline
    assert free_space_link_ledger(
        **{**common, "system_noise_temperature_k": 1000.0}
    ).margin_db[0] < baseline
    assert free_space_link_ledger(**{**common, "data_rate_bps": 2.0e6}).margin_db[0] < baseline


def test_directed_link_spacecraft_geometry_directional_mounting_and_occultation() -> None:
    identity_attitudes = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    body_from_terminal = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    )
    mounting = tuple(float(value) for value in dcm_to_quaternion_bn(body_from_terminal))
    tx_terminal = _identity_terminal(
        "tx.directional",
        "tx",
        pattern=TerminalPattern("axisymmetric_hard_cone", 6.0, np.deg2rad(1.0)),
        mounting=mounting,
    )
    rx_terminal = _identity_terminal("rx.constant", "rx")
    tx_history = _spacecraft_history(
        "tx",
        np.array([7000.0, 0.0, 0.0]),
        attitudes=identity_attitudes,
    )
    rx_history = _spacecraft_history("rx", np.array([7000.0, 1000.0, 0.0]))
    clear = evaluate_directed_link(
        _link_config(tx_terminal, rx_terminal),
        tx_history=tx_history,
        rx_history=rx_history,
        frame_context=FrameContext(jd_utc_start=2451545.0),
    )
    np.testing.assert_array_equal(clear.samples.available, True)
    np.testing.assert_allclose(clear.samples.tx_off_axis_rad, 0.0, atol=1.0e-12)

    occulted_rx = _spacecraft_history("rx", np.array([-7000.0, 0.0, 0.0]))
    occulted = evaluate_directed_link(
        _link_config(_identity_terminal("tx.constant", "tx"), rx_terminal),
        tx_history=_spacecraft_history("tx", np.array([7000.0, 0.0, 0.0])),
        rx_history=occulted_rx,
        frame_context=FrameContext(jd_utc_start=2451545.0),
    )
    np.testing.assert_array_equal(occulted.samples.available, False)
    assert set(occulted.samples.primary_reason) == {"earth_occulted"}

    tangent = evaluate_directed_link(
        _link_config(_identity_terminal("tx.constant", "tx"), rx_terminal),
        tx_history=_spacecraft_history("tx", np.array([WGS84_A_KM, -1000.0, 0.0])),
        rx_history=_spacecraft_history("rx", np.array([WGS84_A_KM, 1000.0, 0.0])),
        frame_context=FrameContext(jd_utc_start=2451545.0),
    )
    assert set(tangent.samples.primary_reason) == {"earth_occulted"}

    with pytest.raises(ValueError, match="inside WGS84"):
        evaluate_directed_link(
            _link_config(_identity_terminal("tx.constant", "tx"), rx_terminal),
            tx_history=_spacecraft_history("tx", np.array([0.0, 0.0, 0.0])),
            rx_history=_spacecraft_history("rx", np.array([7000.0, 1000.0, 0.0])),
            frame_context=FrameContext(jd_utc_start=2451545.0),
        )


def test_directed_link_scalar_runtime_and_batch_samples_have_exact_parity() -> None:
    context = FrameContext(jd_utc_start=2451545.0)
    config = _link_config(
        _identity_terminal("tx.constant", "tx"),
        _identity_terminal("rx.constant", "rx"),
    )
    batch = evaluate_directed_link(
        config,
        tx_history=_spacecraft_history("tx", np.array([7000.0, 0.0, 0.0])),
        rx_history=_spacecraft_history("rx", np.array([7000.0, 1000.0, 0.0])),
        frame_context=context,
    )
    scalar_tx = spacecraft_endpoint_history(
        asset_id="tx",
        state_provider_id="tx.truth",
        times_s=np.array([0.0]),
        positions_eci_km=np.array([[7000.0, 0.0, 0.0]]),
        velocities_eci_km_s=np.zeros((1, 3)),
        attitudes_quat_bn=None,
        attitude_source_kind="not_required",
        attitude_provider_id=None,
    )
    scalar_rx = spacecraft_endpoint_history(
        asset_id="rx",
        state_provider_id="rx.truth",
        times_s=np.array([0.0]),
        positions_eci_km=np.array([[7000.0, 1000.0, 0.0]]),
        velocities_eci_km_s=np.zeros((1, 3)),
        attitudes_quat_bn=None,
        attitude_source_kind="not_required",
        attitude_provider_id=None,
    )
    scalar = evaluate_directed_link_sample(
        config,
        tx_history=scalar_tx,
        rx_history=scalar_rx,
        frame_context=context,
    )
    for field_name in batch.samples.__dataclass_fields__:
        batch_value = getattr(batch.samples, field_name)
        scalar_value = getattr(scalar.samples, field_name)
        if isinstance(batch_value, tuple):
            assert scalar_value == batch_value[:1]
        else:
            np.testing.assert_array_equal(scalar_value, batch_value[:1])


def test_directed_link_fixed_site_zenith_elevation_boundary() -> None:
    times = np.array([0.0, 10.0, 20.0])
    context = FrameContext(jd_utc_start=2451545.0)
    site = fixed_wgs84_site_history(
        asset_id="site",
        state_provider_id="site.wgs84",
        times_s=times,
        geodetic_latitude_deg=0.0,
        longitude_deg=0.0,
        ellipsoidal_height_km=0.0,
        frame_context=context,
    )
    spacecraft_ecef = np.array([WGS84_A_KM + 500.0, 0.0, 0.0])
    positions = []
    velocities = []
    attitudes = []
    for time_s in times:
        position, velocity = transform_state(
            spacecraft_ecef,
            np.zeros(3),
            "ecef",
            "eci",
            t_s=float(time_s),
            context=context,
        )
        rotation = eci_to_ecef_rotation_context(float(time_s), context)
        positions.append(position)
        velocities.append(velocity)
        attitudes.append(_attitude_for_boresight_eci(rotation.T @ np.array([-1.0, 0.0, 0.0])))
    spacecraft = spacecraft_endpoint_history(
        asset_id="spacecraft",
        state_provider_id="spacecraft.truth",
        times_s=times,
        positions_eci_km=np.asarray(positions),
        velocities_eci_km_s=np.asarray(velocities),
        attitudes_quat_bn=np.asarray(attitudes),
        attitude_source_kind="achieved",
        attitude_provider_id="spacecraft.attitude",
    )
    result = evaluate_directed_link(
        _link_config(
            _identity_terminal(
                "spacecraft.tx",
                "spacecraft",
                pattern=TerminalPattern("axisymmetric_hard_cone", 6.0, np.deg2rad(5.0)),
            ),
            _identity_terminal("site.rx", "site", parent_frame="enu"),
            min_elevation_rad=0.5 * np.pi,
        ),
        tx_history=spacecraft,
        rx_history=site,
        frame_context=context,
    )
    np.testing.assert_array_equal(result.samples.available, True)
    np.testing.assert_allclose(result.samples.fixed_site_elevation_rad, 0.5 * np.pi, atol=1.0e-10)


def test_event_refinement_converges_and_sample_bounded_fallback_is_explicit() -> None:
    times = np.array([0.0, 10.0])
    available = np.array([False, True])
    reasons = ("beyond_max_range", "available")
    sample_bounded = refine_availability_transitions(times, available, reasons)
    assert sample_bounded[0].disposition == "sample_bounded"
    refined = refine_availability_transitions(
        times,
        available,
        reasons,
        evaluator_at_time=lambda time_s: (
            time_s >= 4.0,
            "available" if time_s >= 4.0 else "beyond_max_range",
        ),
        time_tolerance_s=1.0e-4,
        max_iterations=64,
    )
    assert refined[0].disposition == "provider_refined"
    assert refined[0].time_s == pytest.approx(4.0, abs=1.0e-4)
    intervals = availability_intervals(times, available, reasons, transitions=refined)
    assert intervals[0].start_s == pytest.approx(4.0, abs=1.0e-4)
    assert intervals[0].end_censored is True


def test_refined_intervals_preserve_boundary_reasons_and_reject_mismatched_evidence() -> None:
    times = np.array([0.0, 10.0])
    available = np.array([False, True])
    reasons = ("earth_occulted", "available")
    refined = refine_availability_transitions(
        times,
        available,
        reasons,
        evaluator_at_time=lambda time_s: (
            time_s >= 4.0,
            "available" if time_s >= 4.0 else "tx_outside_pattern",
        ),
        time_tolerance_s=1.0e-4,
        max_iterations=64,
    )
    interval = availability_intervals(times, available, reasons, transitions=refined)[0]
    assert interval.acquisition_reason == "tx_outside_pattern"
    malformed = replace(refined[0], transition_kind="loss")
    with pytest.raises(ValueError, match="kind"):
        availability_intervals(times, available, reasons, transitions=(malformed,))


def test_directed_link_endpoint_history_rejects_unvalidated_frame_evidence() -> None:
    with pytest.raises(ValueError, match="shape"):
        LinkEndpointHistory(
            asset_id="spacecraft",
            state_provider_id="truth",
            endpoint_kind="spacecraft",
            times_s=np.array([0.0]),
            position_eci_km=np.zeros((1, 2)),
            velocity_eci_km_s=np.zeros((1, 3)),
            dcm_parent_from_eci=None,
            attitude_source_kind="not_required",
            attitude_provider_id=None,
        )
    with pytest.raises(ValueError, match="orthonormal"):
        LinkEndpointHistory(
            asset_id="spacecraft",
            state_provider_id="truth",
            endpoint_kind="spacecraft",
            times_s=np.array([0.0]),
            position_eci_km=np.array([[7000.0, 0.0, 0.0]]),
            velocity_eci_km_s=np.zeros((1, 3)),
            dcm_parent_from_eci=np.zeros((1, 3, 3)),
            attitude_source_kind="achieved",
            attitude_provider_id="attitude",
        )


def test_authorized_runtime_monitor_delays_delivery_to_prevent_zero_time_loop() -> None:
    monitor = AuthorizedLinkRuntimeMonitor(
        RuntimeLinkMonitorConfig(
            monitor_id="link_monitor",
            link_id="spacecraft_to_site",
            authorized_consumer_id="mission.downlink_logic",
            link_config_semantic_sha256="b" * 64,
            task_period_s=5.0,
        )
    )
    event = monitor.evaluate_after_state_commit(
        0.0,
        lambda _time_s: RuntimeLinkEvaluation(
            available=True,
            margin_db=3.0,
            primary_reason="available",
            link_config_semantic_sha256="b" * 64,
            evidence_sha256="c" * 64,
        ),
    )
    assert event is not None and event.eligible_delivery_time_s == 5.0
    assert monitor.deliver_due(0.0, consumer_id="mission.downlink_logic") == ()
    with pytest.raises(PermissionError, match="authorized consumer"):
        monitor.deliver_due(5.0, consumer_id="different.consumer")
    delivered = monitor.deliver_due(5.0, consumer_id="mission.downlink_logic")
    assert delivered == (event,)


def test_runtime_monitor_rejects_unbound_or_inconsistent_link_evidence() -> None:
    config = RuntimeLinkMonitorConfig(
        monitor_id="link_monitor",
        link_id="spacecraft_to_site",
        authorized_consumer_id="mission.downlink_logic",
        link_config_semantic_sha256="b" * 64,
        task_period_s=5.0,
    )
    monitor = AuthorizedLinkRuntimeMonitor(config)
    with pytest.raises(ValueError, match="not bound"):
        monitor.evaluate_after_state_commit(
            0.0,
            lambda _time_s: RuntimeLinkEvaluation(
                available=True,
                margin_db=3.0,
                primary_reason="available",
                link_config_semantic_sha256="d" * 64,
                evidence_sha256="c" * 64,
            ),
        )
    with pytest.raises(ValueError, match="inconsistent"):
        RuntimeLinkEvaluation(
            available=False,
            margin_db=3.0,
            primary_reason="available",
            link_config_semantic_sha256="b" * 64,
            evidence_sha256="c" * 64,
        )


def _communications_config(
    *,
    analysis_id: str = "comm_a",
    required_eb_n0_db: float = -100.0,
    chunk_size: int = 777,
) -> CommunicationsCoverageConfig:
    return CommunicationsCoverageConfig(
        analysis_id=analysis_id,
        service_id="global_s_band_downlink",
        source_asset_id="spacecraft",
        state_provider_id="spacecraft.truth",
        attitude_source_kind="achieved",
        attitude_provider_id="spacecraft.attitude",
        source_terminal_id="spacecraft.s_band_tx",
        source_terminal_pattern=TerminalPattern(
            "axisymmetric_hard_cone",
            6.0,
            np.deg2rad(25.0),
        ),
        quat_body_from_terminal=(1.0, 0.0, 0.0, 0.0),
        earth_terminal_profile=EarthTerminalProfile(
            profile_id="notional_s_band_rx",
            provenance="test fixture v1",
            pattern=TerminalPattern("constant", 12.0),
            minimum_elevation_rad=0.0,
        ),
        direction="spacecraft_to_earth",
        order=5,
        carrier_frequency_hz=2.2e9,
        tx_power_w=10.0,
        data_rate_bps=1.0e6,
        system_noise_temperature_k=500.0,
        required_eb_n0_db=required_eb_n0_db,
        chunk_size=chunk_size,
    )


def test_global_communications_coverage_requires_rf_closure_and_query_compatibility() -> None:
    times = np.array([0.0, 60.0, 120.0])
    context, positions, attitudes = _fixed_ecef_spacecraft_evidence(
        times,
        position_ecef=np.array([WGS84_A_KM + 500.0, 0.0, 0.0]),
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    passing = evaluate_communications_coverage(
        _communications_config(),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    failing = evaluate_communications_coverage(
        _communications_config(analysis_id="comm_fail", required_eb_n0_db=1000.0),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    assert np.all(passing.covered_cell_count > 0)
    np.testing.assert_array_equal(failing.covered_cell_count, 0)
    assert failing.summary["primary_reason_total"]["negative_margin"] > 0
    query = evaluate_coverage_queries(
        passing,
        region_masks=[
            CoverageRegionMask(
                region_id="whole_earth",
                mask_version="v1",
                provenance="all canonical cells",
                cell_indices=tuple(range(healpix_npix(5))),
            )
        ],
    )
    np.testing.assert_array_equal(query.regions[0].covered_cell_count, passing.covered_cell_count)


def test_communications_coverage_chunk_and_artifact_semantics_are_deterministic(tmp_path) -> None:
    times = np.array([0.0, 60.0])
    context, positions, attitudes = _fixed_ecef_spacecraft_evidence(
        times,
        position_ecef=np.array([WGS84_A_KM + 500.0, 0.0, 0.0]),
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    first = evaluate_communications_coverage(
        _communications_config(chunk_size=257),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    second = evaluate_communications_coverage(
        _communications_config(chunk_size=4096),
        times_s=times,
        positions_eci_km=positions,
        attitudes_quat_bn=attitudes,
        frame_context=context,
    )
    np.testing.assert_array_equal(first.covered_cell_count, second.covered_cell_count)
    assert first.interval_semantic_sha256 == second.interval_semantic_sha256
    # Execution chunking remains manifest provenance but does not redefine scientific identity.
    artifacts = write_communications_coverage_artifacts(first, tmp_path / "communications")
    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    assert manifest["semantic_sha256"] == first.interval_semantic_sha256


def test_cadence_sensitivity_packet_is_source_bound_and_limit_gated(tmp_path) -> None:
    context, baseline_positions, baseline_attitudes = _fixed_ecef_spacecraft_evidence(
        np.array([0.0, 120.0]),
        position_ecef=np.array([WGS84_A_KM + 500.0, 0.0, 0.0]),
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    _, refined_positions, refined_attitudes = _fixed_ecef_spacecraft_evidence(
        np.array([0.0, 60.0, 120.0]),
        position_ecef=np.array([WGS84_A_KM + 500.0, 0.0, 0.0]),
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    baseline = evaluate_communications_coverage(
        _communications_config(analysis_id="cadence_baseline"),
        times_s=np.array([0.0, 120.0]),
        positions_eci_km=baseline_positions,
        attitudes_quat_bn=baseline_attitudes,
        frame_context=context,
    )
    refined = evaluate_communications_coverage(
        _communications_config(analysis_id="cadence_refined"),
        times_s=np.array([0.0, 60.0, 120.0]),
        positions_eci_km=refined_positions,
        attitudes_quat_bn=refined_attitudes,
        frame_context=context,
    )
    result = evaluate_coverage_sensitivity(
        comparison_id="cadence_fixture",
        comparison_kind="cadence",
        baseline=baseline,
        refined=refined,
        criteria=CoverageSensitivityCriteria(0.0, 0.0),
    )
    assert result.passed is True
    packet = write_coverage_sensitivity_evidence(result, tmp_path / "cadence.json")
    assert json.loads(packet.read_text(encoding="utf-8"))["semantic_sha256"] == result.semantic_sha256
    assert len(result.matched_assumptions_sha256) == 64

    changed_service = replace(
        refined,
        config=replace(refined.config, required_eb_n0_db=refined.config.required_eb_n0_db + 1.0),
    )
    with pytest.raises(ValueError, match="non-refinement scientific assumptions"):
        evaluate_coverage_sensitivity(
            comparison_id="changed_service",
            comparison_kind="cadence",
            baseline=baseline,
            refined=changed_service,
            criteria=CoverageSensitivityCriteria(1.0, 1.0),
        )

    shifted_times = np.array([0.0, 30.0, 50.0, 120.0])
    _, shifted_positions, shifted_attitudes = _fixed_ecef_spacecraft_evidence(
        shifted_times,
        position_ecef=np.array([WGS84_A_KM + 500.0, 0.0, 0.0]),
        boresight_ecef=np.array([-1.0, 0.0, 0.0]),
    )
    shifted_epochs = evaluate_communications_coverage(
        _communications_config(analysis_id="cadence_shifted"),
        times_s=shifted_times,
        positions_eci_km=shifted_positions,
        attitudes_quat_bn=shifted_attitudes,
        frame_context=context,
    )
    with pytest.raises(ValueError, match="retain every baseline epoch"):
        evaluate_coverage_sensitivity(
            comparison_id="shifted_epochs",
            comparison_kind="cadence",
            baseline=refined,
            refined=shifted_epochs,
            criteria=CoverageSensitivityCriteria(1.0, 1.0),
        )


def _synthetic_coverage(analysis_id: str, mask: np.ndarray) -> object:
    order = 5
    centers = healpix_wgs84_centers(order)
    metrics = summarize_sampled_coverage_mask(mask, np.array([0.0, 10.0, 20.0]))
    return SimpleNamespace(
        config=SimpleNamespace(analysis_id=analysis_id, order=order),
        times_s=np.array([0.0, 10.0, 20.0]),
        covered_cell_count=np.count_nonzero(mask, axis=1),
        instantaneous_covered_fraction=np.count_nonzero(mask, axis=1) / mask.shape[1],
        cell_geodetic_latitude_deg=np.rad2deg(centers.geodetic_latitude_rad),
        cell_longitude_deg=np.rad2deg(centers.longitude_rad),
        cell_metrics=metrics,
        summary={
            "status": "complete",
            "domain_disposition": "global_earth",
            "analysis_id": analysis_id,
            "grid_identity": "healpix_nest_wgs84_authalic_v1",
            "order": order,
            "sample_count": 3,
            "horizon_start_s": 0.0,
            "horizon_end_s": 20.0,
        },
        interval_semantic_sha256=("1" if analysis_id == "member_a" else "2") * 64,
    )


def test_constellation_aggregation_has_exact_union_overlap_and_failed_cells(tmp_path) -> None:
    cells = healpix_npix(5)
    mask_a = np.zeros((3, cells), dtype=bool)
    mask_b = np.zeros((3, cells), dtype=bool)
    mask_a[:, 0] = True
    mask_a[0, 1] = True
    mask_b[:, 0] = True
    mask_b[1:, 2] = True
    member_a = _synthetic_coverage("member_a", mask_a)
    member_b = _synthetic_coverage("member_b", mask_b)
    union = evaluate_constellation_coverage(
        ConstellationCoverageConfig(
            analysis_id="constellation_union",
            member_analysis_ids=("member_b", "member_a"),
            order=5,
            service_definition_id="synthetic_geometric_coverage_v1",
        ),
        [member_b, member_a],
    )
    np.testing.assert_array_equal(union.covered_cell_count, [2, 2, 2])
    assert union.max_multiplicity_per_cell[0] == 2
    assert union.summary["never_service_qualified_cell_count"] == cells - 3
    double = evaluate_constellation_coverage(
        ConstellationCoverageConfig(
            analysis_id="constellation_double",
            member_analysis_ids=("member_a", "member_b"),
            order=5,
            service_definition_id="synthetic_geometric_coverage_v1",
            required_multiplicity=2,
        ),
        [member_a, member_b],
    )
    np.testing.assert_array_equal(double.covered_cell_count, [1, 1, 1])
    artifacts = write_constellation_coverage_artifacts(union, tmp_path / "constellation")
    assert artifacts.manifest_json.is_file()

    communications_member = SimpleNamespace(
        **{
            **vars(member_b),
            "summary": {
                "status": "complete",
                "domain_disposition": "global_earth_communications_service",
                "analysis_id": "member_b",
                "grid_identity": "healpix_nest_wgs84_authalic_v1",
                "order": 5,
                "sample_count": 3,
                "horizon_start_s": 0.0,
                "horizon_end_s": 20.0,
                "service_id": "notional_link",
            },
        }
    )
    with pytest.raises(ValueError, match="one coverage domain disposition"):
        evaluate_constellation_coverage(
            ConstellationCoverageConfig(
                analysis_id="invalid_mixed_service",
                member_analysis_ids=("member_a", "member_b"),
                order=5,
                service_definition_id="invalid_mixed_service",
            ),
            [member_a, communications_member],
        )


def _opportunity(
    opportunity_id: str,
    kind: str,
    start_s: float,
    end_s: float,
    value: float,
    storage_delta: float,
    pointing: tuple[float, float, float],
) -> TaskOpportunity:
    return TaskOpportunity(
        opportunity_id=opportunity_id,
        source_product_sha256="a" * 64,
        asset_id="spacecraft",
        kind=kind,
        start_s=start_s,
        end_s=end_s,
        objective_value=value,
        storage_delta_bytes=storage_delta,
        energy_cost_wh=2.0,
        pointing_unit_eci=pointing,
    )


def test_tasking_exactly_respects_slew_duty_storage_power_and_source_binding(tmp_path) -> None:
    config = CoverageTaskingConfig(
        analysis_id="tasking_fixture",
        asset_id="spacecraft",
        constraints=TaskingConstraints(
            horizon_start_s=0.0,
            horizon_end_s=100.0,
            maximum_slew_rate_rad_s=np.pi / 100.0,
            settling_time_s=0.0,
            maximum_payload_duty_cycle=0.2,
            storage_capacity_bytes=100.0,
            initial_storage_bytes=0.0,
            energy_budget_wh=4.0,
        ),
    )
    opportunities = [
        _opportunity("obs_a", "observation", 0.0, 10.0, 10.0, 100.0, (1.0, 0.0, 0.0)),
        _opportunity("obs_b", "observation", 11.0, 20.0, 12.0, 100.0, (-1.0, 0.0, 0.0)),
        _opportunity("downlink", "downlink", 21.0, 30.0, 2.0, -100.0, (-1.0, 0.0, 0.0)),
    ]
    result = optimize_coverage_tasking(config, reversed(opportunities))
    assert result.selected_opportunity_ids == ("obs_b", "downlink")
    assert result.objective_value == pytest.approx(14.0)
    assert result.final_storage_bytes == pytest.approx(0.0)
    assert result.energy_used_wh == pytest.approx(4.0)
    assert result.rejected_opportunity_reasons["obs_a"].startswith("slew_or_settling_conflict")
    artifacts = write_coverage_tasking_artifacts(result, tmp_path / "tasking")
    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    assert manifest["schedule_semantic_sha256"] == result.schedule_semantic_sha256
    with pytest.raises(ValueError, match="does not belong to asset"):
        optimize_coverage_tasking(
            config,
            [replace(opportunities[0], asset_id="different_spacecraft")],
        )


def test_link_artifacts_include_auditable_ledger_windows_packet_and_plot(tmp_path) -> None:
    result = evaluate_directed_link(
        _link_config(
            _identity_terminal("tx.constant", "tx"),
            _identity_terminal("rx.constant", "rx"),
        ),
        tx_history=_spacecraft_history("tx", np.array([7000.0, 0.0, 0.0])),
        rx_history=_spacecraft_history("rx", np.array([7000.0, 1000.0, 0.0])),
        frame_context=FrameContext(jd_utc_start=2451545.0),
    )
    artifacts = write_directed_link_artifacts(
        result,
        tmp_path / "link",
        include_margin_plot=True,
    )
    assert artifacts.margin_plot_png is not None and artifacts.margin_plot_png.stat().st_size > 1000
    packet = json.loads(artifacts.evidence_packet_json.read_text(encoding="utf-8"))
    assert packet["semantic_sha256"] == result.semantic_sha256
    assert packet["citations"][0]["rows"] == 3
    assert result.windows[0].estimated_delivered_data_bits == pytest.approx(20.0e6)
    assert result.summary["estimated_delivered_data_bits"] == pytest.approx(20.0e6)


def test_validation_fails_closed_for_ambiguous_profiles_resources_and_opportunities() -> None:
    with pytest.raises(ValueError, match="terminal profile"):
        CommunicationsCoverageConfig(
            **{
                **_communications_config().__dict__,
                "earth_terminal_profile": None,
            }
        )
    with pytest.raises(ValueError, match="maximum_candidates"):
        TaskingConstraints(
            horizon_start_s=0.0,
            horizon_end_s=1.0,
            maximum_slew_rate_rad_s=None,
            settling_time_s=0.0,
            maximum_payload_duty_cycle=1.0,
            storage_capacity_bytes=0.0,
            initial_storage_bytes=0.0,
            energy_budget_wh=0.0,
            maximum_candidates=25,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        TaskOpportunity(
            opportunity_id="bad",
            source_product_sha256="not-a-hash",
            asset_id="spacecraft",
            kind="other",
            start_s=0.0,
            end_s=1.0,
            objective_value=1.0,
            storage_delta_bytes=0.0,
            energy_cost_wh=0.0,
        )
    with pytest.raises(ValueError, match="outside WGS84"):
        evaluate_communications_coverage(
            _communications_config(),
            times_s=np.array([0.0, 1.0]),
            positions_eci_km=np.zeros((2, 3)),
            attitudes_quat_bn=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
            frame_context=FrameContext(jd_utc_start=2451545.0),
        )

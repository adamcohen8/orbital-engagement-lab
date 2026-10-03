from __future__ import annotations

import numpy as np

from sim.flight_software import (
    FrameId,
    GnssOwnStateMeasurement,
    IdealOwnStateMeasurement,
    IdealTrackedObjectStateMeasurement,
    InputEvent,
    InputKind,
    MeasurementEvent,
    PacketId,
    Quality,
)
from sim.gnc.attitude_v2 import SensorMounting
from sim.gnc.navigation_v2 import NavigationInitializationMode, OrbitFilterKind, OrbitNavigator
from sim.tests.fsw_v2_helpers import BODY_FRAME, INERTIAL_FRAME, clock, ideal_event
from sim.tests.fsw_v2_orbit_helpers import RELATIVE_FRAME, fault_event, relative_event


def _ideal_target_event(
    *, position_m: tuple[float, float, float], velocity_m_s: tuple[float, float, float]
) -> InputEvent:
    time = clock(1)
    payload = IdealTrackedObjectStateMeasurement("target", position_m, velocity_m_s)
    measurement = MeasurementEvent("ideal_track/target", payload.schema, time, INERTIAL_FRAME, payload)
    return InputEvent(PacketId("ideal-track", "boot", 0), InputKind.MEASUREMENT, time, time, Quality(), measurement)


def _navigator() -> OrbitNavigator:
    return OrbitNavigator(
        initialization=NavigationInitializationMode.IDEAL,
        body_frame=BODY_FRAME,
        inertial_frame=INERTIAL_FRAME,
        relative_frame=RELATIVE_FRAME,
    )


def _gnss_event(sequence: int, tick: int) -> InputEvent:
    time = clock(tick)
    payload = GnssOwnStateMeasurement((7_000_000.0, 0.0, 0.0), (0.0, 7_500.0, 0.0))
    measurement = MeasurementEvent("gnss", payload.schema, time, INERTIAL_FRAME, payload)
    return InputEvent(PacketId("gnss", "boot", sequence), InputKind.MEASUREMENT, time, time, Quality(), measurement)


def test_range_rate_los_and_angular_rate_reconstruct_typed_relative_state() -> None:
    navigator = _navigator()
    navigator.ingest(
        (
            ideal_event(0, 1),
            relative_event(
                0,
                1,
                range_m=1_000.0,
                range_rate_m_s=-2.0,
                los=(1.0, 0.0, 0.0),
                angular_rate_rad_s=(0.0, 0.0, 0.001),
            ),
        )
    )
    track = navigator.solution(clock(1)).relative_track("target")
    assert track is not None
    np.testing.assert_allclose(track.position_m, (1_000.0, 0.0, 0.0))
    np.testing.assert_allclose(track.velocity_m_s, (-2.0, 1.0, 0.0))
    assert track.frame == RELATIVE_FRAME


def test_sensor_fault_and_out_of_order_packets_do_not_overwrite_last_track() -> None:
    navigator = _navigator()
    navigator.ingest((ideal_event(0, 1), relative_event(0, 2, range_m=500.0)))
    navigator.ingest(
        (
            fault_event(0, 3, "relative"),
            relative_event(1, 3, range_m=10.0),
        )
    )
    assert navigator.solution(clock(3)).relative_track("target").range_m == 500.0  # type: ignore[union-attr]

    navigator.ingest((fault_event(1, 4, "relative", active=False), relative_event(2, 1, range_m=20.0)))
    assert navigator.solution(clock(4)).relative_track("target").range_m == 500.0  # type: ignore[union-attr]


def test_delayed_fresh_ekf_measurement_is_rejected_without_raising() -> None:
    navigator = OrbitNavigator(
        initialization=NavigationInitializationMode.IDEAL,
        body_frame=BODY_FRAME,
        inertial_frame=INERTIAL_FRAME,
        relative_frame=RELATIVE_FRAME,
        filter_kind=OrbitFilterKind.EKF,
    )
    navigator.ingest((ideal_event(0, 0),))
    propagated = navigator.solution(clock(10))

    navigator.ingest((_gnss_event(1, 5),))
    solution = navigator.solution(clock(10))

    assert solution.own_state_epoch == propagated.own_state_epoch
    assert solution.own_state_valid
    assert solution.belief.own_state is not None
    assert solution.belief.own_state.validity.value == "degraded"


def test_delayed_relative_ekf_measurement_is_rejected_without_raising() -> None:
    navigator = OrbitNavigator(
        initialization=NavigationInitializationMode.IDEAL,
        body_frame=BODY_FRAME,
        inertial_frame=INERTIAL_FRAME,
        relative_frame=RELATIVE_FRAME,
        filter_kind=OrbitFilterKind.EKF,
    )
    navigator.ingest((ideal_event(0, 0), relative_event(0, 0)))
    navigator.solution(clock(10))

    navigator.ingest((relative_event(1, 5, sensor_id="relative-delayed"),))
    track = navigator.solution(clock(10)).relative_track("target")

    assert track is not None
    assert track.epoch == clock(10)


def test_partial_ideal_own_state_does_not_refresh_orbit_epoch() -> None:
    navigator = _navigator()
    navigator.ingest((ideal_event(0, 0),))
    before = navigator.solution(clock(40))
    time = clock(40)
    payload = IdealOwnStateMeasurement(attitude_quat_body_from_inertial=(1.0, 0.0, 0.0, 0.0))
    measurement = MeasurementEvent("attitude-only", payload.schema, time, BODY_FRAME, payload)
    event = InputEvent(PacketId("attitude-only", "boot", 0), InputKind.MEASUREMENT, time, time, Quality(), measurement)

    navigator.ingest((event,))
    after = navigator.solution(clock(40))

    assert after.own_state_epoch == before.own_state_epoch == clock(0)
    assert after.position_eci_m == before.position_eci_m
    assert after.velocity_eci_m_s == before.velocity_eci_m_s


def test_relative_sensor_frame_is_transformed_with_onboard_believed_mounting() -> None:
    sensor_frame = FrameId("OEL/SENSOR/sat/relative", "frames-v1")
    navigator = OrbitNavigator(
        initialization=NavigationInitializationMode.IDEAL,
        body_frame=BODY_FRAME,
        inertial_frame=INERTIAL_FRAME,
        relative_frame=RELATIVE_FRAME,
        sensor_mountings=(
            SensorMounting(
                "relative",
                (2**-0.5, 0.0, 0.0, 2**-0.5),
                sensor_frame,
            ),
        ),
    )
    navigator.ingest(
        (
            ideal_event(0, 1),
            relative_event(0, 1, range_m=1_000.0, frame=sensor_frame),
        )
    )
    track = navigator.solution(clock(1)).relative_track("target")
    assert track is not None
    np.testing.assert_allclose(track.position_m, (0.0, -1_000.0, 0.0), atol=1.0e-10)
    assert track.frame == RELATIVE_FRAME


def test_ideal_track_is_own_deputy_relative_to_target_chief_in_target_ric() -> None:
    navigator = _navigator()
    navigator.ingest(
        (
            ideal_event(
                0,
                1,
                position_m=(7_000_100.0, 0.0, 0.0),
                velocity_m_s=(0.0, 7_500.0, 0.0),
            ),
            _ideal_target_event(
                position_m=(7_000_000.0, 0.0, 0.0),
                velocity_m_s=(0.0, 7_500.0, 0.0),
            ),
        )
    )
    track = navigator.solution(clock(1)).relative_track("target")
    assert track is not None
    assert track.frame == RELATIVE_FRAME
    np.testing.assert_allclose(track.position_m, (100.0, 0.0, 0.0), atol=1.0e-10)
    assert track.chief_position_eci_m == (7_000_000.0, 0.0, 0.0)


def test_alpha_beta_filter_smooths_relative_measurement_updates_onboard() -> None:
    navigator = OrbitNavigator(
        initialization=NavigationInitializationMode.IDEAL,
        body_frame=BODY_FRAME,
        inertial_frame=INERTIAL_FRAME,
        relative_frame=RELATIVE_FRAME,
        filter_kind=OrbitFilterKind.ALPHA_BETA,
        alpha=0.5,
        beta=0.0,
    )
    navigator.ingest((ideal_event(0, 1), relative_event(0, 1, range_m=1_000.0)))
    navigator.ingest((ideal_event(1, 2), relative_event(1, 2, range_m=1_200.0)))
    track = navigator.solution(clock(2)).relative_track("target")
    assert track is not None
    assert track.position_m[0] == 1_100.0

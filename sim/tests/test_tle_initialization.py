from __future__ import annotations

import numpy as np
import pytest

from sim import SimulationConfig, SimulationSession
from sim.dynamics.orbit.frames import teme_to_eci_matrix_vallado_iau80
from sim.dynamics.orbit.ogp import (
    ogp_propagate_teme,
    ogp_propagate_teme_batch_accelerated,
    ogp_propagate_teme_batch_reference,
    ogp_propagator_name_for_elements,
    ogp_regime_for_elements,
)
from sim.dynamics.orbit.sdp4 import sdp4_initialize, sdp4_propagate_teme, sdp4_propagate_teme_from_context
from sim.dynamics.orbit.sgp4 import (
    SGP4EphemerisProvider,
    sgp4_orbital_period_min,
    sgp4_propagate_teme,
    sgp4_propagate_teme_batch_numba,
    sgp4_propagate_teme_batch_reference,
)
from sim.dynamics.orbit.tle import parse_tle_lines, tle_to_rv_eci

ISS_LINE1 = "1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9005"
ISS_LINE2 = "2 25544  51.6416  43.6012 0005423  52.3066  50.1234 15.50000000  1000"
DEEP_SPACE_LINE1 = "1 90003U 24003A   24001.00000000  .00000000  00000+0  00000+0 0    10"
DEEP_SPACE_LINE2 = "2 90003  10.0000  20.0000 0100000  30.0000  40.0000  4.00000000    10"


def _tle_config(*, initial_jd_utc: float | None = None) -> dict:
    simulator = {
        "duration_s": 1.0,
        "dt_s": 1.0,
        "termination": {"earth_impact_enabled": False},
        "dynamics": {"attitude": {"enabled": False}, "orbit": {"model": "two_body"}},
    }
    if initial_jd_utc is not None:
        simulator["initial_jd_utc"] = initial_jd_utc
    return {
        "scenario_name": "tle_initialization",
        "rocket": {"enabled": False},
        "chaser": {"enabled": False},
        "target": {
            "enabled": True,
            "specs": {"mass_kg": 420.0},
            "initial_state": {
                "tle": {
                    "line1": ISS_LINE1,
                    "line2": ISS_LINE2,
                }
            },
        },
        "simulator": simulator,
        "outputs": {
            "mode": "save",
            "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def _sgp4_config(
    *,
    initial_jd_utc: float | None = None,
    output_frame: str = "eci",
    frame_transform: str | None = None,
) -> dict:
    cfg = _tle_config(initial_jd_utc=initial_jd_utc)
    cfg["target"]["propagation_method"] = "general"
    cfg["target"]["general"] = {"model": "sgp4", "output_frame": output_frame}
    if frame_transform is not None:
        cfg["target"]["general"]["frame_transform"] = frame_transform
    elif output_frame == "eci":
        cfg["target"]["general"]["frame_transform"] = "teme_as_eci"
    cfg["outputs"]["review"] = {"enabled": True, "detail": "standard"}
    return cfg


def test_tle_parser_converts_mean_elements_to_eci_state() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    pos, vel = tle_to_rv_eci(elements)

    assert elements.epoch_jd_utc == 2460310.5
    assert elements.norad_number == "25544"
    assert elements.bstar == 1.027e-4
    assert elements.ephemeris_type == "0"
    assert elements.element_number == 900
    assert elements.revolution_number == 100
    assert np.linalg.norm(pos) > 6500.0
    assert 7.0 < np.linalg.norm(vel) < 8.0


def test_sgp4_propagates_tle_to_teme_state() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    state0 = sgp4_propagate_teme(elements, 0.0)
    state60 = sgp4_propagate_teme(elements, 60.0)
    state2 = sgp4_propagate_teme(elements, 2.0)

    assert state0.error is None
    assert state60.error is None
    assert np.linalg.norm(state0.position_teme_km) > 6500.0
    assert 7.0 < np.linalg.norm(state0.velocity_teme_km_s) < 8.0
    assert np.linalg.norm(state60.position_teme_km - state0.position_teme_km) > 1000.0
    np.testing.assert_allclose(
        state2.position_teme_km,
        [-4428.102235537942, 1241.362812552974, 4990.602843321561],
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        state2.velocity_teme_km_s,
        [-4.077218024018575, -6.150810972865114, -2.077826973779405],
        rtol=0.0,
        atol=1e-12,
    )


def test_sgp4_rejects_deep_space_sdp4_tle_boundary() -> None:
    elements = parse_tle_lines(DEEP_SPACE_LINE1, DEEP_SPACE_LINE2, require_checksum=True)

    state = sgp4_propagate_teme(elements, 0.0)

    assert sgp4_orbital_period_min(elements) == 360.0
    assert state.error is not None
    assert "deep-space SDP4/resonance TLEs" in state.error


def test_ogp_dispatches_near_earth_and_deep_space_regimes() -> None:
    near_earth = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    deep_space = parse_tle_lines(DEEP_SPACE_LINE1, DEEP_SPACE_LINE2, require_checksum=True)

    assert ogp_regime_for_elements(near_earth) == "sgp4"
    assert ogp_propagator_name_for_elements(near_earth) == "OGP-SGP4"
    assert ogp_propagate_teme(near_earth, 0.0).error is None

    assert ogp_regime_for_elements(deep_space) == "sdp4"
    assert ogp_propagator_name_for_elements(deep_space) == "OGP-SDP4"
    deep_state = ogp_propagate_teme(deep_space, 0.0)
    assert deep_state.error is None
    assert np.linalg.norm(deep_state.position_teme_km) > 10000.0


def test_ogp_batch_reference_supports_mixed_sgp4_sdp4_regimes() -> None:
    near_earth = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    deep_space = parse_tle_lines(DEEP_SPACE_LINE1, DEEP_SPACE_LINE2, require_checksum=True)
    offsets_min = np.array([0.0, 60.0], dtype=float)

    batch = ogp_propagate_teme_batch_reference([near_earth, deep_space], offsets_min)

    assert batch.backend == "ogp_scalar_reference"
    assert batch.object_count == 2
    assert batch.sample_count == 2
    assert np.all(batch.success)
    np.testing.assert_allclose(batch.tsince_min[0], offsets_min)
    np.testing.assert_allclose(batch.tsince_min[1], offsets_min)
    near_scalar = ogp_propagate_teme(near_earth, 60.0)
    deep_scalar = ogp_propagate_teme(deep_space, 60.0)
    np.testing.assert_allclose(batch.position_teme_km[0, 1], near_scalar.position_teme_km, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(batch.position_teme_km[1, 1], deep_scalar.position_teme_km, rtol=0.0, atol=1e-12)


def test_ogp_batch_accelerated_matches_reference_for_mixed_regimes() -> None:
    near_earth = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    deep_space = parse_tle_lines(DEEP_SPACE_LINE1, DEEP_SPACE_LINE2, require_checksum=True)
    offsets_min = np.array([[0.0, 2.0, 60.0], [0.0, 60.0, 1440.0]], dtype=float)

    reference = ogp_propagate_teme_batch_reference([near_earth, deep_space], offsets_min)
    accelerated = ogp_propagate_teme_batch_accelerated([near_earth, deep_space], offsets_min)

    assert accelerated.backend.startswith("ogp_mixed")
    assert np.all(accelerated.success)
    np.testing.assert_allclose(accelerated.tsince_min, reference.tsince_min)
    np.testing.assert_allclose(accelerated.position_teme_km, reference.position_teme_km, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(accelerated.velocity_teme_km_s, reference.velocity_teme_km_s, rtol=0.0, atol=1e-12)


def test_sdp4_context_matches_scalar_and_supports_nonmonotonic_calls() -> None:
    deep_space = parse_tle_lines(DEEP_SPACE_LINE1, DEEP_SPACE_LINE2, require_checksum=True)
    context = sdp4_initialize(deep_space)

    assert context.period_min == pytest.approx(360.0)
    for offset_min in [1440.0, 0.0, 60.0, 720.0]:
        contextual = sdp4_propagate_teme_from_context(context, offset_min)
        scalar = sdp4_propagate_teme(deep_space, offset_min)
        assert contextual.error is None
        assert scalar.error is None
        np.testing.assert_allclose(contextual.position_teme_km, scalar.position_teme_km, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(contextual.velocity_teme_km_s, scalar.velocity_teme_km_s, rtol=0.0, atol=1e-12)


def test_sgp4_batch_reference_matches_scalar_common_time_grid() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    offsets_min = np.array([0.0, 2.0, 60.0], dtype=float)

    batch = sgp4_propagate_teme_batch_reference([elements, elements], offsets_min)

    assert batch.backend == "scalar_reference"
    assert batch.object_count == 2
    assert batch.sample_count == 3
    assert batch.position_teme_km.shape == (2, 3, 3)
    assert batch.velocity_teme_km_s.shape == (2, 3, 3)
    assert batch.errors.shape == (2, 3)
    assert np.all(batch.success)
    np.testing.assert_allclose(batch.tsince_min[0], offsets_min)
    np.testing.assert_allclose(batch.tsince_min[1], offsets_min)
    for object_index in range(2):
        for sample_index, offset_min in enumerate(offsets_min):
            scalar = sgp4_propagate_teme(elements, float(offset_min))
            assert scalar.error is None
            np.testing.assert_allclose(
                batch.position_teme_km[object_index, sample_index],
                scalar.position_teme_km,
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                batch.velocity_teme_km_s[object_index, sample_index],
                scalar.velocity_teme_km_s,
                rtol=0.0,
                atol=0.0,
            )


def test_sgp4_batch_reference_supports_per_object_time_grid() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    offsets_min = np.array([[0.0, 2.0], [30.0, 60.0]], dtype=float)

    batch = sgp4_propagate_teme_batch_reference([elements, elements], offsets_min)

    assert batch.position_teme_km.shape == (2, 2, 3)
    np.testing.assert_allclose(batch.tsince_min, offsets_min)
    for object_index in range(2):
        for sample_index in range(2):
            scalar = sgp4_propagate_teme(elements, float(offsets_min[object_index, sample_index]))
            assert scalar.error is None
            np.testing.assert_allclose(
                batch.position_teme_km[object_index, sample_index],
                scalar.position_teme_km,
                rtol=0.0,
                atol=0.0,
            )


def test_sgp4_batch_reference_records_per_sample_errors() -> None:
    near_earth = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    deep_space = parse_tle_lines(DEEP_SPACE_LINE1, DEEP_SPACE_LINE2, require_checksum=True)

    batch = sgp4_propagate_teme_batch_reference([near_earth, deep_space], [0.0, 10.0])

    assert np.all(batch.success[0])
    assert not np.any(batch.success[1])
    assert batch.errors[0, 0] == ""
    assert "deep-space SDP4/resonance TLEs" in batch.errors[1, 0]
    assert "deep-space SDP4/resonance TLEs" in batch.errors[1, 1]
    np.testing.assert_allclose(batch.position_teme_km[1], np.zeros((2, 3)))
    np.testing.assert_allclose(batch.velocity_teme_km_s[1], np.zeros((2, 3)))


def test_sgp4_batch_reference_validates_shape_and_values() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)

    with pytest.raises(ValueError, match="at least one element"):
        sgp4_propagate_teme_batch_reference([], [0.0])
    with pytest.raises(ValueError, match="at least one time sample"):
        sgp4_propagate_teme_batch_reference([elements], [])
    with pytest.raises(ValueError, match="per-object time grid"):
        sgp4_propagate_teme_batch_reference([elements, elements], np.zeros((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        sgp4_propagate_teme_batch_reference([elements], [0.0, np.nan])


def test_sgp4_batch_numba_matches_scalar_reference_common_time_grid() -> None:
    pytest.importorskip("numba")
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    offsets_min = np.array([0.0, 2.0, 60.0, 1440.0], dtype=float)

    reference = sgp4_propagate_teme_batch_reference([elements, elements], offsets_min)
    accelerated = sgp4_propagate_teme_batch_numba([elements, elements], offsets_min)

    assert accelerated.backend == "numba_cpu"
    assert accelerated.object_count == 2
    assert accelerated.sample_count == 4
    assert np.all(accelerated.success)
    np.testing.assert_allclose(accelerated.tsince_min, reference.tsince_min, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(accelerated.position_teme_km, reference.position_teme_km, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(accelerated.velocity_teme_km_s, reference.velocity_teme_km_s, rtol=0.0, atol=1e-12)


def test_sgp4_batch_numba_matches_scalar_reference_per_object_time_grid() -> None:
    pytest.importorskip("numba")
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    offsets_min = np.array([[0.0, 2.0, 30.0], [15.0, 60.0, 120.0]], dtype=float)

    reference = sgp4_propagate_teme_batch_reference([elements, elements], offsets_min)
    accelerated = sgp4_propagate_teme_batch_numba([elements, elements], offsets_min)

    np.testing.assert_allclose(accelerated.position_teme_km, reference.position_teme_km, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(accelerated.velocity_teme_km_s, reference.velocity_teme_km_s, rtol=0.0, atol=1e-12)


def test_sgp4_batch_numba_records_per_sample_errors() -> None:
    pytest.importorskip("numba")
    near_earth = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    deep_space = parse_tle_lines(DEEP_SPACE_LINE1, DEEP_SPACE_LINE2, require_checksum=True)

    accelerated = sgp4_propagate_teme_batch_numba([near_earth, deep_space], [0.0, 10.0])

    assert np.all(accelerated.success[0])
    assert not np.any(accelerated.success[1])
    assert accelerated.errors[0, 0] == ""
    assert "deep-space SDP4/resonance TLEs" in accelerated.errors[1, 0]
    assert "deep-space SDP4/resonance TLEs" in accelerated.errors[1, 1]
    np.testing.assert_allclose(accelerated.position_teme_km[1], np.zeros((2, 3)))
    np.testing.assert_allclose(accelerated.velocity_teme_km_s[1], np.zeros((2, 3)))


def test_sgp4_provider_dispatches_deep_space_tle_to_ogp_sdp4() -> None:
    elements = parse_tle_lines(DEEP_SPACE_LINE1, DEEP_SPACE_LINE2, require_checksum=True)

    provider = SGP4EphemerisProvider.from_tle_block(
        {"line1": DEEP_SPACE_LINE1, "line2": DEEP_SPACE_LINE2, "require_checksum": True},
        mass_kg=420.0,
        start_jd_utc=elements.epoch_jd_utc,
        duration_s=7200.0,
        output_frame="teme",
    )
    metadata = provider.metadata()

    assert metadata.propagator_family == "OGP"
    assert metadata.propagator_name == "OGP-SDP4"
    assert metadata.output_frame == "teme"
    truth = provider.state_at(0.0)
    assert np.linalg.norm(truth.position_eci_km) > 10000.0


def test_satellite_initial_state_accepts_tle_lines() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    expected_pos, expected_vel = tle_to_rv_eci(elements)
    session = SimulationSession.from_config(SimulationConfig.from_dict(_tle_config()))

    result = session.run()
    truth0 = result.truth["target"][0]

    np.testing.assert_allclose(truth0[0:3], expected_pos, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(truth0[3:6], expected_vel, rtol=0.0, atol=1e-12)


def test_tle_initial_state_propagates_to_simulator_initial_jd() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    target_jd = elements.epoch_jd_utc + 0.25
    expected_pos, expected_vel = tle_to_rv_eci(elements, target_jd_utc=target_jd)
    session = SimulationSession.from_config(SimulationConfig.from_dict(_tle_config(initial_jd_utc=target_jd)))

    result = session.run()
    truth0 = result.truth["target"][0]

    np.testing.assert_allclose(truth0[0:3], expected_pos, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(truth0[3:6], expected_vel, rtol=0.0, atol=1e-12)


def test_general_sgp4_object_samples_truth_history() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    session = SimulationSession.from_config(SimulationConfig.from_dict(_sgp4_config(initial_jd_utc=elements.epoch_jd_utc)))

    result = session.run()
    truth0 = result.truth["target"][0]
    truth1 = result.truth["target"][1]
    expected0 = SGP4EphemerisProvider.from_tle_block(
        {"line1": ISS_LINE1, "line2": ISS_LINE2},
        mass_kg=420.0,
        start_jd_utc=elements.epoch_jd_utc,
        duration_s=1.0,
    ).state_at(0.0)

    np.testing.assert_allclose(truth0[0:3], expected0.position_eci_km, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(truth0[3:6], expected0.velocity_eci_km_s, rtol=0.0, atol=1e-12)
    assert np.linalg.norm(truth1[0:3] - truth0[0:3]) > 1.0
    assert result.payload["object_propagation"]["target"]["propagation_method"] == "general"
    assert result.payload["object_propagation"]["target"]["general_model"] == "sgp4"


def test_sgp4_provider_uses_stable_relative_time_arithmetic() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    provider = SGP4EphemerisProvider.from_tle_block(
        {"line1": ISS_LINE1, "line2": ISS_LINE2},
        mass_kg=420.0,
        start_jd_utc=elements.epoch_jd_utc,
        duration_s=7200.0,
    )
    truth = provider.state_at(120.0)
    direct = sgp4_propagate_teme(elements, 2.0)

    np.testing.assert_allclose(truth.position_eci_km, direct.position_teme_km, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(truth.velocity_eci_km_s, direct.velocity_teme_km_s, rtol=0.0, atol=1e-15)


def test_sgp4_provider_supports_explicit_native_teme_output() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    provider = SGP4EphemerisProvider.from_tle_block(
        {"line1": ISS_LINE1, "line2": ISS_LINE2},
        mass_kg=420.0,
        start_jd_utc=elements.epoch_jd_utc,
        duration_s=7200.0,
        output_frame="teme",
    )
    truth = provider.state_at(120.0)
    direct = sgp4_propagate_teme(elements, 2.0)
    metadata = provider.metadata()

    assert metadata.output_frame == "teme"
    assert metadata.frame_transform == "native"
    np.testing.assert_allclose(truth.position_eci_km, direct.position_teme_km, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(truth.velocity_eci_km_s, direct.velocity_teme_km_s, rtol=0.0, atol=1e-15)


def test_teme_to_eci_vallado_iau80_matrix_is_orthonormal_regression() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)

    rot = teme_to_eci_matrix_vallado_iau80(elements.epoch_jd_utc)

    np.testing.assert_allclose(rot @ rot.T, np.eye(3), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        rot,
        [
            [0.9999829035026804, 0.005366776183285927, 0.002321726889060252],
            [-0.005366866969314266, 0.9999855977253935, 0.0000328743170808624],
            [-0.002321517021810083, -0.00004533415439786221, 0.9999973042481325],
        ],
        rtol=0.0,
        atol=1e-15,
    )


def test_sgp4_provider_supports_vallado_iau80_eci_output() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    provider = SGP4EphemerisProvider.from_tle_block(
        {"line1": ISS_LINE1, "line2": ISS_LINE2},
        mass_kg=420.0,
        start_jd_utc=elements.epoch_jd_utc,
        duration_s=7200.0,
        output_frame="eci",
        frame_transform="teme_to_eci_iau80",
    )
    truth = provider.state_at(120.0)
    direct = sgp4_propagate_teme(elements, 2.0)
    metadata = provider.metadata()

    assert metadata.output_frame == "eci"
    assert metadata.frame_transform == "teme_to_eci_iau80"
    np.testing.assert_allclose(
        truth.position_eci_km,
        [-4409.777594653086, 1265.2740376753509, 5000.813029479821],
        rtol=0.0,
        atol=1e-12,
    )
    assert np.linalg.norm(truth.position_eci_km - direct.position_teme_km) > 30.0
    assert np.linalg.norm(truth.velocity_eci_km_s - direct.velocity_teme_km_s) > 0.04


def test_general_sgp4_teme_object_records_state_frame_metadata() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    session = SimulationSession.from_config(
        SimulationConfig.from_dict(_sgp4_config(initial_jd_utc=elements.epoch_jd_utc, output_frame="teme"))
    )

    result = session.run()

    assert result.payload["object_propagation"]["target"]["output_frame"] == "teme"
    assert result.payload["object_propagation"]["target"]["frame_transform"] == "native"
    assert result.payload["object_state_frames"]["target"] == "teme"


def test_general_sgp4_vallado_iau80_eci_object_records_state_frame_metadata() -> None:
    elements = parse_tle_lines(ISS_LINE1, ISS_LINE2)
    session = SimulationSession.from_config(
        SimulationConfig.from_dict(
            _sgp4_config(
                initial_jd_utc=elements.epoch_jd_utc,
                output_frame="eci",
                frame_transform="teme_to_eci_iau80",
            )
        )
    )

    result = session.run()

    assert result.payload["object_propagation"]["target"]["output_frame"] == "eci"
    assert result.payload["object_propagation"]["target"]["frame_transform"] == "teme_to_eci_iau80"
    assert result.payload["object_state_frames"]["target"] == "eci"

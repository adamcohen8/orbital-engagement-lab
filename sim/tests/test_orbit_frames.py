from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.config.scenario_yaml import scenario_config_from_dict
from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.frames import (
    FRAME_MODEL_IAU76_80_EOP,
    FRAME_MODEL_SIMPLE_GMST,
    eci_to_ecef_harmonic,
    eci_to_ecef_rotation_context,
    frame_context_from_mapping,
    normalize_frame_model,
    transform_position,
    transform_state,
)
from sim.dynamics.orbit.propagator import drag_plugin


def _write_minimal_eop(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "VERSION test",
                "NUM_OBSERVED_POINTS 2",
                "2024 01 01 60310.0 0.10 0.20 0.30 0 0 0 0 0 37",
                "2024 01 02 60311.0 0.11 0.21 0.31 0 0 0 0 0 37",
            ]
        ),
        encoding="utf-8",
    )


def test_frame_model_aliases_are_canonical() -> None:
    assert normalize_frame_model("simple") == FRAME_MODEL_SIMPLE_GMST
    assert normalize_frame_model("simple_earth_rotation") == FRAME_MODEL_SIMPLE_GMST
    assert normalize_frame_model("hpop_like") == FRAME_MODEL_IAU76_80_EOP
    assert normalize_frame_model("iau76_80_eop") == FRAME_MODEL_IAU76_80_EOP


def test_iau76_80_eop_alias_matches_legacy_hpop_like_rotation(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    r_eci = np.array([7000.0, 120.0, 30.0], dtype=float)

    legacy = eci_to_ecef_harmonic(
        r_eci,
        60.0,
        jd_utc_start=2460310.5,
        frame_model="hpop_like",
        eop_path=str(eop_path),
    )
    canonical = eci_to_ecef_harmonic(
        r_eci,
        60.0,
        jd_utc_start=2460310.5,
        frame_model="iau76_80_eop",
        eop_path=str(eop_path),
    )

    np.testing.assert_allclose(canonical, legacy, rtol=0.0, atol=0.0)


def test_frame_context_records_eop_time_scale_provenance(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    ctx = frame_context_from_mapping(
        {"model": "iau76_80_eop", "eop_path": str(eop_path)},
        jd_utc_start=2460310.5,
    )

    meta = ctx.metadata(sample_t_s=0.0)

    assert meta["model"] == "iau76_80_eop"
    assert meta["legacy_frame_model"] == "hpop_like"
    assert meta["time_scale_model"] == "eop_utc_ut1_tt"
    assert meta["eop_path"] == str(eop_path.resolve())
    assert meta["dut1_s"] == 0.30
    assert meta["tt_minus_utc_s"] == 69.184
    assert meta["polar_motion_applied"] is True


def test_manual_eop_frame_context_affects_rotation_without_eop_path() -> None:
    base = frame_context_from_mapping({"model": "iau76_80_eop"}, jd_utc_start=2460310.5)
    manual = frame_context_from_mapping(
        {
            "model": "iau76_80_eop",
            "dut1_s": 0.30,
            "xp_arcsec": 0.10,
            "yp_arcsec": 0.20,
            "dat_s": 37.0,
        },
        jd_utc_start=2460310.5,
    )
    r_eci = np.array([7000.0, 120.0, 30.0], dtype=float)

    base_ecef = transform_position(r_eci, "eci", "ecef", t_s=120.0, context=base)
    manual_ecef = transform_position(r_eci, "eci", "ecef", t_s=120.0, context=manual)

    assert manual.metadata()["polar_motion_applied"] is True
    assert np.linalg.norm(manual_ecef - base_ecef) > 1.0e-4


def test_eop_frame_context_without_epoch_does_not_claim_or_apply_eop(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    ctx = frame_context_from_mapping({"model": "iau76_80_eop", "eop_path": str(eop_path)})

    assert ctx.metadata()["polar_motion_applied"] is False
    with pytest.raises(ValueError, match="requires simulator.initial_jd_utc"):
        eci_to_ecef_rotation_context(0.0, ctx)


def test_scenario_frames_eop_path_requires_initial_epoch(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)

    with pytest.raises(ValueError, match="simulator.frames EOP settings require simulator.initial_jd_utc"):
        scenario_config_from_dict(
            {
                "scenario_name": "missing_frame_epoch",
                "simulator": {
                    "frames": {
                        "model": "iau76_80_eop",
                        "eop_path": str(eop_path),
                    }
                },
                "objects": {},
            }
        )


def test_scenario_frames_manual_eop_requires_initial_epoch() -> None:
    with pytest.raises(ValueError, match="simulator.frames EOP settings require simulator.initial_jd_utc"):
        scenario_config_from_dict(
            {
                "scenario_name": "missing_manual_frame_epoch",
                "simulator": {
                    "frames": {
                        "model": "iau76_80_eop",
                        "dut1_s": 0.2,
                    }
                },
                "objects": {},
            }
        )


def test_scenario_iau_frame_model_without_eop_does_not_require_initial_epoch() -> None:
    cfg = scenario_config_from_dict(
        {
            "scenario_name": "iau_frame_no_eop",
            "simulator": {"frames": {"model": "iau76_80_eop"}},
            "objects": {},
        }
    )

    assert cfg.simulator.frames.model == "iau76_80_eop"
    assert cfg.simulator.initial_jd_utc is None


def test_frame_context_nutation_corrections_affect_rotation_and_provenance(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    base = frame_context_from_mapping(
        {"model": "iau76_80_eop", "eop_path": str(eop_path)},
        jd_utc_start=2460310.5,
    )
    corrected = frame_context_from_mapping(
        {"model": "iau76_80_eop", "eop_path": str(eop_path), "ddpsi_rad": 1.0e-6, "ddeps_rad": -2.0e-6},
        jd_utc_start=2460310.5,
    )

    r_eci = np.array([7000.0, 120.0, 30.0], dtype=float)
    base_ecef = transform_position(r_eci, "eci", "ecef", t_s=120.0, context=base)
    corrected_ecef = transform_position(r_eci, "eci", "ecef", t_s=120.0, context=corrected)

    assert corrected.metadata()["nutation_corrections_applied"] is True
    assert np.linalg.norm(corrected_ecef - base_ecef) > 1.0e-4


def test_context_transform_matches_canonical_harmonic_helper(tmp_path: Path) -> None:
    eop_path = tmp_path / "EOP-All.txt"
    _write_minimal_eop(eop_path)
    ctx = frame_context_from_mapping(
        {"model": "iau76_80_eop", "eop_path": str(eop_path)},
        jd_utc_start=2460310.5,
    )
    r_eci = np.array([7000.0, 120.0, 30.0], dtype=float)

    from_context = transform_position(r_eci, "eci", "ecef", t_s=120.0, context=ctx)
    direct = eci_to_ecef_harmonic(
        r_eci,
        120.0,
        jd_utc_start=2460310.5,
        frame_model="iau76_80_eop",
        eop_path=str(eop_path),
    )

    np.testing.assert_allclose(from_context, direct, rtol=0.0, atol=0.0)


def test_transform_state_applies_rotating_frame_velocity_term() -> None:
    ctx = frame_context_from_mapping({"model": "simple_gmst"})
    r_eci = np.array([7000.0, 0.0, 0.0], dtype=float)
    v_eci = np.array([0.0, 7.5, 0.0], dtype=float)

    r_ecef, v_ecef = transform_state(r_eci, v_eci, "eci", "ecef", t_s=0.0, context=ctx)
    expected_v_ecef = np.array([0.0, 7.5 - EARTH_ROT_RATE_RAD_S * 7000.0, 0.0], dtype=float)
    roundtrip_r, roundtrip_v = transform_state(r_ecef, v_ecef, "ecef", "eci", t_s=0.0, context=ctx)

    np.testing.assert_allclose(r_ecef, r_eci, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(v_ecef, expected_v_ecef, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(roundtrip_r, r_eci, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(roundtrip_v, v_eci, rtol=0.0, atol=1e-9)


def test_manual_eop_fields_propagate_to_drag_relative_velocity() -> None:
    class _Ctx:
        mass_kg = 100.0
        cd = 2.2
        area_m2 = 1.0

    x_eci = np.array([7000.0, 120.0, 30.0, 0.0, 7.5, 0.0], dtype=float)
    base_env = {
        "density_kg_m3": 1.0e-12,
        "drag_frame_model": "hpop_like",
        "jd_utc_start": 2460310.5,
    }
    manual_env = {
        **base_env,
        "dut1_s": 30.0,
        "xp_arcsec": 100.0,
        "yp_arcsec": 200.0,
        "dat_s": 37.0,
        "ddpsi_rad": 1.0e-3,
        "ddeps_rad": -2.0e-3,
    }

    base_drag = drag_plugin(120.0, x_eci, base_env, _Ctx())
    manual_drag = drag_plugin(120.0, x_eci, manual_env, _Ctx())

    assert np.linalg.norm(manual_drag - base_drag) > 1.0e-14

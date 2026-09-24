import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.config import scenario_config_from_dict
from sim.config.scenario.models import GroundStationSection
from sim.dynamics.orbit.frames import frame_context_from_mapping, transform_state
from sim.estimation.ground_station_od import predict_ground_station_measurement
from sim.ground_segment import GroundSegment, SimulatedGroundSegment
from sim.ground_stations import evaluate_ground_station_measurements

JD = 2461303.5
FRAMES = frame_context_from_mapping({}, jd_utc_start=JD)
STATION = GroundStationSection(
    id="test",
    lat_deg=0.0,
    lon_deg=0.0,
    measurements={
        "enabled": True,
        "update_cadence_s": 1.0,
        "noise": {"range_sigma_km": 0.01, "angle_sigma_deg": 0.02, "range_rate_sigma_km_s": 0.00001},
        "seed": 4,
    },
)
R, V = transform_state(np.array([7000.0, 0, 0]), np.array([0.0, 7.0, 0.0]), "ecef", "eci", t_s=0, context=FRAMES)
X = np.concatenate([R, V])


def config(prior=True, **extra):
    return {
        "enabled": True,
        "priors": {
            "sat": {
                "state": X.tolist(),
                "covariance": np.diag([1.0, 1.0, 1.0, 0.001, 0.001, 0.001]).tolist(),
                "epoch_s": 0.0,
            }
        }
        if prior
        else {},
        **extra,
    }


def receiver(**extra):
    return GroundSegment(config(**extra), stations=[STATION], object_ids=["sat"], jd_utc_start=JD, frame_context=FRAMES)


def packet(t, arrival=None, kind="telemetry", data=None):
    return dict(
        station_id="test",
        object_id="sat",
        time_s=t,
        received_time_s=t if arrival is None else arrival,
        kind=kind,
        data={"temperature_k": 290.0} if data is None else data,
    )


def tracking(t, state=X, arrival=None):
    values = predict_ground_station_measurement(
        target_state_eci=state, station=asdict(STATION), t_s=t, jd_utc_start=JD, frame_context=FRAMES
    )
    components = ["azimuth_deg", "elevation_deg", "range_km", "range_rate_km_s"]
    return packet(
        t,
        arrival,
        "tracking",
        dict(
            time_s=t, components=components, vector=[values[k] for k in components], sigma=[0.02, 0.02, 0.01, 0.00001]
        ),
    )


def test_unknown_stays_unknown_without_prior():
    g = receiver(prior=False)
    s = g.advance(0, packets=[tracking(0)])
    assert s["objects"]["sat"]["orbit"] is None
    assert s["packets"][0]["disposition"] == "no_prior"
    held = g.advance(30)["objects"]["sat"]
    assert held["orbit"] is None
    assert held["tracking"]["test"]["measurement_time_s"] == 0
    assert held["tracking"]["test"]["age_s"] == 30


def test_outage_holds_telemetry_but_predicts_orbit_and_covariance():
    g = receiver(stale_after_s=5)
    first = g.advance(0, packets=[packet(0)])
    last = g.advance(30)
    a, b = first["objects"]["sat"], last["objects"]["sat"]
    assert b["telemetry"]["temperature_k"]["value"] == 290
    assert b["telemetry"]["temperature_k"]["age_s"] == 30
    assert b["telemetry"]["temperature_k"]["stale"]
    assert b["orbit"]["state"] != a["orbit"]["state"]
    assert np.trace(b["orbit"]["covariance"]) > np.trace(a["orbit"]["covariance"])
    assert b["orbit"]["epoch_s"] == 30
    assert b["orbit"]["posterior_epoch_s"] == 0


def test_off_grid_packet_is_received_on_first_processing_sample():
    g = receiver()
    first = g.advance(3.0, packets=[packet(0.0, arrival=2.4)])
    row = first["objects"]["sat"]["telemetry"]["temperature_k"]
    assert row["received_time_s"] == 2.4
    assert row["status"] == "received"
    assert g.advance(4.0)["objects"]["sat"]["telemetry"]["temperature_k"]["status"] == "held"


def test_delayed_tracking_matches_measurement_epoch_update():
    immediate, delayed = receiver(), receiver()
    b = immediate.estimator.predict(immediate.posterior["sat"], 2.0)
    p = tracking(2.0, b.state + np.array([0.02, 0.01, 0, 0, 0, 0]))
    immediate.advance(2.0, packets=[p])
    expected = immediate.advance(5.0)["objects"]["sat"]["orbit"]
    delayed.advance(4.0)
    p["received_time_s"] = 5.0
    result = delayed.advance(5.0, packets=[p])["objects"]["sat"]["orbit"]
    np.testing.assert_allclose(result["state"], expected["state"], atol=1e-10)
    np.testing.assert_allclose(result["covariance"], expected["covariance"], atol=1e-10)
    assert result["posterior_epoch_s"] == 2.0


def test_old_tracking_rejected_and_old_telemetry_cannot_replace_new():
    g = receiver()
    g.advance(2.0, packets=[tracking(2.0), packet(2.0, data={"temperature_k": 300.0})])
    s = g.advance(3.0, packets=[tracking(1.0, arrival=3.0), packet(1.0, 3.0)])
    assert all(p["disposition"] != "accepted" for p in s["packets"])
    assert s["objects"]["sat"]["telemetry"]["temperature_k"]["value"] == 300.0


def test_invalid_packet_does_not_advance_time():
    g = receiver()
    with pytest.raises(ValueError):
        g.advance(0.0, packets=[packet(2.0)])
    assert g.advance(0.0)["time_s"] == 0


def test_snapshot_copy_isolation():
    g = receiver()
    s = g.advance(0.0, packets=[packet(0.0)])
    s["objects"]["sat"]["orbit"]["state"][0] = 0
    s["objects"]["sat"]["telemetry"]["temperature_k"]["value"] = 0
    assert g.snapshot()["objects"]["sat"]["telemetry"]["temperature_k"]["value"] == 290
    assert g.snapshot()["objects"]["sat"]["orbit"]["state"][0] != 0


def test_streamed_noise_matches_batch():
    kwargs = dict(ground_stations=[STATION], jd_utc_start=JD, frame_context=FRAMES)
    batch = evaluate_ground_station_measurements(**kwargs, t_s=np.arange(3.0), truth_hist={"sat": np.tile(X, (3, 1))})[
        "test"
    ]["targets"]["sat"]["measurements"]
    streams = {}
    rows = []
    for t in range(3):
        rows += evaluate_ground_station_measurements(
            **kwargs, t_s=np.array([t]), truth_hist={"sat": X.reshape(1, -1)}, stream_state=streams
        )["test"]["targets"]["sat"]["measurements"]
    assert rows == batch
    assert len(rows) == 3
    assert rows[0]["vector"] != rows[1]["vector"]


def test_contact_gates_and_latency_do_not_leak_truth():
    outages = [{"station_id": "test", "start_s": 1.0, "end_s": 4.0, "services": ["downlink"]}]
    adapter = SimulatedGroundSegment(
        config(latency_s=2.0, outages=outages),
        stations=[STATION],
        object_ids=["sat"],
        jd_utc_start=JD,
        frame_context=FRAMES,
    )
    for t in range(4):
        s = adapter.step(float(t), truth={"sat": X}, resources={"sat": {"temperature_k": 290.0 + t}})
        assert s["contacts"]["test"]["sat"]["tracking"]
        assert s["contacts"]["test"]["sat"]["downlink"] == (t == 0)
        if t < 2:
            assert not s["objects"]["sat"]["telemetry"]
        else:
            row = s["objects"]["sat"]["telemetry"]["temperature_k"]
            assert row["value"] == 290.0
            assert row["received_time_s"] == 2.0
    assert "truth_" not in json.dumps(adapter.ground.history)


def test_loss_of_sight_gates_all_services():
    adapter = SimulatedGroundSegment(
        config(), stations=[STATION], object_ids=["sat"], jd_utc_start=JD, frame_context=FRAMES
    )
    s = adapter.step(0.0, truth={"sat": -X}, resources={"sat": {"temperature_k": 290.0}})
    assert not any(s["contacts"]["test"]["sat"][k] for k in ("tracking", "downlink", "uplink"))
    assert not s["packets"]


@pytest.mark.parametrize(
    "bad",
    [
        {"cadence_s": 0},
        {"latency_s": -1},
        {"enabled": "yes"},
        {"process_noise_diag": [-1] * 6},
        {"oops": 1},
        {"priors": {"sat": {}}},
    ],
)
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        receiver(**bad)


def scenario(tmp_path):
    doc = yaml.safe_load(Path("configs/spacecraft_resources_demo.yaml").read_text())
    doc["objects"]["sat"] = doc["objects"].pop("spacecraft")
    doc["objects"]["sat"]["initial_state"]["position_eci_km"] = X[:3].tolist()
    doc["objects"]["sat"]["initial_state"]["velocity_eci_km_s"] = X[3:].tolist()
    doc["simulator"].update(duration_s=5.0, dt_s=1.0)
    doc["simulator"]["dynamics"]["orbit"]["orbit_substep_s"] = 1.0
    doc["ground_stations"] = [asdict(STATION)]
    doc["ground_segment"] = config()
    doc["outputs"]["output_dir"] = str(tmp_path)
    return doc


def test_headless_api_and_review_evidence(tmp_path):
    from sim.api import SimulationConfig, SimulationSession

    doc = scenario(tmp_path)
    result = SimulationSession(SimulationConfig.from_dict(doc)).run()
    snap = result.snapshot(5)
    assert snap.ground_segment["time_s"] == 5.0
    assert snap.ground_segment["objects"]["sat"]["telemetry"]
    assert (tmp_path / "ground_segment.json").exists()
    with sqlite3.connect(tmp_path / "review/run.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM ground_segment_state").fetchone()[0] == 6
        assert conn.execute("SELECT COUNT(*) FROM ground_segment_packets").fetchone()[0] >= 6


def test_config_cadence_and_unknown_references(tmp_path):
    doc = scenario(tmp_path)
    doc["ground_segment"]["cadence_s"] = 0.5
    with pytest.raises(ValueError, match="cadences"):
        scenario_config_from_dict(doc)
    doc["ground_segment"] = config()
    doc["ground_segment"]["priors"]["absent"] = doc["ground_segment"]["priors"].pop("sat")
    with pytest.raises(ValueError, match="unknown object"):
        scenario_config_from_dict(doc)


def test_prediction_step_budget_rejects_valid_but_unbounded_work(tmp_path):
    doc = scenario(tmp_path)
    doc["ground_segment"]["prediction_step_s"] = 1e-12
    with pytest.raises(ValueError, match="too many steps"):
        scenario_config_from_dict(doc)
    with pytest.raises(ValueError, match="bounded step count"):
        receiver(prediction_step_s=1e-12).advance(1.0)

    doc["ground_segment"] = config()
    doc["ground_segment"]["priors"]["sat"]["epoch_s"] = -10_001.0
    with pytest.raises(ValueError, match="too many steps"):
        scenario_config_from_dict(doc)


def test_reconnection_refreshes_telemetry_after_latency():
    adapter = SimulatedGroundSegment(
        config(
            latency_s=1.0,
            outages=[dict(station_id="test", start_s=1.0, end_s=3.0, services=["tracking", "downlink", "uplink"])],
        ),
        stations=[STATION],
        object_ids=["sat"],
        jd_utc_start=JD,
        frame_context=FRAMES,
    )
    for t in range(5):
        s = adapter.step(float(t), truth={"sat": X}, resources={"sat": {"temperature_k": 290.0 + t}})
        if t in (1, 2, 3):
            assert s["objects"]["sat"]["telemetry"]["temperature_k"]["value"] == 290.0
    assert s["objects"]["sat"]["telemetry"]["temperature_k"]["value"] == 293.0
    assert s["objects"]["sat"]["orbit"]["posterior_epoch_s"] == 3.0


def test_pruning_retains_live_knowledge():
    g = receiver()
    g.advance(0.0, packets=[packet(0.0), tracking(0.0)])
    g.advance(5.0)
    g.retain_from(5.0)
    assert len(g.history) == 1
    assert not g.snapshot(0.0)
    assert g.advance(6.0)["objects"]["sat"]["telemetry"]["temperature_k"]["value"] == 290.0


def test_multiple_stations_share_posterior():
    from dataclasses import replace

    station2 = replace(STATION, id="second", lon_deg=2.0)
    g = GroundSegment(config(), stations=[STATION, station2], object_ids=["sat"], jd_utc_start=JD, frame_context=FRAMES)
    p1 = tracking(0.0)
    p2 = tracking(0.0)
    p2["station_id"] = "second"
    values = predict_ground_station_measurement(target_state_eci=X, station=asdict(station2), t_s=0.0, jd_utc_start=JD)
    p2["data"]["vector"] = [values[k] for k in p2["data"]["components"]]
    s = g.advance(0.0, packets=[p1, p2])
    assert all(p["disposition"] == "accepted" for p in s["packets"])
    assert len(s["objects"]["sat"]["tracking"]) == 2
    assert np.trace(s["objects"]["sat"]["orbit"]["covariance"]) < 3.0


def test_custom_step_rejected_before_advancing(tmp_path):
    from sim.api import SimulationConfig, SimulationSession

    session = SimulationSession(SimulationConfig.from_dict(scenario(tmp_path)))
    with pytest.raises(ValueError, match="fixed"):
        session.step(dt_s=2.0)
    assert session.step().ground_segment["time_s"] == 1.0


def test_unobserved_truth_change_does_not_change_ground_estimate():
    cfg = config(outages=[dict(station_id="test", start_s=0.0, end_s=10.0, services=["tracking", "downlink"])])
    a = SimulatedGroundSegment(cfg, stations=[STATION], object_ids=["sat"], jd_utc_start=JD)
    b = SimulatedGroundSegment(cfg, stations=[STATION], object_ids=["sat"], jd_utc_start=JD)
    for t in (0.0, 1.0, 2.0):
        sa = a.step(t, truth={"sat": X}, resources={"sat": {"temperature_k": 290.0}})
        sb = b.step(
            t, truth={"sat": X + np.array([100.0, 0, 0, 0, 1.0, 0])}, resources={"sat": {"temperature_k": 400.0}}
        )
        assert sa["objects"] == sb["objects"]

from __future__ import annotations

import numpy as np
import pytest

from sim.core.models import Measurement, StateTruth
from sim.knowledge.object_tracking import (
    KnowledgeConditionConfig,
    KnowledgeEKFConfig,
    KnowledgeNoiseConfig,
    ObjectKnowledgeBase,
    TrackedObjectConfig,
)
from sim.sensors.access import AccessConfig, AccessModel
from sim.sensors.composite import CompositeSensorModel
from sim.sensors.joint_state import JointStateSensor
from sim.sensors.models import OwnStateSensor, RelativeSensor, SensorNoiseConfig
from sim.sensors.noisy_own_state import NoisyOwnStateSensor


def _truth(
    *,
    position: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    t_s: float = 0.0,
) -> StateTruth:
    return StateTruth(
        position_eci_km=np.array(position if position is not None else [7000.0, 0.0, 0.0], dtype=float),
        velocity_eci_km_s=np.array(velocity if velocity is not None else [0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.array([0.01, -0.02, 0.03], dtype=float),
        mass_kg=100.0,
        t_s=float(t_s),
    )


def test_own_state_sensor_zero_noise_returns_truth_state() -> None:
    truth = _truth()
    sensor = OwnStateSensor(
        noise=SensorNoiseConfig(sigma=np.zeros(6), bias=np.zeros(6)),
        rng=np.random.default_rng(1),
    )

    meas = sensor.measure(truth, env={}, t_s=truth.t_s)

    assert meas is not None
    assert np.allclose(meas.vector, np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)))
    assert meas.t_s == truth.t_s


def test_own_state_sensor_latency_preserves_acquisition_time() -> None:
    sensor = OwnStateSensor(
        noise=SensorNoiseConfig(sigma=np.zeros(6), bias=np.zeros(6), latency_s=2.0),
        rng=np.random.default_rng(11),
    )

    first = sensor.measure(_truth(t_s=0.0), env={}, t_s=0.0)
    second = sensor.measure(_truth(position=np.array([7002.0, 0.0, 0.0]), t_s=2.0), env={}, t_s=2.0)

    assert first is None
    assert second is not None
    assert np.isclose(second.t_s, 0.0)
    assert np.allclose(second.vector[:3], np.array([7000.0, 0.0, 0.0]))


class _FixedSensor:
    def __init__(self, vector: list[float], sample_t_s: float) -> None:
        self.vector = np.array(vector, dtype=float)
        self.sample_t_s = float(sample_t_s)

    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement:
        return Measurement(vector=self.vector, t_s=self.sample_t_s)


def test_composite_sensor_rejects_mixed_acquisition_times() -> None:
    sensor = CompositeSensorModel(
        sensors=[
            _FixedSensor([1.0, 2.0, 3.0], sample_t_s=8.0),
            _FixedSensor([4.0, 5.0, 6.0], sample_t_s=10.0),
        ]
    )

    meas = sensor.measure(_truth(t_s=10.0), env={}, t_s=10.0)

    assert meas is None


def test_composite_sensor_concatenates_synchronized_measurements() -> None:
    sensor = CompositeSensorModel(
        sensors=[
            _FixedSensor([1.0, 2.0, 3.0], sample_t_s=8.0),
            _FixedSensor([4.0, 5.0, 6.0], sample_t_s=8.0),
        ]
    )

    meas = sensor.measure(_truth(t_s=10.0), env={}, t_s=10.0)

    assert meas is not None
    assert np.isclose(meas.t_s, 8.0)
    assert np.allclose(meas.vector, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_relative_sensor_range_rate_matches_truth_geometry() -> None:
    observer = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.5, 0.0]))
    target = _truth(position=np.array([7003.0, 4.0, 0.0]), velocity=np.array([0.0, 7.4, 0.0]))
    sensor = RelativeSensor(
        target_id="target",
        mode="range_rate",
        noise=SensorNoiseConfig(sigma=np.zeros(2), bias=np.zeros(2)),
        rng=np.random.default_rng(2),
    )

    meas = sensor.measure(observer, env={"world_truth": {"target": target}}, t_s=0.0)

    rel_r = target.position_eci_km - observer.position_eci_km
    rel_v = target.velocity_eci_km_s - observer.velocity_eci_km_s
    expected_range = np.linalg.norm(rel_r)
    expected_range_rate = float(np.dot(rel_v, rel_r / expected_range))
    assert meas is not None
    assert np.allclose(meas.vector, np.array([expected_range, expected_range_rate]))


def test_access_model_enforces_cadence_range_and_fov() -> None:
    access = AccessModel(
        AccessConfig(
            update_cadence_s=10.0,
            max_range_km=5.0,
            fov_half_angle_rad=np.deg2rad(20.0),
        )
    )
    observer = np.array([7000.0, 0.0, 0.0], dtype=float)

    assert access.can_update(observer, observer + np.array([3.0, 0.0, 0.0]), 0.0, boresight_eci=np.array([1.0, 0.0, 0.0]))
    assert not access.can_update(
        observer,
        observer + np.array([3.0, 0.0, 0.0]),
        5.0,
        boresight_eci=np.array([1.0, 0.0, 0.0]),
    )
    assert not access.can_update(
        observer,
        observer + np.array([6.0, 0.0, 0.0]),
        11.0,
        boresight_eci=np.array([1.0, 0.0, 0.0]),
    )
    assert not access.can_update(
        observer,
        observer + np.array([0.0, 3.0, 0.0]),
        12.0,
        boresight_eci=np.array([1.0, 0.0, 0.0]),
    )


def test_joint_state_sensor_zero_noise_preserves_normalized_attitude_and_cadence() -> None:
    truth = _truth()
    sensor = JointStateSensor(
        pos_sigma_km=0.0,
        vel_sigma_km_s=0.0,
        quat_sigma=0.0,
        omega_sigma_rad_s=0.0,
        update_cadence_s=2.0,
        rng=np.random.default_rng(3),
    )

    first = sensor.measure(truth, env={}, t_s=0.0)
    skipped = sensor.measure(truth, env={}, t_s=1.0)
    second = sensor.measure(truth, env={}, t_s=2.0)

    assert first is not None
    assert skipped is None
    assert second is not None
    assert np.allclose(first.vector[:6], np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)))
    assert np.allclose(first.vector[6:10], truth.attitude_quat_bn)
    assert np.isclose(np.linalg.norm(first.vector[6:10]), 1.0)
    assert np.allclose(first.vector[10:13], truth.angular_rate_body_rad_s)


def test_noisy_own_state_sensor_noise_statistics_are_centered() -> None:
    truth = _truth()
    sensor = NoisyOwnStateSensor(pos_sigma_km=0.01, vel_sigma_km_s=0.001, rng=np.random.default_rng(4))

    samples = np.array([sensor.measure(truth, env={}, t_s=float(i)).vector for i in range(512)], dtype=float)
    errors = samples - np.hstack((truth.position_eci_km, truth.velocity_eci_km_s))

    assert np.all(np.abs(np.mean(errors[:, :3], axis=0)) < 0.002)
    assert np.all(np.abs(np.mean(errors[:, 3:], axis=0)) < 0.0002)
    assert np.all(np.std(errors[:, :3], axis=0) > 0.005)
    assert np.all(np.std(errors[:, 3:], axis=0) > 0.0005)


def test_object_knowledge_base_exposes_raw_state_measurement_snapshot() -> None:
    observer = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.5, 0.0]))
    target = _truth(position=np.array([7001.0, 2.0, 3.0]), velocity=np.array([0.01, 7.49, -0.02]))
    target_state = np.hstack((target.position_eci_km, target.velocity_eci_km_s))
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(
                    pos_sigma_km=np.zeros(3),
                    vel_sigma_km_s=np.zeros(3),
                ),
                ekf=KnowledgeEKFConfig(
                    process_noise_diag=np.ones(6) * 1e-12,
                    meas_noise_diag=np.ones(6) * 1e-12,
                    init_cov_diag=np.ones(6) * 1e-6,
                ),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    beliefs = knowledge.update(observer, {"target": target}, t_s=0.0)
    measurements = knowledge.measurement_snapshot()

    assert "target" in beliefs
    assert "target" in measurements
    assert np.allclose(measurements["target"], target_state)


def test_measured_state_estimator_trusts_latest_state_measurement() -> None:
    observer = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.5, 0.0]))
    target0 = _truth(position=np.array([7001.0, 2.0, 3.0]), velocity=np.array([0.01, 7.49, -0.02]))
    target1 = _truth(position=np.array([7002.0, 2.5, 3.5]), velocity=np.array([0.02, 7.48, -0.01]))
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(
                    pos_sigma_km=np.zeros(3),
                    vel_sigma_km_s=np.zeros(3),
                ),
                estimator="measured_state",
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    first = knowledge.update(observer, {"target": target0}, t_s=0.0)["target"]
    second = knowledge.update(observer, {"target": target1}, t_s=1.0)["target"]

    np.testing.assert_allclose(first.state, np.hstack((target0.position_eci_km, target0.velocity_eci_km_s)))
    np.testing.assert_allclose(second.state, np.hstack((target1.position_eci_km, target1.velocity_eci_km_s)))
    assert np.all(np.diag(second.covariance) > 0.0)
    assert knowledge.consistency_summary()["target"]["update_count"] == 2


def test_measured_state_estimator_requires_state_measurement_model() -> None:
    with pytest.raises(ValueError, match="measured_state requires measurement_model='state'"):
        ObjectKnowledgeBase(
            observer_id="chaser",
            tracked_objects=[
                TrackedObjectConfig(
                    target_id="target",
                    estimator="measured_state",
                    measurement_model="relative_range_rate",
                )
            ],
            dt_s=1.0,
            rng=np.random.default_rng(5),
        )


def test_relative_knowledge_requires_explicit_initial_state_prior() -> None:
    observer = _truth(position=np.array([7000.0, 0.0, 0.0]), velocity=np.array([0.0, 7.5, 0.0]))
    target = _truth(position=np.array([7001.0, 2.0, 3.0]), velocity=np.array([0.01, 7.49, -0.02]))
    knowledge = ObjectKnowledgeBase(
        observer_id="chaser",
        tracked_objects=[
            TrackedObjectConfig(
                target_id="target",
                conditions=KnowledgeConditionConfig(refresh_rate_s=1.0),
                sensor_noise=KnowledgeNoiseConfig(range_sigma_km=0.0, range_rate_sigma_km_s=0.0),
                measurement_model="relative_range_rate",
                ekf=KnowledgeEKFConfig(init_cov_diag=np.ones(6)),
            )
        ],
        dt_s=1.0,
        rng=np.random.default_rng(5),
    )

    with pytest.raises(ValueError, match="initial_state_eci_km_s"):
        knowledge.update(observer, {"target": target}, t_s=0.0)

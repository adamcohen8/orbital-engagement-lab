"""Parity and selection tests for the optional, parallel Trainer backend."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("oel_rust_game")
pytest.importorskip("oel_rust_orbit")

from sim.api import SimulationConfig
from sim.control.orbit.predictive_engagement import select_evasion_action, select_intercept_action
from sim.control.orbit.ric_pd import RICPDTransferController
from sim.flight_software import (
    GamePilotMode,
    GamePilotReferenceFlightSoftwareStack,
    canonical_json_bytes,
    to_primitive,
)
from sim.flight_software.reference_stacks import RpoReferenceFlightSoftwareStack
from sim.flight_software.rust_game_control import RustRICPDTransferController, native_predictive_action
from sim.flight_software.rust_game_stacks import RustGamePilotFlightSoftwareStack, RustRpoFlightSoftwareStack
from sim.game.attempt_lifecycle import _start_game_attempt
from sim.game.backend import RustGamePhysicsSession, configure_game_backend, create_game_physics_session
from sim.game.manual import KeyboardCommandState
from sim.game.operator import OperatorBurn, OperatorBurnPlan
from sim.game.session import GamePhysicsSession
from sim.game.training import RPOTrainingConfig, RPOTrainingTracker
from sim.gnc.orbit_v2 import TranslationMode
from sim.tests.fsw_v2_helpers import batch, boot_event, ideal_event
from sim.tests.fsw_v2_orbit_helpers import fault_event, navigation_batch, rpo_config
from sim.tests.game_fsw_v2_helpers import game_stack_config
from sim.tests.test_game_fsw_pilot_stack import _batch, _pilot_event


def assert_parity(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_parity(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_parity(a, b)
    elif isinstance(left, float):
        assert right == pytest.approx(left, rel=2e-12, abs=2e-12, nan_ok=True)
    else:
        assert left == right


@pytest.mark.parametrize("mode", tuple(GamePilotMode))
def test_pilot_modes_and_restore(mode):
    config = game_stack_config(mode)
    python, rust = GamePilotReferenceFlightSoftwareStack(config), RustGamePilotFlightSoftwareStack(config)
    assert python.identity.implementation_hash != rust.identity.implementation_hash
    for s in (python, rust):
        s.boot(boot_event())
    for invocation in range(1, 5):
        current = _batch(
            invocation,
            _pilot_event(
                invocation,
                invocation,
                ("translate_r", 0.75),
                ("translate_i", -0.75),
                ("pitch", 0.3),
                ("throttle", 0.6),
                ("deployment", 0.4),
                ("bank", -0.6),
                pressed=("fire",),
            ),
        )
        assert_parity(to_primitive(python.step(current)), to_primitive(rust.step(current)))
    checkpoint = rust.snapshot()
    current = _batch(5, _pilot_event(5, 5, ("translate_c", 0.6)))
    expected = rust.step(current)
    rust.restore(checkpoint)
    assert canonical_json_bytes(rust.step(current)) == canonical_json_bytes(expected)
    assert rust._navigator.numeric_backend == "rust"
    with pytest.raises(ValueError, match="implementation"):
        rust.restore(python.snapshot())


@pytest.mark.parametrize("mode", tuple(RpoReferenceFlightSoftwareStack.supported_primary_modes))
def test_rpo_modes_faults_and_restore(mode):
    config = rpo_config(mode, waypoints=((500.0, 0.0, 0.0, 0.0, 0.0, 0.0),) if mode is TranslationMode.WAYPOINT else ())
    config = replace(
        config,
        control=replace(
            config.control, prediction_horizon_s=120.0, prediction_step_s=30.0, prediction_pulse_duration_s=60.0
        ),
    )
    python, rust = RpoReferenceFlightSoftwareStack(config), RustRpoFlightSoftwareStack(config)
    for s in (python, rust):
        s.boot(boot_event())
    for invocation in range(1, 4):
        current = navigation_batch(invocation, range_m=900.0 + invocation)
        assert_parity(to_primitive(python.step(current)), to_primitive(rust.step(current)))
    checkpoint = rust.snapshot()
    current = navigation_batch(4, range_m=800.0)
    expected = rust.step(current)
    rust.restore(checkpoint)
    assert canonical_json_bytes(rust.step(current)) == canonical_json_bytes(expected)
    assert type(rust._controller).__name__ == "RustTranslationController"
    faulted = batch(5, ideal_event(5, 5), fault_event(0, 5, "translation"))
    # Both runtimes receive the same fault boundary after equivalent prior state.
    python.step(current)
    assert_parity(to_primitive(python.step(faulted)), to_primitive(rust.step(faulted)))


@pytest.mark.parametrize("evasion", (False, True))
@pytest.mark.parametrize(
    "state",
    ((900.0, 120.0, -30.0, -0.4, 0.1, 0.02), (50.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1000.0, 0.0, 0.0, -0.1, 0.0, 0.0)),
)
def test_predictive_batched_parity(evasion, state):
    params = dict(
        mean_motion_rad_s=0.001078,
        max_acceleration_m_s2=0.02,
        horizon_s=180.0,
        step_s=30.0,
        pulse_duration_s=45.0,
        capture_radius_m=100.0,
        capture_margin_m=20.0,
        acceleration_fractions=(0.5, 1.0),
    )
    if evasion:
        params["opponent_max_acceleration_m_s2"] = 0.03
    expected = (select_evasion_action if evasion else select_intercept_action)(np.asarray(state), **params)
    actual = native_predictive_action(np.asarray(state), evasion=evasion, **params)
    assert_parity(to_primitive(expected), to_primitive(actual))


@pytest.mark.parametrize("maximum", (float("inf"), 1e-5, 0.0))
def test_transfer_all_phases_parity(maximum):
    kwargs = dict(
        max_accel_km_s2=maximum,
        mean_motion_rad_s=0.001078,
        transfer_time_s=4800.0,
        final_brake_start_s=180.0,
        terminal_start_s=750.0,
        terminal_range_km=0.2,
    )
    python, rust = RICPDTransferController(**kwargs), RustRICPDTransferController(**kwargs)
    for time, distance in (
        (0.0, 1.0),
        (100.0, 1.0),
        (301.0, 1.0),
        (4100.0, 1.0),
        (4650.0, 1.0),
        (4801.0, 1.0),
        (4900.0, 0.01),
    ):
        state = np.array([distance, 0.4, 0.2, -0.0001, 0.0002, 0.00003])
        expected = python.guide_relative_state(state, np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0]), t_s=time)
        actual = rust.guide_relative_state(state, np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0]), t_s=time)
        np.testing.assert_allclose(
            actual.acceleration_eci_km_s2, expected.acceleration_eci_km_s2, rtol=2e-12, atol=1e-16
        )
        assert_parity(expected.mode_flags, actual.mode_flags)
        assert_parity(python.snapshot_state(), rust.snapshot_state())


def test_default_explicit_selection_and_missing_wheel(monkeypatch):
    config = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_04_rendezvous.yaml")
    before = config.to_dict()
    assert configure_game_backend(config).scenario.metadata["game"]["backend"] == "rust"
    assert isinstance(create_game_physics_session(config), RustGamePhysicsSession)
    assert type(create_game_physics_session(config, backend="python")) is GamePhysicsSession
    python_root = config.to_dict()
    python_root.setdefault("metadata", {}).setdefault("game", {})["backend"] = "python"
    python_config = SimulationConfig.from_dict(python_root, source_path=config.source_path)
    configured = configure_game_backend(python_config)
    assert configured.to_dict()["simulator"]["dynamics"]["orbit"]["numeric_backend"] == "python"
    assert python_config.to_dict() == python_root
    assert type(create_game_physics_session(python_config)) is GamePhysicsSession
    assert config.to_dict() == before
    assert isinstance(create_game_physics_session(config, backend="rust"), RustGamePhysicsSession)
    with pytest.raises(ValueError, match="backend"):
        create_game_physics_session(config, backend="bogus")
    from sim.flight_software import rust_game_backend as native

    native.extension.cache_clear()

    def missing(_):
        raise ImportError("missing test wheel")

    monkeypatch.setattr(native, "import_module", missing)
    with pytest.raises(RuntimeError, match="requires.*wheel"):
        create_game_physics_session(config)
    native.extension.cache_clear()


@pytest.mark.parametrize("source", sorted(Path("sim/game/configs").glob("*.yaml")), ids=lambda p: p.stem)
def test_all_trainer_levels_parallel_physics(source):
    config = SimulationConfig.from_yaml(source)
    sessions = [create_game_physics_session(config, backend=b) for b in ("python", "rust")]
    snapshots = [s.reset(seed=7821) for s in sessions]
    for _tick in range(20):
        snapshots = [s.step(dt_s=0.5) for s in sessions]
        for oid in snapshots[0].truth:
            np.testing.assert_allclose(snapshots[1].truth[oid], snapshots[0].truth[oid], rtol=1e-12, atol=2e-8)
        assert snapshots[0].applied_thrust.keys() == snapshots[1].applied_thrust.keys()
        for oid in snapshots[0].applied_thrust:
            np.testing.assert_allclose(
                snapshots[1].applied_thrust[oid], snapshots[0].applied_thrust[oid], rtol=2e-12, atol=1e-15
            )
    for agent in sessions[1]._engine.agents.values():
        if agent.flight_software_runtime and agent.flight_software_runtime.stack.stack_id != "fsw.passive":
            assert agent.flight_software_runtime.stack.numeric_backend == "rust"
        path = agent.dynamics.orbit_propagator.last_numeric_path
        assert path.startswith("rust_")
        if source.stem != "game_training_rpo_bonus_drag_racing":
            assert path.startswith("rust_native")


@pytest.mark.parametrize("burn_time,delta_v", ((0.0, 0.1), (0.2, 0.001), (0.2, 0.5)))
def test_rust_operator_exact_impulse_and_score_parity(burn_time, delta_v):
    source = SimulationConfig.from_yaml("sim/game/configs/game_training_rpo_00_tutorial.yaml")
    plan = OperatorBurnPlan((OperatorBurn(burn_time, (delta_v, 0.0, 0.0)),))
    outputs = []
    for backend in ("python", "rust"):
        config = configure_game_backend(source, backend)
        training = RPOTrainingConfig.from_metadata(dict(config.scenario.metadata))
        session, _, initial = _start_game_attempt(
            config,
            command_state=KeyboardCommandState(),
            training_cfg=training,
            controlled_object_id="chaser",
            attitude_rate_deg_s=45.0,
            control_mode="ric_translation",
            ric_reference_object_id="target",
            operator_burn_plan=plan,
        )
        tracker = RPOTrainingTracker(training)
        tracker.record(initial)
        snapshot = session.step(dt_s=1.0)
        tracker.record(snapshot)
        session.record_scoring(tracker.score())
        realized = float(np.linalg.norm(snapshot.applied_thrust["chaser"])) * 1000.0
        assert realized == pytest.approx(delta_v, rel=1e-12, abs=1e-12)
        runtime = session._engine.agents["chaser"].flight_software_runtime
        active = [r for r in runtime.evidence.realizations if np.linalg.norm(r.realized_force_n) > 0.0]
        assert active[0].interval_start_ns == round(burn_time * 1e9)
        assert active[0].interval_end_ns - active[0].interval_start_ns == 1_000_000
        outputs.append(session.game_review_evidence())
    # Scoring identity and categorical outcome remain equal; trajectories differ
    # only at floating-point roundoff from native arithmetic.
    assert_parity(outputs[0]["game_scoring_events"], outputs[1]["game_scoring_events"])


@pytest.mark.parametrize(
    "name,args",
    (
        ("test_game_session_uses_stack_commands_and_physical_realization", ()),
        ("test_game_session_realizes_timed_translation_tap_as_fractional_thrust", (False,)),
        ("test_game_session_realizes_timed_translation_tap_as_fractional_thrust", (True,)),
        (
            "test_game_pilot_input_transitions_release_fsw_at_next_physics_interval",
            ("sim/game/configs/game_training_rpo_04_rendezvous.yaml", 0.25),
        ),
        (
            "test_game_pilot_input_transitions_release_fsw_at_next_physics_interval",
            ("sim/game/configs/game_training_rpo_bonus_cislunar_rendezvous.yaml", 1.0),
        ),
        ("test_game_session_realizes_timed_attitude_thrust_tap_for_accumulated_duration", ()),
        ("test_game_delta_v_budget_limits_physical_realization", ()),
        ("test_cislunar_game_translation_realizes_moon_centered_radial_thrust", ()),
    ),
)
def test_existing_physical_input_contracts_with_rust(monkeypatch, name, args):
    from sim.tests import test_game_fsw_runtime_integration as regression

    original = regression._start_game_attempt

    def native_attempt(config, **kwargs):
        result = original(configure_game_backend(config, "rust"), **kwargs)
        assert isinstance(result[0], RustGamePhysicsSession)
        return result

    monkeypatch.setattr(regression, "_start_game_attempt", native_attempt)
    getattr(regression, name)(*args)


@pytest.mark.parametrize(
    "name",
    (
        "test_typed_mission_load_atomically_replaces_goal_and_reruns_executive_gates",
        "test_mission_load_goal_parameters_replace_controller_target_atomically",
        "test_configured_keep_out_recovery_preempts_primary_and_commands_retreat",
    ),
)
def test_shared_mission_and_recovery_policy_with_native_components(monkeypatch, name):
    from sim.tests import test_fsw_stack_owned_recovery as regression

    stacks = []

    def native_stack(config):
        stack = RustRpoFlightSoftwareStack(config)
        stacks.append(stack)
        return stack

    monkeypatch.setattr(regression, "RpoReferenceFlightSoftwareStack", native_stack)
    getattr(regression, name)()
    assert all(type(s._controller).__name__ == "RustTranslationController" for s in stacks)


def test_unsupported_native_stack_configuration_fails_explicitly():
    from sim.gnc.orbit_v2 import TranslationAllocatorKind, TranslationControlLaw

    config = rpo_config()
    with pytest.raises(ValueError, match="reference_pd"):
        RustRpoFlightSoftwareStack(
            replace(config, control=replace(config.control, control_law=TranslationControlLaw.HCW_LQR))
        )
    with pytest.raises(ValueError, match="ideal-wrench"):
        RustRpoFlightSoftwareStack(
            replace(config, allocator=replace(config.allocator, kind=TranslationAllocatorKind.CONTINUOUS_ENGINE))
        )

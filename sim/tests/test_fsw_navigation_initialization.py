from __future__ import annotations

import pytest

from sim.gnc.navigation_v2 import LoadedOwnState, NavigationInitializationMode, OrbitNavigator
from sim.tests.fsw_v2_helpers import BODY_FRAME, INERTIAL_FRAME, clock, ideal_event
from sim.tests.fsw_v2_orbit_helpers import RELATIVE_FRAME, gnss_event


def _navigator(mode: NavigationInitializationMode, loaded: LoadedOwnState | None = None) -> OrbitNavigator:
    return OrbitNavigator(
        initialization=mode,
        body_frame=BODY_FRAME,
        inertial_frame=INERTIAL_FRAME,
        relative_frame=RELATIVE_FRAME,
        loaded_own_state=loaded,
    )


def test_cold_navigation_waits_for_a_supported_observation() -> None:
    navigator = _navigator(NavigationInitializationMode.COLD)
    assert not navigator.solution(clock(0)).own_state_valid
    navigator.ingest((ideal_event(0, 1),))
    assert not navigator.solution(clock(1)).own_state_valid
    navigator.ingest((gnss_event(0, 2),))
    solution = navigator.solution(clock(2))
    assert solution.own_state_valid
    assert solution.position_eci_m == (7_000_000.0, 0.0, 0.0)


def test_loaded_navigation_starts_from_explicit_onboard_state() -> None:
    loaded = LoadedOwnState((7_100_000.0, 0.0, 0.0), (0.0, 7_400.0, 0.0), clock(0))
    solution = _navigator(NavigationInitializationMode.LOADED, loaded).solution(clock(1))
    assert solution.own_state_valid
    assert solution.position_eci_m == loaded.position_eci_m
    assert solution.belief.own_state is not None
    assert solution.belief.own_state.frame == INERTIAL_FRAME


def test_ideal_navigation_accepts_only_explicit_ideal_sensor_packets() -> None:
    navigator = _navigator(NavigationInitializationMode.IDEAL)
    navigator.ingest((ideal_event(0, 1),))
    solution = navigator.solution(clock(1))
    assert solution.own_state_valid
    assert solution.attitude.valid_for_control


def test_loaded_mode_requires_a_loaded_state() -> None:
    with pytest.raises(ValueError, match="requires loaded_own_state"):
        _navigator(NavigationInitializationMode.LOADED)

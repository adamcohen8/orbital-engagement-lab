from __future__ import annotations

import pytest

from sim.flight_software.clocks import AffineClockModel, IdealClockModel, compare_clock_tags
from sim.flight_software.contracts import ClockScale, ClockTag, TimeValidity, ValidityInterval


def test_ideal_clock_quantizes_and_round_trips_elapsed_ticks() -> None:
    clock = IdealClockModel("clock", tick_period_ns=1_000)
    tag = clock.tag_from_sim_time_ns(12_345)
    assert tag.ticks == 12
    assert clock.sim_time_ns_from_tag(tag) == 12_000


def test_affine_clock_models_bias_drift_and_invalid_negative_epoch() -> None:
    clock = AffineClockModel("clock", tick_period_ns=100, bias_ns=10, drift_fraction=0.1)
    assert clock.tag_from_sim_time_ns(1_000).ticks == 11
    invalid = AffineClockModel("clock", bias_ns=-100).tag_from_sim_time_ns(0)
    assert invalid.validity is TimeValidity.INVALID
    assert invalid.ticks == 0


def test_clock_comparison_and_validity_require_one_domain() -> None:
    first = ClockTag("clock", 1, 1_000, ClockScale.ONBOARD)
    second = ClockTag("clock", 2, 1_000, ClockScale.ONBOARD)
    assert compare_clock_tags(first, second) == -1
    with pytest.raises(ValueError, match="not directly comparable"):
        compare_clock_tags(first, ClockTag("other", 2, 1_000, ClockScale.ONBOARD))
    with pytest.raises(ValueError, match="same clock domain"):
        ValidityInterval(first, ClockTag("other", 2, 1_000, ClockScale.ONBOARD))

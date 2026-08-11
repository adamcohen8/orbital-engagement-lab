from __future__ import annotations

from sim.flight_software.events import SatelliteEvent, SatelliteEventKernel, SatelliteEventPhase


def test_simultaneous_events_follow_normative_phase_order() -> None:
    kernel = SatelliteEventKernel()
    phases = (
        SatelliteEventPhase.OUTPUT_SAMPLE,
        SatelliteEventPhase.EVIDENCE,
        SatelliteEventPhase.COMMAND_EFFECTIVE,
        SatelliteEventPhase.COMMAND_PUBLICATION,
        SatelliteEventPhase.TASK_RELEASE,
        SatelliteEventPhase.INPUT_DELIVERY,
        SatelliteEventPhase.SAMPLE_AND_STATUS,
    )
    for index, phase in enumerate(phases):
        kernel.schedule(SatelliteEvent(1_000, phase, phase.name.lower(), order=index))
    handled: list[str] = []
    advances: list[tuple[int, int]] = []
    trace = kernel.run_until(
        1_000,
        advance_physics=lambda start, end: advances.append((start, end)),
        handle_event=lambda event: handled.append(event.event_id),
    )
    assert advances == [(0, 1_000)]
    assert handled == [phase.name.lower() for phase in reversed(phases)]
    assert [entry.phase for entry in trace] == [
        SatelliteEventPhase.PHYSICS_ADVANCE,
        SatelliteEventPhase.PHYSICAL_COMMIT,
        *reversed(phases),
    ]


def test_same_time_task_order_and_input_tie_break_are_deterministic() -> None:
    kernel = SatelliteEventKernel(start_time_ns=10)
    for source, sequence, order in (("z", 2, 1), ("a", 4, 1), ("a", 3, 1), ("task", 0, 0)):
        phase = SatelliteEventPhase.TASK_RELEASE if source == "task" else SatelliteEventPhase.INPUT_DELIVERY
        kernel.schedule(
            SatelliteEvent(10, phase, f"{source}-{sequence}", order=order, source_id=source, sequence=sequence)
        )
    handled: list[str] = []
    kernel.run_until(10, advance_physics=lambda *_: None, handle_event=lambda event: handled.append(event.event_id))
    assert handled == ["a-3", "a-4", "z-2", "task-0"]


def test_same_time_tasks_use_declared_task_order() -> None:
    kernel = SatelliteEventKernel()
    kernel.schedule(SatelliteEvent(10, SatelliteEventPhase.TASK_RELEASE, "allocation", order=3))
    kernel.schedule(SatelliteEvent(10, SatelliteEventPhase.TASK_RELEASE, "navigation", order=0))
    kernel.schedule(SatelliteEvent(10, SatelliteEventPhase.TASK_RELEASE, "control", order=2))
    handled: list[str] = []
    kernel.run_until(10, advance_physics=lambda *_: None, handle_event=lambda event: handled.append(event.event_id))
    assert handled == ["navigation", "control", "allocation"]


def test_command_generated_at_boundary_never_acts_retroactively() -> None:
    kernel = SatelliteEventKernel()
    intervals: list[tuple[int, int, str]] = []
    active = "old"

    def advance(start: int, end: int) -> None:
        intervals.append((start, end, active))

    def handle(event: SatelliteEvent) -> None:
        nonlocal active
        if event.phase is SatelliteEventPhase.COMMAND_EFFECTIVE:
            active = "new"

    kernel.schedule(SatelliteEvent(100, SatelliteEventPhase.COMMAND_EFFECTIVE, "new-command"))
    kernel.run_until(200, advance_physics=advance, handle_event=handle)
    assert intervals == [(0, 100, "old"), (100, 200, "new")]

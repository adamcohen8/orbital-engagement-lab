from __future__ import annotations

from sim.utils.parallel import format_parallel_fallback_reason


def test_format_parallel_fallback_reason_preserves_detail() -> None:
    assert format_parallel_fallback_reason(PermissionError("sandbox denied")) == "PermissionError: sandbox denied"


def test_format_parallel_fallback_reason_explains_empty_exception() -> None:
    assert (
        format_parallel_fallback_reason(EOFError())
        == "EOFError: worker process exited without an error message"
    )

"""Native density-frame export preserves callback protocols and guards."""
import numpy as np
import pytest

native = pytest.importorskip("oel_rust_orbit")
pytestmark = pytest.mark.skipif(
    not hasattr(native.ONPStageContext, "supports_density_frame_inputs"),
    reason="requires rebuilt density-frame export candidate",
)
STATE = [7000.0, 300.0, 900.0, 0.1, 7.4, 0.8]


def _stage(callback, fallback, *, bodies=False, rotation=False, paths=(), codes=(2, 3)):
    frames = [(0, None, [0.0] * 7, [], "") for _ in range(5)]
    return native.ONPStageContext(
        None, frames, list(codes), 2, [0.0] * 6, False, callback, None,
        4, None, None, None, list(paths), fallback,
        density_body_inputs=bodies, density_frame_inputs=rotation,
    )


@pytest.mark.parametrize("bodies,rotation,arity", [(False, False, 3), (True, False, 4), (True, True, 5)])
def test_density_frame_protocol_keeps_legacy_arity_and_exact_matrix(bodies, rotation, arity):
    calls = []
    def density(*args):
        calls.append(args)
        return 1e-12 + args[2][0] * 1e-18
    stage = _stage(density, lambda *_: pytest.fail("unexpected fallback"), bodies=bodies, rotation=rotation)
    matrix_owner = _stage(None, lambda *_: pytest.fail("unexpected matrix fallback"), codes=(1,))
    states = [STATE, [value * 1.01 for value in STATE]]
    rows = [stage(120.0, state) for state in states]
    assert rows[0][9] != rows[1][9]
    assert rows[0][10:13] != rows[1][10:13]
    for call, row, state in zip(calls, rows, states):
        assert len(call) == arity
        np.testing.assert_array_equal(call[1], state)
        if bodies:
            np.testing.assert_array_equal(call[3][0], row[13:16])
            np.testing.assert_array_equal(call[3][1], row[16:19])
        if rotation:
            np.testing.assert_array_equal(call[4], matrix_owner(120.0, state)[:9])


def test_density_frame_protocol_requires_prepared_bodies():
    with pytest.raises(ValueError, match="density frame inputs require prepared body inputs"):
        _stage(lambda *_: 0.0, lambda *_: [], rotation=True)


def test_density_frame_protocol_retains_callback_errors_and_resource_fallback(tmp_path):
    resource = tmp_path / "input.txt"
    resource.write_text("initial")
    fallback_calls = []
    def density(*_):
        raise RuntimeError("density owner failed")
    def fallback(*args):
        fallback_calls.append(args)
        return [0.0] * 19
    stage = _stage(density, fallback, bodies=True, rotation=True, paths=[str(resource)])
    with pytest.raises(RuntimeError, match="density owner failed"):
        stage(120.0, STATE)
    resource.write_text("rewritten")
    assert stage(120.0, STATE) == [0.0] * 19
    assert stage.using_fallback()
    assert len(fallback_calls) == 1 and len(fallback_calls[0]) == 2

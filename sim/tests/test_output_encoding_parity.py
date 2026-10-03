"""Output encoding optimizations retain every value and rendered pixel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim.utils.io import _iter_json_safe, _native_json_fallback, json_safe, write_json


@pytest.mark.parametrize("indent", [None, 0, 2, "\t"])
def test_bounded_json_leaf_encoding_retains_exact_historical_bytes(indent) -> None:
    payload = {
        "escaped": [["comma, quote\"", "new\nline", "é雪", True, None]],
        "numeric": [[0, -0.0, 1e-25, float("nan"), float("inf"), -float("inf")]],
        "primitive_mapping": {0: "before collision", "0": "after collision", "value": float("nan")},
        "numpy": [np.float64(1.2), np.int64(3)],
        "long": [float(index) for index in range(2050)],
    }
    # Compact mode historically uses no separator whitespace.
    kwargs = {"separators": (",", ":")} if indent is None else {}
    expected = json.dumps(json_safe(payload), indent=indent, allow_nan=False, **kwargs)
    assert "".join(_iter_json_safe(payload, indent=indent)) == expected


def test_png_fast_compression_preserves_pixels_metadata_and_dpi(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    from sim.plotting.style import OELArtifactMetadata, save_oel_figure

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1, 2], [1, 3, 2], label="position")
    ax.set(xlabel="Time, seconds", ylabel="Distance, km", title="Encoding parity")
    ax.legend()
    fig.tight_layout()
    fast = tmp_path / "fast.png"
    historical = tmp_path / "historical.png"
    try:
        save_oel_figure(
            fig, fast, dpi=80,
            metadata=OELArtifactMetadata(scenario_name="parity", generated_utc="2026-09-27", version="test"),
            artifact_id="parity",
        )
        # Same fully styled/footer-bearing canvas through historical encoding.
        fig.savefig(historical, dpi=80, pil_kwargs={"compress_level": 6})
        with Image.open(fast) as actual, Image.open(historical) as expected:
            assert np.array_equal(np.asarray(actual), np.asarray(expected))
            assert actual.info == expected.info
            assert actual.size == expected.size
    finally:
        plt.close(fig)


def test_native_json_chunks_preserve_bytes_fallback_and_bound(tmp_path: Path) -> None:
    native = pytest.importorskip("oel_rust_orbit")
    if not hasattr(native, "json_safe_chunks"):
        pytest.skip("requires optional native JSON encoder")

    class Integer(int):
        def __repr__(self):
            return "unusable custom repr"

    payload = {
        "keys": {3: "overwritten", "3": "retained", None: "none"},
        "aliases": [[1, 2], [1, 2]],
        "fallback": [np.array([1.5]).item(), np.int64(8), Path("snow 雪"), Integer(17)],
        "floats": [-0.0, 1e-300, 1e300, float("nan"), float("inf"), -float("inf")],
        "large_int": 10 ** 200,
        "unicode": "surrogate\ud800 snow雪 newline\n" * 100000,
    }
    expected = json.dumps(json_safe(payload), indent=2, allow_nan=False)
    chunks = list(native.json_safe_chunks(payload, _native_json_fallback))
    assert len(chunks) > 1
    assert all(0 < len(chunk.encode("utf-8")) <= 2 * 1024 * 1024 for chunk in chunks)
    assert "".join(chunks) == expected
    output = tmp_path / "safe.json"
    write_json(str(output), payload)
    assert output.read_text() == expected


def test_native_json_errors_preserve_existing_atomic_output(tmp_path: Path) -> None:
    native = pytest.importorskip("oel_rust_orbit")
    if not hasattr(native, "json_safe_chunks"):
        pytest.skip("requires optional native JSON encoder")
    output = tmp_path / "safe.json"
    output.write_text("original")
    loop = []
    loop.append(loop)
    for payload, error in (({"cycle": loop}, ValueError), ({"bad": object()}, TypeError)):
        iterator = native.json_safe_chunks(payload, _native_json_fallback)
        with pytest.raises(error):
            next(iterator)
        with pytest.raises(StopIteration):
            next(iterator)
        with pytest.raises(error):
            write_json(str(output), payload)
        assert output.read_text() == "original"
        assert not output.with_name("safe.json.tmp").exists()


@pytest.mark.parametrize("mutation", ["append", "truncate"])
def test_native_json_retains_sequence_mutation_during_scalar_fallback(mutation) -> None:
    native = pytest.importorskip("oel_rust_orbit")
    if not hasattr(native, "json_safe_chunks"):
        pytest.skip("requires optional native JSON encoder")

    def payload():
        rows = []

        class Scalar:
            def item(self):
                if mutation == "append":
                    rows.append(4)
                else:
                    rows.pop()
                return 1

        rows.extend([Scalar(), 2])
        return {"rows": rows}

    expected = "".join(_iter_json_safe(payload(), indent=2))
    assert "".join(native.json_safe_chunks(payload(), _native_json_fallback)) == expected


def test_custom_dict_key_mutation_retains_python_error_and_atomic_cleanup(tmp_path: Path) -> None:
    native = pytest.importorskip("oel_rust_orbit")
    if not hasattr(native, "json_safe_chunks"):
        pytest.skip("requires optional native JSON encoder")

    def payload():
        rows = {}

        class Key:
            def __str__(self):
                rows["added"] = 3
                return "converted"

        rows.update({"start": 1, Key(): 2})
        return {"rows": rows}

    with pytest.raises(RuntimeError, match="dictionary changed size"):
        "".join(_iter_json_safe(payload(), indent=2))
    iterator = native.json_safe_chunks(payload(), _native_json_fallback)
    with pytest.raises(RuntimeError, match="dictionary changed size"):
        next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)
    output = tmp_path / "atomic.json"
    output.write_text("original")
    with pytest.raises(RuntimeError, match="dictionary changed size"):
        write_json(str(output), payload())
    assert output.read_text() == "original"
    assert not output.with_name("atomic.json.tmp").exists()


def test_text_layout_cache_invalidates_mutations_and_restores_on_error() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    from sim.plotting.style import _cache_builtin_text_layouts

    fig, ax = plt.subplots()
    text = ax.text(0.3, 0.4, "first\nsecond", rotation=25)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    try:
        with pytest.raises(RuntimeError, match="render failure"):
            with _cache_builtin_text_layouts(fig):
                assert "_get_layout" in text.__dict__
                mutations = (
                    lambda: text.set_text("new $x^2$\nrow"),
                    lambda: text.set_fontsize(19),
                    lambda: text.set_fontweight("bold"),
                    lambda: text.set_rotation(60),
                    lambda: text.set_rotation_mode("anchor"),
                    lambda: text.set_horizontalalignment("right"),
                    lambda: text.set_verticalalignment("top"),
                    lambda: text.set_multialignment("center"),
                    lambda: text.set_linespacing(1.6),
                    lambda: fig.set_dpi(120),
                )
                for mutate in mutations:
                    mutate()
                    for _ in range(2):
                        actual = text._get_layout(renderer)
                        expected = Text._get_layout(text, renderer)
                        np.testing.assert_array_equal(actual[0].bounds, expected[0].bounds)
                        assert actual[1:] == expected[1:]
                raise RuntimeError("render failure")
        assert "_get_layout" not in text.__dict__
    finally:
        plt.close(fig)


def test_text_layout_cache_restores_when_discovery_raises() -> None:
    import matplotlib.pyplot as plt

    from sim.plotting.style import _cache_builtin_text_layouts

    fig, axes = plt.subplots(1, 2)
    text = axes[0].text(0.3, 0.4, "existing")
    original_factory = axes[0].xaxis._get_tick

    def broken_legend():
        assert "_get_layout" in text.__dict__
        assert "_get_tick" in axes[0].xaxis.__dict__
        raise RuntimeError("legend discovery failure")

    axes[1].get_legend = broken_legend
    try:
        with pytest.raises(RuntimeError, match="legend discovery failure"):
            with _cache_builtin_text_layouts(fig):
                pytest.fail("discovery should fail before yielding")
        assert "_get_layout" not in text.__dict__
        assert "_get_tick" not in axes[0].xaxis.__dict__
        assert axes[0].xaxis._get_tick == original_factory
    finally:
        plt.close(fig)

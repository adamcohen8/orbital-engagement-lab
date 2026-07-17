from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim.utils.io import json_safe, write_json


def test_write_json_replaces_non_finite_numbers_with_null(tmp_path: Path) -> None:
    payload = {
        "finite": 1.5,
        "nan_value": float("nan"),
        "inf_value": float("inf"),
        "nested": {"neg_inf": float("-inf"), "scalar": np.float64(2.5)},
        "items": [1, np.float64(float("nan"))],
        "path": tmp_path / "artifact.txt",
    }

    out = tmp_path / "payload.json"
    write_json(str(out), payload)
    expected_text = json.dumps(json_safe(payload), indent=2, allow_nan=False)
    assert out.read_text(encoding="utf-8") == expected_text
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["finite"] == 1.5
    assert loaded["nan_value"] is None
    assert loaded["inf_value"] is None
    assert loaded["nested"]["neg_inf"] is None
    assert loaded["nested"]["scalar"] == 2.5
    assert loaded["items"] == [1.0, None]
    assert loaded["path"] == str(tmp_path / "artifact.txt")


def test_write_json_preserves_existing_file_when_serialization_fails(tmp_path: Path) -> None:
    out = tmp_path / "payload.json"
    out.write_text('{"status": "old"}', encoding="utf-8")

    class NotJsonSerializable:
        pass

    try:
        write_json(str(out), {"bad": NotJsonSerializable()})
    except TypeError:
        pass
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("write_json should reject non-serializable payloads")

    assert json.loads(out.read_text(encoding="utf-8")) == {"status": "old"}
    assert not (tmp_path / "payload.json.tmp").exists()


def test_json_safe_handles_numpy_scalars_and_paths(tmp_path: Path) -> None:
    cleaned = json_safe(
        {
            "count": np.int64(3),
            "flag": np.bool_(True),
            "metric": np.float64(4.25),
            "artifact": tmp_path / "demo.txt",
        }
    )

    assert cleaned == {
        "count": 3,
        "flag": True,
        "metric": 4.25,
        "artifact": str(tmp_path / "demo.txt"),
    }

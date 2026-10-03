from __future__ import annotations

from pathlib import Path

import pytest

from sim.interchange.materialization import (
    OGPMaterializationError,
    ONPMaterializationError,
    materialize_ogp,
    materialize_onp,
)


@pytest.mark.parametrize(
    ("materializer", "error_type"),
    [(materialize_onp, ONPMaterializationError), (materialize_ogp, OGPMaterializationError)],
)
def test_materializers_reject_aliasing_source_scenario_and_manifest(
    tmp_path: Path, materializer, error_type
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    scenario = tmp_path / "scenario.yaml"
    common = {
        "scenario_name": "alias_guard",
        "output_dir": tmp_path / "outputs",
        "duration_s": 10.0,
        "dt_s": 1.0,
    }

    with pytest.raises(error_type, match="targets must be distinct"):
        materializer(source, scenario_name=common["scenario_name"], scenario_path=source, **{k: v for k, v in common.items() if k != "scenario_name"})
    with pytest.raises(error_type, match="targets must be distinct"):
        materializer(source, scenario_name=common["scenario_name"], scenario_path=scenario, manifest_path=scenario, **{k: v for k, v in common.items() if k != "scenario_name"})
    with pytest.raises(error_type, match="targets must be distinct"):
        materializer(source, scenario_name=common["scenario_name"], scenario_path=scenario, manifest_path=source, **{k: v for k, v in common.items() if k != "scenario_name"})


def test_compiler_identity_includes_contract_and_mass_property_dependencies(monkeypatch) -> None:
    import sim.spacecraft_twin.services as services

    captured: dict[str, str] = {}
    monkeypatch.setattr(services, "file_digest", lambda path: str(path))
    monkeypatch.setattr(services, "digest", lambda payload: captured.update(payload) or "digest")

    assert services.compiler_identity() == "digest"
    assert "sim.spacecraft_twin.contracts" in captured
    assert "sim.digital_twin.mass_properties" in captured

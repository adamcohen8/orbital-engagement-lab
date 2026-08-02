from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import sim.handoff as handoff
from sim.interchange.architecture import INTERCHANGE_CAPABILITY_FAMILIES
from sim.interchange.cli import main as handoff_main
from sim.interchange.contracts import AGE_STATUSES, INTEGRITY_STATUSES, QUALITY_DISPOSITIONS
from sim.interchange.materialization import canonical_scenario_digest, materialize_onp
from sim.interchange.provenance import canonical_json_bytes, compute_manifest_id, compute_product_id
from sim.interchange.validation import validate_document, validate_product

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "sim" / "interchange" / "examples"
SCHEMAS = ROOT / "sim" / "interchange" / "schemas"
ACCEPTED_PRODUCT = EXAMPLES / "state_estimate_accepted_current.json"
FIXTURE_MATRIX = EXAMPLES / "validation_fixture_matrix.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _with_identity(document: dict) -> dict:
    document["product_id"] = compute_product_id(document)
    return document


def test_phase1_accepted_state_product_is_valid_and_promotable() -> None:
    product = _load(ACCEPTED_PRODUCT)
    report = validate_product(product, source_path=ACCEPTED_PRODUCT)

    assert report.valid is True
    assert report.promotable is True
    assert report.issues == ()
    assert compute_product_id(product) == product["product_id"]


def test_phase1_fixture_matrix_covers_every_quality_and_freshness_state() -> None:
    base = _load(ACCEPTED_PRODUCT)
    matrix = _load(FIXTURE_MATRIX)
    observed_quality = set()
    observed_integrity = set()
    observed_age = set()

    for case in matrix["cases"]:
        product = deepcopy(base)
        product["producer"]["run_id"] = case["case_id"]
        product["quality"]["disposition"] = case["quality_disposition"]
        product["freshness"]["integrity_status"] = case["integrity_status"]
        product["freshness"]["age_status"] = case["age_status"]
        product["freshness"]["policy"] = dict(case.get("age_policy", {}))
        _with_identity(product)

        report = validate_product(product, source_path=ACCEPTED_PRODUCT)

        assert report.valid is case["expected_valid"], case["case_id"]
        assert report.promotable is case["expected_promotable"], case["case_id"]
        observed_quality.add(case["quality_disposition"])
        observed_integrity.add(case["integrity_status"])
        observed_age.add(case["age_status"])

    assert observed_quality == set(QUALITY_DISPOSITIONS)
    assert observed_integrity == set(INTEGRITY_STATUSES)
    assert observed_age == set(AGE_STATUSES)


def test_canonical_identity_is_deterministic_and_excludes_paths_and_timestamps() -> None:
    product = _load(ACCEPTED_PRODUCT)
    reordered = dict(reversed(list(product.items())))
    relocated = deepcopy(reordered)
    relocated["created_utc"] = "2030-01-01T00:00:00Z"
    relocated["freshness"]["evaluated_utc"] = "2030-01-01T00:00:01Z"
    relocated["provenance"]["source_artifacts"][0]["path"] = "/relocated/source_od_report.json"

    assert canonical_json_bytes({"b": 1, "a": [2, 3]}) == b'{"a":[2,3],"b":1}'
    assert compute_product_id(reordered) == product["product_id"]
    assert compute_product_id(relocated) == product["product_id"]

    relocated["payload"]["state"]["values"][0] += 1.0
    assert compute_product_id(relocated) != product["product_id"]


def test_canonical_serialization_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_bytes({"value": float("nan")})


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda product: product.update({"schema_version": 2}), "schema.unsupported_version"),
        (lambda product: product.update({"unexpected": True}), "schema.unknown_field"),
        (lambda product: product["payload"]["state"].pop("epoch"), "schema.required"),
        (lambda product: product["payload"]["state"].update({"frame": "TEME"}), "state.frame_incompatible"),
    ],
)
def test_semantically_incomplete_or_incompatible_products_fail_precisely(mutation, code: str) -> None:
    product = _load(ACCEPTED_PRODUCT)
    mutation(product)
    _with_identity(product)

    report = validate_product(product, source_path=ACCEPTED_PRODUCT)

    assert report.valid is False
    assert any(issue.code == code for issue in report.issues)


def test_current_integrity_fails_when_source_hash_does_not_match() -> None:
    product = _load(ACCEPTED_PRODUCT)
    product["provenance"]["source_artifacts"][0]["sha256"] = "0" * 64
    product["payload"]["estimator_evidence"]["source_report_sha256"] = "0" * 64
    _with_identity(product)

    report = validate_product(product, source_path=ACCEPTED_PRODUCT)

    assert report.valid is False
    assert any(issue.code == "provenance.source_hash_mismatch" for issue in report.errors)


def test_skipping_source_verification_never_upgrades_promotability() -> None:
    product = _load(ACCEPTED_PRODUCT)

    report = validate_product(product, source_path=ACCEPTED_PRODUCT, verify_sources=False)

    assert report.valid is True
    assert report.promotable is False
    assert any(issue.code == "provenance.verification_skipped" for issue in report.blockers)


def test_inspect_and_validate_cli_are_read_only_and_report_blocked_exit_code(tmp_path: Path, capsys) -> None:
    product = _load(ACCEPTED_PRODUCT)
    product["quality"]["disposition"] = "rejected"
    _with_identity(product)
    path = tmp_path / "rejected.json"
    path.write_text(json.dumps(product), encoding="utf-8")
    before = path.read_bytes()

    assert handoff_main(["inspect", str(path), "--json", "--no-verify-sources"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["quality"]["disposition"] == "rejected"
    assert inspected["validation"]["promotable"] is False

    assert handoff_main(["validate-product", str(path), "--json", "--no-verify-sources"]) == 3
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True
    assert validated["promotable"] is False
    assert path.read_bytes() == before


def test_handoff_manifest_can_be_inspected_without_materialization() -> None:
    source = _load(ACCEPTED_PRODUCT)
    manifest = {
        "schema_id": "oel-handoff-manifest-v1",
        "schema_version": 1,
        "manifest_id": "oel.handoff_manifest:" + "0" * 64,
        "created_utc": "2026-08-01T12:30:00Z",
        "source_product_ids": [source["product_id"]],
        "source_hashes": {source["product_id"]: source["product_id"].split(":", 1)[1]},
        "adapter": {"adapter_id": "fixture", "adapter_version": "1"},
        "materialization_options": {"duration_s": 3600.0},
        "defaults_applied": {},
        "overrides": [],
        "source_markings": source["data_markings"],
        "output_markings": source["data_markings"],
        "output": {"kind": "scenario", "path": "generated.yaml", "digest": "a" * 64, "status": "validated"},
        "validation": {
            "safe_validation_result": {"status": "ok"},
            "ordinary_validation_result": {"status": "trust_required"},
        },
        "warnings": [],
        "failures": [],
        "recommended_next_action": "Review the generated scenario.",
        "execution_occurred": False,
    }
    manifest["manifest_id"] = compute_manifest_id(manifest)

    report = validate_document(manifest)

    assert report.valid is True
    assert report.document_type == "manifest"
    assert report.promotable is False


def test_interchange_facade_exports_only_focused_owner_capabilities() -> None:
    for family in INTERCHANGE_CAPABILITY_FAMILIES:
        owner = __import__(family.module, fromlist=["*"])
        for capability in family.capabilities:
            assert getattr(handoff, capability) is getattr(owner, capability)

    facade_path = ROOT / "sim" / "handoff.py"
    tree = ast.parse(facade_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imports
    assert all(module.startswith("sim.interchange") for module in imports)
    assert not any("adapters" in module for module in imports)


def test_phase1_json_schemas_are_closed_and_parseable() -> None:
    envelope = _load(SCHEMAS / "oel-product-envelope-v1.schema.json")
    state = _load(SCHEMAS / "oel-state-estimate-v1.schema.json")
    manifest = _load(SCHEMAS / "oel-handoff-manifest-v1.schema.json")

    assert envelope["properties"]["schema_id"]["const"] == "oel-product-envelope-v1"
    assert envelope["additionalProperties"] is False
    assert state["additionalProperties"] is False
    assert manifest["properties"]["execution_occurred"]["const"] is False


def test_accepted_product_materializes_a_valid_passive_onp_scenario_without_execution(tmp_path: Path) -> None:
    scenario_path = tmp_path / "continuation.yaml"
    result = materialize_onp(
        ACCEPTED_PRODUCT,
        scenario_name="accepted_product_continuation",
        scenario_path=scenario_path,
        output_dir=tmp_path / "run",
        duration_s=600.0,
        dt_s=10.0,
    )

    assert result["status"] == "materialized"
    assert result["execution_occurred"] is False
    assert result["safe_validation"]["ok"] is True
    assert result["ordinary_validation"]["status"] == "trust_required"
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    source = _load(ACCEPTED_PRODUCT)
    source_state = source["payload"]["state"]
    obj = scenario["objects"]["example_satellite"]
    assert obj["initial_state"]["position_eci_km"] == source_state["values"][:3]
    assert obj["initial_state"]["velocity_eci_km_s"] == source_state["values"][3:]
    assert scenario["simulator"]["initial_jd_utc"] == source_state["epoch"]["value"]
    assert scenario["simulator"]["dynamics"]["orbit"] == {"model": "two_body", "orbit_substep_s": 10.0}
    assert scenario["outputs"]["review"] == {"enabled": True, "detail": "standard"}
    assert canonical_scenario_digest(scenario) == json.loads(
        Path(result["manifest_path"]).read_text(encoding="utf-8")
    )["output"]["digest"]
    manifest_report = validate_document(_load(Path(result["manifest_path"])))
    assert manifest_report.valid is True


def test_nonpromotable_product_writes_blocked_manifest_but_no_scenario(tmp_path: Path) -> None:
    product = _load(ACCEPTED_PRODUCT)
    product["quality"]["disposition"] = "ambiguous"
    _with_identity(product)
    product_path = tmp_path / "ambiguous.json"
    product_path.write_text(json.dumps(product), encoding="utf-8")
    scenario_path = tmp_path / "must_not_exist.yaml"

    result = materialize_onp(
        product_path,
        scenario_name="blocked_continuation",
        scenario_path=scenario_path,
        output_dir=tmp_path / "run",
        duration_s=600.0,
        dt_s=10.0,
    )

    assert result["status"] == "blocked"
    assert result["execution_occurred"] is False
    assert scenario_path.exists() is False
    assert Path(result["manifest_path"]).is_file()
    assert any(item["code"] == "quality.ambiguous" for item in result["failures"])


def test_materialize_onp_cli_reports_success_and_never_executes(tmp_path: Path, capsys) -> None:
    scenario_path = tmp_path / "cli_continuation.yaml"
    exit_code = handoff_main(
        [
            "materialize-onp",
            "--state-product",
            str(ACCEPTED_PRODUCT),
            "--scenario-name",
            "cli_continuation",
            "--output",
            str(scenario_path),
            "--run-output-dir",
            str(tmp_path / "run"),
            "--duration-s",
            "300",
            "--dt-s",
            "10",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "materialized"
    assert payload["execution_occurred"] is False
    assert scenario_path.is_file()

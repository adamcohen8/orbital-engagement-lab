from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sim.analysis.mission_scheduling import MissionSchedulingError
from sim.analysis.mission_scheduling_sources import (
    MissionSchedulingSourcePlan,
    build_mission_scheduling_problem_from_sources,
    verify_source_built_mission_schedule,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/python/mission_scheduling_source_chain.py"


@pytest.fixture(scope="module")
def generated_chain(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("mission-source-chain")
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--output-root", str(root)],
        cwd=ROOT,
        env={
            **os.environ,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(root.parent / f"{root.name}-matplotlib"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["source_status"] == "verified"
    return root


def _plan(root: Path) -> dict:
    return json.loads((root / "source_plan.json").read_text(encoding="utf-8"))


def _copy_chain(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _rewrite_retained_plan(evidence: Path, plan: dict) -> None:
    plan_path = evidence / "normalized_source_plan.json"
    content = (
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    plan_path.write_bytes(content)
    manifest_path = evidence / "mission_schedule_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_plan_semantic_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    receipt = next(
        item for item in manifest["artifacts"] if item["path"] == "normalized_source_plan.json"
    )
    receipt["bytes"] = len(content)
    receipt["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_end_to_end_chain_selects_two_observations_and_delivers_both(
    generated_chain: Path,
) -> None:
    verified = verify_source_built_mission_schedule(generated_chain / "evidence")
    schedule = json.loads(
        (generated_chain / "evidence/schedule/mission_schedule_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    delivery_rows = (
        generated_chain / "evidence/schedule/mission_data_delivery.csv"
    ).read_text(encoding="utf-8")

    assert verified["source_status"] == "verified"
    assert verified["source_count"] == 5
    assert len(schedule["selected_opportunity_ids"]) == 4
    assert "sat-b-gs1-contended" not in "\n".join(schedule["selected_opportunity_ids"])
    assert delivery_rows.count(",True") == 2


def test_builder_converts_link_bits_to_bytes_and_preserves_source_identities(
    generated_chain: Path,
) -> None:
    built = build_mission_scheduling_problem_from_sources(
        _plan(generated_chain), base_dir=generated_chain
    )
    opportunities = {item.opportunity_id: item for item in built.problem.opportunities}

    assert opportunities["sat-a-gs1:interval:0"].downlink_capacity_bytes == 125_000_000.0
    assert opportunities["sat-a-collection:public_source_chain_sat_a:0000"].data_volume_bytes > 0.0
    assert len({item.source_product_sha256 for item in built.problem.opportunities}) == 5


def test_source_order_does_not_change_problem_or_schedule_identity(generated_chain: Path) -> None:
    forward_payload = _plan(generated_chain)
    reverse_payload = copy.deepcopy(forward_payload)
    reverse_payload["assets"].reverse()
    reverse_payload["collection_sources"].reverse()
    reverse_payload["link_sources"].reverse()

    forward = build_mission_scheduling_problem_from_sources(
        forward_payload, base_dir=generated_chain
    )
    reverse = build_mission_scheduling_problem_from_sources(
        reverse_payload, base_dir=generated_chain
    )
    assert reverse.result.input_semantic_sha256 == forward.result.input_semantic_sha256
    assert reverse.result.schedule_semantic_sha256 == forward.result.schedule_semantic_sha256


def test_collection_candidate_and_task_ledger_mismatch_fails_closed(
    generated_chain: Path, tmp_path: Path
) -> None:
    root = _copy_chain(generated_chain, tmp_path / "chain")
    path = root / "generated_sources/sat-a_collection.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["task_opportunities"][0]["start_s"] += 1.0
    path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(MissionSchedulingError, match="differs from its accepted candidate"):
        build_mission_scheduling_problem_from_sources(_plan(root), base_dir=root)


def test_independent_collection_resource_screen_is_rejected(
    generated_chain: Path, tmp_path: Path
) -> None:
    root = _copy_chain(generated_chain, tmp_path / "chain")
    path = root / "generated_sources/sat-a_collection.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["opportunity_candidates"][0]["resource_screen"]["enabled"] = True
    path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(MissionSchedulingError, match="must disable its independent resource screen"):
        build_mission_scheduling_problem_from_sources(_plan(root), base_dir=root)


def test_directed_link_artifact_tamper_fails_receipt_check(
    generated_chain: Path, tmp_path: Path
) -> None:
    root = _copy_chain(generated_chain, tmp_path / "chain")
    intervals = root / "generated_sources/sat-a-gs1/link_intervals.csv"
    intervals.write_text(intervals.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(MissionSchedulingError, match="artifact receipt mismatch"):
        build_mission_scheduling_problem_from_sources(_plan(root), base_dir=root)


def test_directed_link_semantic_identity_is_recomputed_from_retained_evidence(
    generated_chain: Path, tmp_path: Path
) -> None:
    root = _copy_chain(generated_chain, tmp_path / "chain")
    source = root / "generated_sources/sat-a-gs1"
    manifest_path = source / "link_analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["normalized_config"]["data_rate_bps"] = 50_000_000.0
    intervals_path = source / "link_intervals.csv"
    intervals = intervals_path.read_text(encoding="utf-8")
    assert "1000000000.0" in intervals
    intervals_path.write_text(intervals.replace("1000000000.0", "500000000.0"), encoding="utf-8")
    manifest["artifacts"]["link_intervals.csv"]["sha256"] = hashlib.sha256(
        intervals_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MissionSchedulingError, match="semantic SHA-256 does not match"):
        build_mission_scheduling_problem_from_sources(_plan(root), base_dir=root)


def test_directed_link_uplink_is_not_converted_to_downlink(
    generated_chain: Path, tmp_path: Path
) -> None:
    root = _copy_chain(generated_chain, tmp_path / "chain")
    manifest_path = root / "generated_sources/sat-a-gs1/link_analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["normalized_config"]
    config["tx_terminal"], config["rx_terminal"] = config["rx_terminal"], config["tx_terminal"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MissionSchedulingError, match="endpoint identities do not match"):
        build_mission_scheduling_problem_from_sources(_plan(root), base_dir=root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda plan: plan.update(epoch_jd_utc=2451546.0), "does not match source-plan epoch"),
        (lambda plan: plan.update(horizon_end_s=90.0), "interval outside the horizon"),
        (
            lambda plan: plan["link_sources"][0].update(station_asset_id="WRONG-STATION"),
            "endpoint identities do not match",
        ),
    ],
)
def test_epoch_horizon_and_endpoint_mismatches_fail_closed(
    generated_chain: Path, mutation, message: str
) -> None:
    payload = _plan(generated_chain)
    mutation(payload)
    with pytest.raises(MissionSchedulingError, match=message):
        build_mission_scheduling_problem_from_sources(payload, base_dir=generated_chain)


def test_slew_constrained_assets_require_declared_link_pointing(generated_chain: Path) -> None:
    payload = _plan(generated_chain)
    payload["assets"][0]["maximum_slew_rate_rad_s"] = 0.1
    with pytest.raises(MissionSchedulingError, match="requires explicit pointing_unit_eci"):
        MissionSchedulingSourcePlan.from_mapping(payload)


def test_source_ids_cannot_escape_retained_product_directory(generated_chain: Path) -> None:
    payload = _plan(generated_chain)
    payload["collection_sources"][0]["source_id"] = "../escape"
    with pytest.raises(MissionSchedulingError, match="simple portable identifier"):
        MissionSchedulingSourcePlan.from_mapping(payload)


@pytest.mark.parametrize("source_id", ["a:b", "CON", "aux.txt", "trailing.", "snowman-☃"])
def test_source_ids_must_be_portable_on_supported_filesystems(
    generated_chain: Path, source_id: str
) -> None:
    payload = _plan(generated_chain)
    payload["collection_sources"][0]["source_id"] = source_id
    with pytest.raises(MissionSchedulingError, match="simple portable identifier"):
        MissionSchedulingSourcePlan.from_mapping(payload)


def test_source_ids_must_be_case_insensitively_unique(generated_chain: Path) -> None:
    payload = _plan(generated_chain)
    payload["collection_sources"][0]["source_id"] = "Shared-Source"
    payload["link_sources"][0]["source_id"] = "shared-source"
    with pytest.raises(MissionSchedulingError, match="case-insensitively unique"):
        MissionSchedulingSourcePlan.from_mapping(payload)


def test_retained_source_tamper_fails_authoritative_replay(
    generated_chain: Path, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    shutil.copytree(generated_chain / "evidence", evidence)
    retained = evidence / "source_products/sat-a-collection/collection_evidence.json"
    retained.write_text(retained.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(MissionSchedulingError, match="receipt mismatch"):
        verify_source_built_mission_schedule(evidence)


def test_source_manifest_claim_tamper_fails_authoritative_replay(
    generated_chain: Path, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    shutil.copytree(generated_chain / "evidence", evidence)
    manifest_path = evidence / "mission_schedule_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["source_product_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MissionSchedulingError, match="claims differ"):
        verify_source_built_mission_schedule(evidence)


def test_source_replay_rejects_paths_outside_retained_inventory(
    generated_chain: Path, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    shutil.copytree(generated_chain / "evidence", evidence)
    plan = json.loads((evidence / "normalized_source_plan.json").read_text(encoding="utf-8"))
    plan["collection_sources"][0]["path"] = str(
        (generated_chain / "generated_sources/sat-a_collection.json").resolve()
    )
    _rewrite_retained_plan(evidence, plan)

    with pytest.raises(MissionSchedulingError, match="canonical retained path"):
        verify_source_built_mission_schedule(evidence)


def test_source_replay_requires_exact_per_source_receipt_inventory(
    generated_chain: Path, tmp_path: Path
) -> None:
    evidence = tmp_path / "evidence"
    shutil.copytree(generated_chain / "evidence", evidence)
    manifest_path = evidence / "mission_schedule_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["artifacts"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MissionSchedulingError, match="exact required artifact inventory"):
        verify_source_built_mission_schedule(evidence)


def test_build_solve_and_replay_sources_cli(generated_chain: Path, tmp_path: Path) -> None:
    output = tmp_path / "cli-evidence"
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.mission_scheduling",
            "build-solve",
            str(generated_chain / "source_plan.json"),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    replayed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.mission_scheduling",
            "replay-sources",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert built.returncode == 0, built.stdout + built.stderr
    assert replayed.returncode == 0, replayed.stdout + replayed.stderr
    assert json.loads(replayed.stdout)["source_status"] == "verified"

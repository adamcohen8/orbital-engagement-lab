"""Content-bound adapters from OEL collection/link evidence to mission scheduling."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.analysis.collection_opportunity import COLLECTION_OPPORTUNITY_EVIDENCE_SCHEMA
from sim.analysis.directed_link import (
    DIRECTED_LINK_CONTRACT_VERSION,
    recompute_directed_link_semantic_sha256,
)
from sim.analysis.mission_scheduling import (
    MAX_PUBLIC_MISSION_OPPORTUNITIES,
    AssetScheduleConstraints,
    MissionOpportunity,
    MissionSchedulingArtifacts,
    MissionSchedulingError,
    MissionSchedulingProblem,
    MissionSchedulingResult,
    solve_mission_schedule,
    verify_mission_scheduling_artifacts,
    write_mission_scheduling_artifacts,
)

MISSION_SCHEDULING_SOURCE_PLAN_SCHEMA = "oel.mission_scheduling_source_plan.v1"
MISSION_SCHEDULING_SOURCE_EVIDENCE_SCHEMA = "oel.mission_scheduling_source_evidence.v1"
_EPS = 1.0e-9
_PORTABLE_SOURCE_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*\Z")
_DOS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _required(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise MissionSchedulingError(f"{field} must be a non-empty string.")
    return result


def _source_id(value: Any, field: str) -> str:
    result = _required(value, field)
    reserved_stem = result.split(".", 1)[0].casefold()
    if (
        len(result) > 128
        or _PORTABLE_SOURCE_ID.fullmatch(result) is None
        or result.endswith(".")
        or reserved_stem in _DOS_RESERVED_NAMES
    ):
        raise MissionSchedulingError(f"{field} must be a simple portable identifier.")
    return result


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise MissionSchedulingError(f"{field} must be finite.")
    return result


def _digest_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_digest(value: Any) -> str:
    return _digest_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def _valid_digest(value: Any, field: str) -> str:
    digest = _required(value, field).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MissionSchedulingError(f"{field} must be a lowercase SHA-256 digest.")
    return digest


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MissionSchedulingError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MissionSchedulingError(f"{label} must be a JSON object.")
    return value, content


def _normalized_pointing(value: Any, field: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise MissionSchedulingError(f"{field} must contain three finite values.")
    if abs(float(np.linalg.norm(vector)) - 1.0) > 1.0e-10:
        raise MissionSchedulingError(f"{field} must be normalized within 1e-10.")
    return tuple(float(item) for item in vector)


@dataclass(frozen=True)
class CollectionEvidenceSource:
    source_id: str
    path: str
    asset_id: str
    objective_scale: float = 1.0
    energy_cost_wh: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CollectionEvidenceSource:
        return cls(
            source_id=value.get("source_id", ""),
            path=value.get("path", ""),
            asset_id=value.get("asset_id", ""),
            objective_scale=value.get("objective_scale", 1.0),
            energy_cost_wh=value.get("energy_cost_wh", 0.0),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _source_id(self.source_id, "collection source_id"))
        object.__setattr__(self, "path", _required(self.path, "collection path"))
        object.__setattr__(self, "asset_id", _required(self.asset_id, "collection asset_id"))
        scale = _finite(self.objective_scale, "objective_scale")
        energy = _finite(self.energy_cost_wh, "energy_cost_wh")
        if scale < 0.0 or energy < 0.0:
            raise MissionSchedulingError("Collection objective scale and energy cost must be nonnegative.")
        object.__setattr__(self, "objective_scale", scale)
        object.__setattr__(self, "energy_cost_wh", energy)


@dataclass(frozen=True)
class LinkEvidenceSource:
    source_id: str
    path: str
    asset_id: str
    station_asset_id: str
    station_id: str
    energy_cost_wh: float = 0.0
    pointing_unit_eci: tuple[float, float, float] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LinkEvidenceSource:
        return cls(
            source_id=value.get("source_id", ""),
            path=value.get("path", ""),
            asset_id=value.get("asset_id", ""),
            station_asset_id=value.get("station_asset_id", ""),
            station_id=value.get("station_id", value.get("station_asset_id", "")),
            energy_cost_wh=value.get("energy_cost_wh", 0.0),
            pointing_unit_eci=_normalized_pointing(
                value.get("pointing_unit_eci"), "link pointing_unit_eci"
            ),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _source_id(self.source_id, "link source_id"))
        for field in ("path", "asset_id", "station_asset_id", "station_id"):
            object.__setattr__(self, field, _required(getattr(self, field), f"link {field}"))
        energy = _finite(self.energy_cost_wh, "energy_cost_wh")
        if energy < 0.0:
            raise MissionSchedulingError("Link energy_cost_wh must be nonnegative.")
        object.__setattr__(self, "energy_cost_wh", energy)
        object.__setattr__(
            self,
            "pointing_unit_eci",
            _normalized_pointing(self.pointing_unit_eci, "link pointing_unit_eci"),
        )


@dataclass(frozen=True)
class MissionSchedulingSourcePlan:
    analysis_id: str
    epoch_jd_utc: float
    horizon_start_s: float
    horizon_end_s: float
    assets: tuple[AssetScheduleConstraints, ...]
    collection_sources: tuple[CollectionEvidenceSource, ...]
    link_sources: tuple[LinkEvidenceSource, ...]
    require_observation_delivery_by_horizon: bool = True
    minimum_selected_observations: int = 1
    maximum_candidates: int = MAX_PUBLIC_MISSION_OPPORTUNITIES
    schema_version: str = MISSION_SCHEDULING_SOURCE_PLAN_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MissionSchedulingSourcePlan:
        return cls(
            schema_version=str(
                value.get("schema_version", MISSION_SCHEDULING_SOURCE_PLAN_SCHEMA)
            ).strip(),
            analysis_id=value.get("analysis_id", ""),
            epoch_jd_utc=value.get("epoch_jd_utc", float("nan")),
            horizon_start_s=value.get("horizon_start_s", float("nan")),
            horizon_end_s=value.get("horizon_end_s", float("nan")),
            assets=tuple(AssetScheduleConstraints.from_mapping(item) for item in value.get("assets", ())),
            collection_sources=tuple(
                CollectionEvidenceSource.from_mapping(item)
                for item in value.get("collection_sources", ())
            ),
            link_sources=tuple(
                LinkEvidenceSource.from_mapping(item) for item in value.get("link_sources", ())
            ),
            require_observation_delivery_by_horizon=value.get(
                "require_observation_delivery_by_horizon", True
            ),
            minimum_selected_observations=value.get("minimum_selected_observations", 1),
            maximum_candidates=value.get("maximum_candidates", MAX_PUBLIC_MISSION_OPPORTUNITIES),
        )

    def __post_init__(self) -> None:
        if self.schema_version != MISSION_SCHEDULING_SOURCE_PLAN_SCHEMA:
            raise MissionSchedulingError(f"Unsupported source-plan schema {self.schema_version!r}.")
        object.__setattr__(self, "analysis_id", _required(self.analysis_id, "analysis_id"))
        for field in ("epoch_jd_utc", "horizon_start_s", "horizon_end_s"):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if self.horizon_end_s <= self.horizon_start_s:
            raise MissionSchedulingError("Source-plan horizon end must be after its start.")
        if not self.assets or not self.collection_sources or not self.link_sources:
            raise MissionSchedulingError("Source plans require assets, collection sources, and link sources.")
        if not isinstance(self.require_observation_delivery_by_horizon, bool):
            raise MissionSchedulingError("require_observation_delivery_by_horizon must be boolean.")
        identifiers = [item.source_id for item in (*self.collection_sources, *self.link_sources)]
        if len({identifier.casefold() for identifier in identifiers}) != len(identifiers):
            raise MissionSchedulingError(
                "Source IDs must be case-insensitively unique across collection and link products."
            )
        asset_ids = {item.asset_id for item in self.assets}
        if len(asset_ids) != len(self.assets):
            raise MissionSchedulingError("Source-plan asset IDs must be unique.")
        for source in (*self.collection_sources, *self.link_sources):
            if source.asset_id not in asset_ids:
                raise MissionSchedulingError(
                    f"Source {source.source_id!r} names unknown scheduling asset {source.asset_id!r}."
                )
        slew_assets = {
            item.asset_id for item in self.assets if item.maximum_slew_rate_rad_s is not None
        }
        for source in self.link_sources:
            if source.asset_id in slew_assets and source.pointing_unit_eci is None:
                raise MissionSchedulingError(
                    f"Link source {source.source_id!r} requires explicit pointing_unit_eci because asset "
                    f"{source.asset_id!r} enables slew constraints."
                )
        if isinstance(self.maximum_candidates, bool) or int(self.maximum_candidates) != self.maximum_candidates:
            raise MissionSchedulingError("maximum_candidates must be an integer.")
        if not 1 <= int(self.maximum_candidates) <= MAX_PUBLIC_MISSION_OPPORTUNITIES:
            raise MissionSchedulingError(
                f"maximum_candidates must lie within [1, {MAX_PUBLIC_MISSION_OPPORTUNITIES}]."
            )
        object.__setattr__(self, "maximum_candidates", int(self.maximum_candidates))
        if (
            isinstance(self.minimum_selected_observations, bool)
            or int(self.minimum_selected_observations) != self.minimum_selected_observations
            or int(self.minimum_selected_observations) < 0
        ):
            raise MissionSchedulingError("minimum_selected_observations must be a nonnegative integer.")
        object.__setattr__(self, "minimum_selected_observations", int(self.minimum_selected_observations))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["assets"] = sorted(value["assets"], key=lambda item: item["asset_id"])
        value["collection_sources"] = sorted(
            value["collection_sources"], key=lambda item: item["source_id"]
        )
        value["link_sources"] = sorted(value["link_sources"], key=lambda item: item["source_id"])
        return value


@dataclass(frozen=True)
class VerifiedSourceProduct:
    source_id: str
    kind: str
    source_product_sha256: str
    upstream_semantic_sha256: str
    opportunities: tuple[MissionOpportunity, ...]
    files: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True)
class SourceBuiltMissionSchedule:
    source_plan: MissionSchedulingSourcePlan
    problem: MissionSchedulingProblem
    result: MissionSchedulingResult
    sources: tuple[VerifiedSourceProduct, ...]


@dataclass(frozen=True)
class SourceBuiltMissionArtifacts:
    output_dir: Path
    manifest_json: Path
    source_plan_json: Path
    source_products_dir: Path
    schedule_artifacts: MissionSchedulingArtifacts


def _resolve(base_dir: Path, authored: str) -> Path:
    path = Path(authored).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def _require_epoch(actual: Any, expected: float, source_id: str) -> None:
    value = _finite(actual, f"{source_id} epoch_jd_utc")
    if abs(value - expected) > 1.0e-12:
        raise MissionSchedulingError(
            f"Source {source_id!r} epoch {value} does not match source-plan epoch {expected}."
        )


def _verify_collection_source(
    plan: MissionSchedulingSourcePlan,
    source: CollectionEvidenceSource,
    base_dir: Path,
) -> VerifiedSourceProduct:
    path = _resolve(base_dir, source.path)
    evidence, content = _read_json_object(path, "collection evidence")
    if evidence.get("schema_version") != COLLECTION_OPPORTUNITY_EVIDENCE_SCHEMA:
        raise MissionSchedulingError(f"Collection source {source.source_id!r} has an unsupported schema.")
    if evidence.get("status") != "completed":
        raise MissionSchedulingError(f"Collection source {source.source_id!r} is not completed.")
    problem_hash = _valid_digest(evidence.get("problem_sha256"), "collection problem_sha256")
    frame = evidence.get("frame_time_provenance")
    if not isinstance(frame, dict):
        raise MissionSchedulingError(f"Collection source {source.source_id!r} lacks frame/time provenance.")
    _require_epoch(frame.get("jd_utc_start"), plan.epoch_jd_utc, source.source_id)
    samples = evidence.get("sample_ledger")
    if not isinstance(samples, list) or not samples:
        raise MissionSchedulingError(f"Collection source {source.source_id!r} lacks a sample ledger.")
    sample_times = [_finite(item.get("time_s"), "collection sample time_s") for item in samples]
    if any(second <= first for first, second in zip(sample_times, sample_times[1:])):
        raise MissionSchedulingError(f"Collection source {source.source_id!r} sample times are not increasing.")
    if sample_times[0] > plan.horizon_start_s + _EPS or sample_times[-1] < plan.horizon_end_s - _EPS:
        raise MissionSchedulingError(
            f"Collection source {source.source_id!r} does not cover the scheduling horizon."
        )
    candidates_raw = evidence.get("opportunity_candidates")
    tasks_raw = evidence.get("task_opportunities")
    if not isinstance(candidates_raw, list) or not isinstance(tasks_raw, list):
        raise MissionSchedulingError(f"Collection source {source.source_id!r} lacks opportunity ledgers.")
    accepted: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates_raw:
        if not isinstance(candidate, dict):
            raise MissionSchedulingError("Collection opportunity candidates must be JSON objects.")
        identifier = _required(candidate.get("opportunity_id"), "collection opportunity_id")
        if identifier in accepted or sum(
            1 for item in candidates_raw if isinstance(item, dict) and item.get("opportunity_id") == identifier
        ) != 1:
            raise MissionSchedulingError("Collection candidate opportunity IDs must be unique.")
        if candidate.get("accepted") is True:
            screen = candidate.get("resource_screen")
            if not isinstance(screen, dict) or screen.get("enabled") is not False:
                raise MissionSchedulingError(
                    f"Collection source {source.source_id!r} must disable its independent resource screen; "
                    "the multi-asset scheduler owns the shared resource ledger."
                )
            accepted[identifier] = candidate
    if len(tasks_raw) != len(accepted):
        raise MissionSchedulingError("Collection accepted-candidate and task-opportunity counts differ.")
    summary = evidence.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("accepted_opportunity_count") != len(tasks_raw)
    ):
        raise MissionSchedulingError("Collection summary accepted count differs from its task ledger.")
    product_hash = _digest_bytes(content)
    opportunities: list[MissionOpportunity] = []
    seen: set[str] = set()
    for task in tasks_raw:
        if not isinstance(task, dict):
            raise MissionSchedulingError("Collection task opportunities must be JSON objects.")
        identifier = _required(task.get("opportunity_id"), "collection task opportunity_id")
        if identifier in seen or identifier not in accepted:
            raise MissionSchedulingError("Collection task IDs must uniquely match accepted candidates.")
        seen.add(identifier)
        candidate = accepted[identifier]
        if task.get("kind") != "observation" or task.get("asset_id") != source.asset_id:
            raise MissionSchedulingError(f"Collection task {identifier!r} has incompatible kind or asset identity.")
        if _finite(task.get("energy_cost_wh"), "collection task energy_cost_wh") != 0.0:
            raise MissionSchedulingError("Collection task energy must remain zero before source-plan assignment.")
        if _valid_digest(task.get("source_product_sha256"), "task source_product_sha256") != problem_hash:
            raise MissionSchedulingError(f"Collection task {identifier!r} is not bound to the evidence problem.")
        comparisons = (
            (task.get("start_s"), candidate.get("collection_start_s"), "start"),
            (task.get("end_s"), candidate.get("collection_end_s"), "end"),
            (task.get("storage_delta_bytes"), candidate.get("generated_data_bytes"), "data"),
            (task.get("objective_value"), candidate.get("collection_duration_s"), "objective"),
        )
        if any(abs(_finite(left, label) - _finite(right, label)) > _EPS for left, right, label in comparisons):
            raise MissionSchedulingError(f"Collection task {identifier!r} differs from its accepted candidate ledger.")
        opportunities.append(
            MissionOpportunity(
                opportunity_id=f"{source.source_id}:{identifier}",
                source_product_sha256=product_hash,
                asset_id=source.asset_id,
                kind="observation",
                start_s=task["start_s"],
                end_s=task["end_s"],
                objective_value=float(task["objective_value"]) * source.objective_scale,
                energy_cost_wh=source.energy_cost_wh,
                data_volume_bytes=task["storage_delta_bytes"],
                pointing_unit_eci=task.get("pointing_unit_eci"),
                target_id=task.get("target_id"),
            )
        )
    if not opportunities:
        raise MissionSchedulingError(f"Collection source {source.source_id!r} contains no accepted opportunities.")
    return VerifiedSourceProduct(
        source_id=source.source_id,
        kind="collection",
        source_product_sha256=product_hash,
        upstream_semantic_sha256=problem_hash,
        opportunities=tuple(opportunities),
        files=(("collection_evidence.json", content),),
    )


def _verified_link_files(directory: Path, manifest: Mapping[str, Any]) -> tuple[tuple[str, bytes], ...]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise MissionSchedulingError("Directed-link manifest requires artifact receipts.")
    files: list[tuple[str, bytes]] = []
    manifest_path = directory / "link_analysis_manifest.json"
    files.append((manifest_path.name, manifest_path.read_bytes()))
    for name, receipt in sorted(artifacts.items()):
        relative = Path(str(name))
        if (
            len(relative.parts) != 1
            or relative.name in {"", ".", "..", "link_analysis_manifest.json"}
            or not isinstance(receipt, dict)
        ):
            raise MissionSchedulingError("Directed-link artifact receipts must use simple filenames.")
        try:
            content = (directory / relative).read_bytes()
        except OSError as exc:
            raise MissionSchedulingError(f"Could not read directed-link artifact {relative}: {exc}") from exc
        if _digest_bytes(content) != _valid_digest(receipt.get("sha256"), f"{relative} sha256"):
            raise MissionSchedulingError(f"Directed-link artifact receipt mismatch for {relative}.")
        files.append((relative.name, content))
    required = {
        "link_evidence_packet.json",
        "link_intervals.csv",
        "link_samples.csv",
        "link_summary.json",
        "link_transitions.json",
    }
    if not required.issubset({name for name, _ in files}):
        raise MissionSchedulingError("Directed-link evidence is missing required semantic artifacts.")
    return tuple(files)


def _csv_boolean(value: Any, field: str) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized not in {"true", "false"}:
        raise MissionSchedulingError(f"{field} must be true or false.")
    return normalized == "true"


def _verify_link_source(
    plan: MissionSchedulingSourcePlan,
    source: LinkEvidenceSource,
    base_dir: Path,
) -> VerifiedSourceProduct:
    directory = _resolve(base_dir, source.path)
    manifest, _ = _read_json_object(directory / "link_analysis_manifest.json", "directed-link manifest")
    if manifest.get("contract_version") != DIRECTED_LINK_CONTRACT_VERSION or manifest.get("status") != "complete":
        raise MissionSchedulingError(f"Link source {source.source_id!r} is not a completed supported product.")
    semantic_hash = _valid_digest(manifest.get("semantic_sha256"), "link semantic_sha256")
    _valid_digest(manifest.get("input_evidence_sha256"), "link input_evidence_sha256")
    frame = manifest.get("frame")
    if not isinstance(frame, dict):
        raise MissionSchedulingError(f"Link source {source.source_id!r} lacks frame metadata.")
    _require_epoch(frame.get("jd_utc_start"), plan.epoch_jd_utc, source.source_id)
    config = manifest.get("normalized_config")
    if not isinstance(config, dict):
        raise MissionSchedulingError(f"Link source {source.source_id!r} lacks normalized_config.")
    if (
        config.get("analysis_id") != manifest.get("analysis_id")
        or config.get("link_id") != manifest.get("link_id")
    ):
        raise MissionSchedulingError("Directed-link normalized identity differs from its manifest.")
    tx_terminal = config.get("tx_terminal")
    rx_terminal = config.get("rx_terminal")
    if not isinstance(tx_terminal, dict) or not isinstance(rx_terminal, dict):
        raise MissionSchedulingError("Directed-link normalized terminals must be JSON objects.")
    if (
        tx_terminal.get("asset_id") != source.asset_id
        or rx_terminal.get("asset_id") != source.station_asset_id
    ):
        raise MissionSchedulingError(f"Link source {source.source_id!r} endpoint identities do not match the plan.")
    if rx_terminal.get("parent_frame") != "enu":
        raise MissionSchedulingError(f"Link source {source.source_id!r} station endpoint is not fixed-site ENU.")
    if tx_terminal.get("parent_frame") != "body":
        raise MissionSchedulingError(f"Link source {source.source_id!r} spacecraft endpoint is not body-frame.")
    files = _verified_link_files(directory, manifest)
    file_map = dict(files)
    try:
        packet = json.loads(file_map["link_evidence_packet.json"].decode("utf-8"))
        summary = json.loads(file_map["link_summary.json"].decode("utf-8"))
        transitions_packet = json.loads(file_map["link_transitions.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MissionSchedulingError("Directed-link semantic artifact is missing or invalid.") from exc
    if (
        not isinstance(packet, dict)
        or packet.get("semantic_sha256") != semantic_hash
        or packet.get("input_evidence_sha256") != manifest.get("input_evidence_sha256")
        or packet.get("analysis_id") != manifest.get("analysis_id")
        or packet.get("link_id") != manifest.get("link_id")
        or packet.get("summary") != summary
    ):
        raise MissionSchedulingError("Directed-link evidence packet does not match its manifest.")
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "complete"
        or summary.get("analysis_id") != manifest.get("analysis_id")
        or summary.get("link_id") != manifest.get("link_id")
    ):
        raise MissionSchedulingError("Directed-link summary is not complete.")
    direction = summary.get("direction")
    if (
        not isinstance(direction, dict)
        or direction.get("tx_asset_id") != source.asset_id
        or direction.get("rx_asset_id") != source.station_asset_id
    ):
        raise MissionSchedulingError("Directed-link summary direction differs from the declared endpoints.")
    if (
        not isinstance(transitions_packet, dict)
        or transitions_packet.get("schema") != "oel.directed-link-transitions.v1"
        or transitions_packet.get("analysis_id") != manifest.get("analysis_id")
        or transitions_packet.get("link_id") != manifest.get("link_id")
        or not isinstance(transitions_packet.get("transitions"), list)
        or any(not isinstance(item, dict) for item in transitions_packet["transitions"])
    ):
        raise MissionSchedulingError("Directed-link transition evidence does not match its manifest.")
    data_rate = _finite(config.get("data_rate_bps"), "link data_rate_bps")
    if data_rate <= 0.0:
        raise MissionSchedulingError("Directed-link data_rate_bps must be positive.")
    rows = list(csv.DictReader(file_map["link_intervals.csv"].decode("utf-8").splitlines()))
    if summary.get("interval_count") != len(rows):
        raise MissionSchedulingError("Directed-link summary interval count differs from its CSV ledger.")
    sample_rows = list(csv.DictReader(file_map["link_samples.csv"].decode("utf-8").splitlines()))
    if not sample_rows:
        raise MissionSchedulingError("Directed-link sample evidence is empty.")
    try:
        recomputed_semantic_hash = recompute_directed_link_semantic_sha256(
            normalized_config=config,
            input_evidence_sha256=manifest.get("input_evidence_sha256", ""),
            time_s=[_finite(row.get("time_s"), "link sample time_s") for row in sample_rows],
            range_km=[_finite(row.get("range_km"), "link sample range_km") for row in sample_rows],
            margin_db=[_finite(row.get("margin_db"), "link sample margin_db") for row in sample_rows],
            available=[
                _csv_boolean(row.get("available"), "link sample available") for row in sample_rows
            ],
            primary_reason=[_required(row.get("primary_reason"), "link sample primary_reason") for row in sample_rows],
            intervals=rows,
            transitions=transitions_packet["transitions"],
            refinement_provider_id=manifest.get("refinement_provider_id"),
        )
    except (TypeError, ValueError) as exc:
        raise MissionSchedulingError(f"Directed-link semantic evidence is invalid: {exc}") from exc
    if recomputed_semantic_hash != semantic_hash:
        raise MissionSchedulingError("Directed-link semantic SHA-256 does not match retained evidence.")
    opportunities: list[MissionOpportunity] = []
    interval_ids: set[str] = set()
    for row in rows:
        start = _finite(row.get("start_s"), "link interval start_s")
        end = _finite(row.get("end_s"), "link interval end_s")
        duration = _finite(row.get("duration_s"), "link interval duration_s")
        bits = _finite(row.get("estimated_delivered_data_bits"), "estimated_delivered_data_bits")
        if end <= start or abs(duration - (end - start)) > _EPS:
            raise MissionSchedulingError("Directed-link interval timing is inconsistent.")
        if start < plan.horizon_start_s - _EPS or end > plan.horizon_end_s + _EPS:
            raise MissionSchedulingError(f"Link source {source.source_id!r} contains an interval outside the horizon.")
        if abs(bits - duration * data_rate) > max(_EPS, abs(bits) * 1.0e-12):
            raise MissionSchedulingError("Directed-link interval capacity differs from duration times data rate.")
        interval_index = _required(row.get("interval_index"), "link interval_index")
        if interval_index in interval_ids:
            raise MissionSchedulingError("Directed-link interval identifiers must be unique.")
        interval_ids.add(interval_index)
        opportunities.append(
            MissionOpportunity(
                opportunity_id=f"{source.source_id}:interval:{interval_index}",
                source_product_sha256=semantic_hash,
                asset_id=source.asset_id,
                kind="downlink",
                start_s=start,
                end_s=end,
                objective_value=0.0,
                energy_cost_wh=source.energy_cost_wh,
                downlink_capacity_bytes=bits / 8.0,
                station_id=source.station_id,
                pointing_unit_eci=source.pointing_unit_eci,
            )
        )
    if not opportunities:
        raise MissionSchedulingError(f"Link source {source.source_id!r} contains no available intervals.")
    return VerifiedSourceProduct(
        source_id=source.source_id,
        kind="link",
        source_product_sha256=semantic_hash,
        upstream_semantic_sha256=semantic_hash,
        opportunities=tuple(opportunities),
        files=files,
    )


def build_mission_scheduling_problem_from_sources(
    source_plan: MissionSchedulingSourcePlan | Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> SourceBuiltMissionSchedule:
    """Verify OEL products, construct the exact problem, and solve it."""

    plan = (
        source_plan
        if isinstance(source_plan, MissionSchedulingSourcePlan)
        else MissionSchedulingSourcePlan.from_mapping(source_plan)
    )
    root = Path(base_dir).expanduser().resolve()
    sources = tuple(
        [
            _verify_collection_source(plan, source, root)
            for source in sorted(plan.collection_sources, key=lambda item: item.source_id)
        ]
        + [
            _verify_link_source(plan, source, root)
            for source in sorted(plan.link_sources, key=lambda item: item.source_id)
        ]
    )
    opportunities = tuple(item for source in sources for item in source.opportunities)
    problem = MissionSchedulingProblem(
        analysis_id=plan.analysis_id,
        horizon_start_s=plan.horizon_start_s,
        horizon_end_s=plan.horizon_end_s,
        assets=plan.assets,
        opportunities=opportunities,
        require_observation_delivery_by_horizon=plan.require_observation_delivery_by_horizon,
        minimum_selected_observations=plan.minimum_selected_observations,
        maximum_candidates=plan.maximum_candidates,
    )
    return SourceBuiltMissionSchedule(
        source_plan=plan,
        problem=problem,
        result=solve_mission_schedule(problem),
        sources=sources,
    )


def _retained_plan(build: SourceBuiltMissionSchedule) -> MissionSchedulingSourcePlan:
    collections = tuple(
        replace(item, path=f"source_products/{item.source_id}/collection_evidence.json")
        for item in build.source_plan.collection_sources
    )
    links = tuple(
        replace(item, path=f"source_products/{item.source_id}")
        for item in build.source_plan.link_sources
    )
    return replace(build.source_plan, collection_sources=collections, link_sources=links)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_source_built_mission_schedule(
    build: SourceBuiltMissionSchedule,
    output_dir: str | Path,
) -> SourceBuiltMissionArtifacts:
    """Retain verified source products and write a self-contained schedule packet."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise MissionSchedulingError(f"output_dir must be absent or empty; refusing to mix evidence in {destination}.")
    destination.mkdir(parents=True, exist_ok=True)
    products_dir = destination / "source_products"
    products_dir.mkdir()
    source_records: list[dict[str, Any]] = []
    for source in build.sources:
        source_dir = products_dir / source.source_id
        source_dir.mkdir()
        receipts: list[dict[str, Any]] = []
        for name, content in source.files:
            path = source_dir / name
            path.write_bytes(content)
            receipts.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "bytes": len(content),
                    "sha256": _digest_bytes(content),
                }
            )
        source_records.append(
            {
                "source_id": source.source_id,
                "kind": source.kind,
                "source_product_sha256": source.source_product_sha256,
                "upstream_semantic_sha256": source.upstream_semantic_sha256,
                "extracted_opportunity_ids": [item.opportunity_id for item in source.opportunities],
                "artifacts": receipts,
            }
        )
    retained_plan = _retained_plan(build)
    plan_path = destination / "normalized_source_plan.json"
    _write_json(plan_path, retained_plan.to_dict())
    replayed_build = build_mission_scheduling_problem_from_sources(retained_plan, base_dir=destination)
    if replayed_build.problem.to_dict() != build.problem.to_dict():
        raise MissionSchedulingError("Retained source products did not reproduce the normalized scheduling problem.")
    schedule_artifacts = write_mission_scheduling_artifacts(build.result, destination / "schedule")
    manifest_path = destination / "mission_schedule_source_manifest.json"
    plan_content = plan_path.read_bytes()
    schedule_manifest_content = schedule_artifacts.manifest_json.read_bytes()
    _write_json(
        manifest_path,
        {
            "schema_version": MISSION_SCHEDULING_SOURCE_EVIDENCE_SCHEMA,
            "analysis_id": build.source_plan.analysis_id,
            "status": build.result.status,
            "source_plan_semantic_sha256": _canonical_digest(retained_plan.to_dict()),
            "input_semantic_sha256": build.result.input_semantic_sha256,
            "schedule_semantic_sha256": build.result.schedule_semantic_sha256,
            "sources": sorted(source_records, key=lambda item: item["source_id"]),
            "artifacts": [
                {
                    "path": plan_path.name,
                    "bytes": len(plan_content),
                    "sha256": _digest_bytes(plan_content),
                },
                {
                    "path": "schedule/mission_schedule_manifest.json",
                    "bytes": len(schedule_manifest_content),
                    "sha256": _digest_bytes(schedule_manifest_content),
                },
            ],
        },
    )
    return SourceBuiltMissionArtifacts(
        output_dir=destination,
        manifest_json=manifest_path,
        source_plan_json=plan_path,
        source_products_dir=products_dir,
        schedule_artifacts=schedule_artifacts,
    )


def build_solve_mission_schedule_from_sources(
    source_plan: MissionSchedulingSourcePlan | Mapping[str, Any],
    *,
    base_dir: str | Path,
    output_dir: str | Path,
) -> SourceBuiltMissionArtifacts:
    build = build_mission_scheduling_problem_from_sources(source_plan, base_dir=base_dir)
    return write_source_built_mission_schedule(build, output_dir)


def _retained_source_inventory(
    plan: MissionSchedulingSourcePlan,
    source_claims: list[dict[str, Any]],
    root: Path,
) -> None:
    expected_sources: dict[str, tuple[str, str]] = {}
    for source in plan.collection_sources:
        expected_path = f"source_products/{source.source_id}/collection_evidence.json"
        if source.path != expected_path:
            raise MissionSchedulingError(
                f"Collection source {source.source_id!r} must use its canonical retained path."
            )
        expected_sources[source.source_id] = ("collection", expected_path)
    for source in plan.link_sources:
        expected_path = f"source_products/{source.source_id}"
        if source.path != expected_path:
            raise MissionSchedulingError(
                f"Link source {source.source_id!r} must use its canonical retained path."
            )
        expected_sources[source.source_id] = ("link", expected_path)

    claims_by_id: dict[str, dict[str, Any]] = {}
    for claim in source_claims:
        source_id = _source_id(claim.get("source_id"), "source claim source_id")
        if source_id in claims_by_id:
            raise MissionSchedulingError("Source manifest contains duplicate source claims.")
        claims_by_id[source_id] = claim
    if set(claims_by_id) != set(expected_sources):
        raise MissionSchedulingError("Source manifest claims do not match the retained source plan.")

    for source_id, (expected_kind, retained_path) in expected_sources.items():
        claim = claims_by_id[source_id]
        if claim.get("kind") != expected_kind:
            raise MissionSchedulingError("Source manifest kind differs from the retained source plan.")
        receipts = claim.get("artifacts")
        if not isinstance(receipts, list) or any(not isinstance(item, dict) for item in receipts):
            raise MissionSchedulingError("Source manifest artifact receipts must be lists of objects.")
        received_paths = [str(item.get("path", "")) for item in receipts]
        if expected_kind == "collection":
            required_paths = {retained_path}
        else:
            manifest_path = root / retained_path / "link_analysis_manifest.json"
            retained_manifest, _ = _read_json_object(
                manifest_path, f"retained link manifest for {source_id}"
            )
            artifacts = retained_manifest.get("artifacts")
            if not isinstance(artifacts, dict) or not artifacts:
                raise MissionSchedulingError("Retained directed-link manifest requires artifact receipts.")
            artifact_names: set[str] = set()
            for name in artifacts:
                relative = Path(str(name))
                if len(relative.parts) != 1 or relative.name in {
                    "",
                    ".",
                    "..",
                    "link_analysis_manifest.json",
                }:
                    raise MissionSchedulingError(
                        "Retained directed-link artifact receipts must use simple filenames."
                    )
                artifact_names.add(relative.name)
            required_paths = {
                f"{retained_path}/link_analysis_manifest.json",
                *(f"{retained_path}/{name}" for name in artifact_names),
            }
        if len(received_paths) != len(set(received_paths)) or set(received_paths) != required_paths:
            raise MissionSchedulingError(
                f"Source {source_id!r} manifest does not contain the exact required artifact inventory."
            )


def verify_source_built_mission_schedule(output_dir: str | Path) -> dict[str, Any]:
    """Verify retained source receipts, rebuild the problem, and replay the optimum."""

    root = Path(output_dir).expanduser().resolve()
    manifest, _ = _read_json_object(
        root / "mission_schedule_source_manifest.json", "mission schedule source manifest"
    )
    if manifest.get("schema_version") != MISSION_SCHEDULING_SOURCE_EVIDENCE_SCHEMA:
        raise MissionSchedulingError("Unsupported mission schedule source manifest schema.")
    outer_artifacts = manifest.get("artifacts")
    if not isinstance(outer_artifacts, list) or len(outer_artifacts) != 2 or {
        item.get("path") for item in outer_artifacts if isinstance(item, dict)
    } != {"normalized_source_plan.json", "schedule/mission_schedule_manifest.json"}:
        raise MissionSchedulingError("Source manifest does not contain the exact required artifact set.")
    source_claims = manifest.get("sources")
    if not isinstance(source_claims, list) or any(not isinstance(item, dict) for item in source_claims):
        raise MissionSchedulingError("Source manifest requires source-product claims.")
    for group in (outer_artifacts, *[item.get("artifacts") for item in source_claims]):
        if not isinstance(group, list):
            raise MissionSchedulingError("Source manifest artifact receipts must be lists.")
        for receipt in group:
            if not isinstance(receipt, dict):
                raise MissionSchedulingError("Source artifact receipts must be JSON objects.")
            relative = Path(str(receipt.get("path", "")))
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
                content = path.read_bytes()
            except (ValueError, OSError) as exc:
                raise MissionSchedulingError(f"Invalid retained source artifact {relative}: {exc}") from exc
            if len(content) != receipt.get("bytes") or _digest_bytes(content) != receipt.get("sha256"):
                raise MissionSchedulingError(f"Retained source artifact receipt mismatch for {relative}.")
    plan_payload, plan_content = _read_json_object(root / "normalized_source_plan.json", "source plan")
    plan = MissionSchedulingSourcePlan.from_mapping(plan_payload)
    if _canonical_digest(plan.to_dict()) != manifest.get("source_plan_semantic_sha256"):
        raise MissionSchedulingError("Retained source plan semantic SHA-256 mismatch.")
    plan_receipt = next(
        item for item in outer_artifacts if item["path"] == "normalized_source_plan.json"
    )
    if _digest_bytes(plan_content) != plan_receipt["sha256"]:
        raise MissionSchedulingError("Retained source plan file receipt mismatch.")
    _retained_source_inventory(plan, source_claims, root)
    rebuilt = build_mission_scheduling_problem_from_sources(plan, base_dir=root)
    expected_claims = sorted(
        (
            source.source_id,
            source.kind,
            source.source_product_sha256,
            source.upstream_semantic_sha256,
            tuple(item.opportunity_id for item in source.opportunities),
        )
        for source in rebuilt.sources
    )
    actual_claims: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    for item in source_claims:
        opportunity_ids = item.get("extracted_opportunity_ids")
        if not isinstance(opportunity_ids, list):
            raise MissionSchedulingError("Source-product opportunity claims must be lists.")
        actual_claims.append(
            (
                _source_id(item.get("source_id"), "source claim source_id"),
                _required(item.get("kind"), "source claim kind"),
                _valid_digest(item.get("source_product_sha256"), "source claim product SHA-256"),
                _valid_digest(
                    item.get("upstream_semantic_sha256"), "source claim upstream SHA-256"
                ),
                tuple(_required(value, "source claim opportunity_id") for value in opportunity_ids),
            )
        )
    actual_claims.sort()
    if actual_claims != expected_claims:
        raise MissionSchedulingError("Source-product manifest claims differ from retained evidence.")
    scheduled_problem, _ = _read_json_object(
        root / "schedule/normalized_problem.json", "scheduled normalized problem"
    )
    if _canonical_digest(rebuilt.problem.to_dict()) != _canonical_digest(scheduled_problem):
        raise MissionSchedulingError("Retained source products do not reproduce the scheduled problem.")
    replay = verify_mission_scheduling_artifacts(root / "schedule")
    if (
        manifest.get("analysis_id") != replay["analysis_id"]
        or manifest.get("status") != rebuilt.result.status
        or manifest.get("input_semantic_sha256") != replay["input_semantic_sha256"]
        or manifest.get("schedule_semantic_sha256") != replay["schedule_semantic_sha256"]
    ):
        raise MissionSchedulingError("Source manifest claims differ from authoritative replay.")
    return {
        **replay,
        "source_status": "verified",
        "source_plan_semantic_sha256": manifest["source_plan_semantic_sha256"],
        "source_count": len(rebuilt.sources),
    }


__all__ = [
    "MISSION_SCHEDULING_SOURCE_EVIDENCE_SCHEMA",
    "MISSION_SCHEDULING_SOURCE_PLAN_SCHEMA",
    "CollectionEvidenceSource",
    "LinkEvidenceSource",
    "MissionSchedulingSourcePlan",
    "SourceBuiltMissionArtifacts",
    "SourceBuiltMissionSchedule",
    "VerifiedSourceProduct",
    "build_mission_scheduling_problem_from_sources",
    "build_solve_mission_schedule_from_sources",
    "verify_source_built_mission_schedule",
    "write_source_built_mission_schedule",
]

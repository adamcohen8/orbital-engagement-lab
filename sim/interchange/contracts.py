from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

PRODUCT_ENVELOPE_SCHEMA_ID = "oel-product-envelope-v1"
PRODUCT_ENVELOPE_SCHEMA_VERSION = 1
HANDOFF_MANIFEST_SCHEMA_ID = "oel-handoff-manifest-v1"
HANDOFF_MANIFEST_SCHEMA_VERSION = 1


class QualityDisposition(str, Enum):
    ACCEPTED = "accepted"
    REVIEW_REQUIRED = "review_required"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class IntegrityStatus(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    NOT_EVALUATED = "not_evaluated"


class AgeStatus(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    NOT_EVALUATED = "not_evaluated"
    NOT_APPLICABLE = "not_applicable"


class DataScope(str, Enum):
    PUBLIC = "public"
    PRIVATE_PRO = "private_pro"
    CUSTOMER_RESTRICTED = "customer_restricted"
    GOVERNED_RESTRICTED = "governed_restricted"


QUALITY_DISPOSITIONS = tuple(item.value for item in QualityDisposition)
INTEGRITY_STATUSES = tuple(item.value for item in IntegrityStatus)
AGE_STATUSES = tuple(item.value for item in AgeStatus)
DATA_SCOPES = tuple(item.value for item in DataScope)


@dataclass(frozen=True)
class ProductEnvelope:
    """Immutable value wrapper for Product Envelope v1.

    Nested mappings are copied on ingress and egress so callers cannot mutate
    the semantic document behind a previously computed product identity.
    """

    schema_id: str
    schema_version: int
    product_kind: str
    product_id: str
    created_utc: str
    producer: dict[str, Any]
    payload: dict[str, Any]
    quality: dict[str, Any]
    freshness: dict[str, Any]
    provenance: dict[str, Any]
    data_markings: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProductEnvelope:
        return cls(
            schema_id=str(value.get("schema_id", "")),
            schema_version=int(value.get("schema_version", 0)),
            product_kind=str(value.get("product_kind", "")),
            product_id=str(value.get("product_id", "")),
            created_utc=str(value.get("created_utc", "")),
            producer=deepcopy(dict(value.get("producer", {}) or {})),
            payload=deepcopy(dict(value.get("payload", {}) or {})),
            quality=deepcopy(dict(value.get("quality", {}) or {})),
            freshness=deepcopy(dict(value.get("freshness", {}) or {})),
            provenance=deepcopy(dict(value.get("provenance", {}) or {})),
            data_markings=deepcopy(dict(value.get("data_markings", {}) or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "product_kind": self.product_kind,
            "product_id": self.product_id,
            "created_utc": self.created_utc,
            "producer": deepcopy(self.producer),
            "payload": deepcopy(self.payload),
            "quality": deepcopy(self.quality),
            "freshness": deepcopy(self.freshness),
            "provenance": deepcopy(self.provenance),
            "data_markings": deepcopy(self.data_markings),
        }


@dataclass(frozen=True)
class HandoffManifest:
    """Read-only value wrapper for Handoff Manifest v1."""

    document: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HandoffManifest:
        return cls(document=deepcopy(dict(value)))

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.document)

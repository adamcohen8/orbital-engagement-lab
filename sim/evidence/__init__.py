"""Generic OEL workflow-evidence envelopes.

Domain workflows retain their authoritative artifacts and schemas.  This
package writes a compact, common sidecar that humans, CI, reports, notebooks,
and optional integrations can inspect without redefining domain semantics.
"""

from sim.evidence.workflow import (
    artifact_reference,
    build_workflow_evidence,
    load_workflow_evidence,
    write_workflow_evidence,
)

__all__ = [
    "artifact_reference",
    "build_workflow_evidence",
    "load_workflow_evidence",
    "write_workflow_evidence",
]

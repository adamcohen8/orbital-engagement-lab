"""Static ownership map for the stable :mod:`sim.api` contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicApiFamily:
    name: str
    module: str
    capabilities: tuple[str, ...]
    facade: str = "sim.api"


PUBLIC_API_FAMILIES: tuple[PublicApiFamily, ...] = (
    PublicApiFamily("config", "sim.public_api.config", ("SimulationConfig",)),
    PublicApiFamily("snapshots", "sim.public_api.snapshots", ("SimulationSnapshot",)),
    PublicApiFamily("results", "sim.public_api.results", ("SimulationResult", "MetricStudyResult")),
    PublicApiFamily(
        "sessions",
        "sim.public_api.session",
        ("SimulationSession", "HostedSimulationSession", "TrustedSimulationSession"),
    ),
    PublicApiFamily(
        "workspaces",
        "sim.public_api.workspace",
        ("SimulationWorkspace", "HostedSimulationWorkspace", "TrustedSimulationWorkspace"),
    ),
    PublicApiFamily(
        "controller_adapters",
        "sim.public_api.controller_adapters",
        ("_coerce_controller_return", "_controller_object", "_mission_object"),
    ),
    PublicApiFamily("feature_routing", "sim.public_api.feature_routing", ("_require_private_workflow",)),
)

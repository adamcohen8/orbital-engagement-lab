from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from machine_learning.gym_env import (
    GymEnvConfig,
    MultiAgentEnvConfig,
    ObservationField,
    default_observation_fields_for_agent,
)
from sim.config import SimulationScenarioConfig


@dataclass(frozen=True)
class PolicyCard:
    policy_name: str
    policy_type: str
    generated_utc: str
    scenario_name: str
    training_envelope: dict[str, Any]
    observation_source: dict[str, Any]
    privileged_data: dict[str, Any]
    reward: dict[str, Any]
    safety_gates: dict[str, Any]
    ood_checks: dict[str, Any]
    provenance: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scenario_dict(scenario: SimulationScenarioConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(scenario, SimulationScenarioConfig):
        return scenario.to_dict()
    return dict(scenario)


def _field_dict(field: ObservationField) -> dict[str, Any]:
    return {"path": str(field.path), "scale": float(field.scale)}


def _field_source(path: str) -> str:
    root = str(path).split(".", 1)[0].strip().lower()
    if root == "truth":
        return "oracle_truth"
    if root == "belief":
        return "observer_belief"
    if root == "knowledge":
        return "observer_knowledge"
    if root == "metrics":
        return "simulator_metric"
    if root == "sampled_parameters":
        return "sampled_parameter"
    return "custom"


def classify_observation_fields(fields: tuple[ObservationField, ...]) -> dict[str, Any]:
    rows = [{"path": str(field.path), "source": _field_source(field.path), "scale": float(field.scale)} for field in fields]
    sources = sorted({str(row["source"]) for row in rows})
    truth_paths = [str(row["path"]) for row in rows if row["source"] == "oracle_truth"]
    simulator_metric_paths = [str(row["path"]) for row in rows if row["source"] == "simulator_metric"]
    privileged_paths = [*truth_paths, *simulator_metric_paths]
    if truth_paths:
        policy_label = "oracle_baseline"
    elif simulator_metric_paths:
        policy_label = "simulator_metric_baseline"
    elif sources and all(source in {"observer_belief", "observer_knowledge", "sampled_parameter"} for source in sources):
        policy_label = "observer_owned"
    else:
        policy_label = "mixed"
    return {
        "policy_label": policy_label,
        "sources": sources,
        "fields": rows,
        "privileged_paths": privileged_paths,
        "uses_privileged_data": bool(privileged_paths),
    }


def _callable_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        module = str(value.get("module", "") or "")
        attr = str(value.get("class_name", value.get("function", "")) or "")
        return f"{module}:{attr}" if attr else module
    if isinstance(value, str):
        return value
    return f"{value.__class__.__module__}.{value.__class__.__name__}"


def _git_provenance(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), check=False, capture_output=True, text=True
        )
        if commit.returncode == 0:
            out["commit"] = commit.stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=str(root), check=False, capture_output=True, text=True
        )
        if branch.returncode == 0:
            out["branch"] = branch.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=str(root), check=False, capture_output=True, text=True
        )
        if status.returncode == 0:
            rows = [line for line in status.stdout.splitlines() if line.strip()]
            out["dirty"] = bool(rows)
            out["status_short"] = rows
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def _scenario_envelope(scenario: dict[str, Any]) -> dict[str, Any]:
    sim = dict(scenario.get("simulator", {}) or {})
    dynamics = dict(sim.get("dynamics", {}) or {})
    orbit = dict(dynamics.get("orbit", {}) or {})
    attitude = dict(dynamics.get("attitude", {}) or {})
    return {
        "duration_s": sim.get("duration_s"),
        "dt_s": sim.get("dt_s"),
        "orbit_model": orbit.get("model"),
        "orbit_flags": {
            key: bool(orbit.get(key, False))
            for key in ("j2", "j3", "j4", "drag", "srp", "third_body_sun", "third_body_moon")
        },
        "attitude_enabled": bool(attitude.get("enabled", True)),
        "episode_variations": [],
    }


def build_policy_card(
    env_cfg: GymEnvConfig | MultiAgentEnvConfig,
    *,
    policy_name: str = "unnamed_policy",
    policy_type: str = "rl_policy",
    training_envelope: dict[str, Any] | None = None,
    reward: dict[str, Any] | None = None,
    safety_gates: dict[str, Any] | None = None,
    ood_checks: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> PolicyCard:
    scenario = _scenario_dict(env_cfg.scenario)
    scenario_name = str(scenario.get("scenario_name", "unnamed_scenario") or "unnamed_scenario")
    if isinstance(env_cfg, GymEnvConfig):
        fields_by_agent = {
            str(env_cfg.controlled_agent_id): tuple(env_cfg.observation_fields)
            or default_observation_fields_for_agent(str(env_cfg.controlled_agent_id))
        }
        reward_default = {
            "default": "RelativeDistanceReward",
            "configured": _callable_name(env_cfg.reward_fn),
            "termination": _callable_name(env_cfg.termination_fn),
        }
        variations = list(env_cfg.episode_variations or ())
        controlled_agents = [str(env_cfg.controlled_agent_id)]
    else:
        fields_by_agent = {
            str(agent_id): tuple(env_cfg.observation_fields_by_agent.get(agent_id, ()))
            or default_observation_fields_for_agent(str(agent_id))
            for agent_id in env_cfg.controlled_agent_ids
        }
        reward_default = {
            "default": "RelativeDistanceReward per controlled agent",
            "configured_by_agent": {
                str(agent_id): _callable_name(fn) for agent_id, fn in dict(env_cfg.reward_fns_by_agent or {}).items()
            },
            "termination": _callable_name(env_cfg.termination_fn),
        }
        variations = list(env_cfg.episode_variations or ())
        controlled_agents = [str(agent_id) for agent_id in env_cfg.controlled_agent_ids]

    observation_by_agent = {agent_id: classify_observation_fields(fields) for agent_id, fields in fields_by_agent.items()}
    privileged_paths = {
        agent_id: list(summary.get("privileged_paths", []) or []) for agent_id, summary in observation_by_agent.items()
    }
    uses_privileged = any(paths for paths in privileged_paths.values())
    label = "oracle_baseline" if any(s["policy_label"] == "oracle_baseline" for s in observation_by_agent.values()) else (
        "observer_owned"
        if all(s["policy_label"] == "observer_owned" for s in observation_by_agent.values())
        else "mixed"
    )
    envelope = _scenario_envelope(scenario)
    envelope["episode_variations"] = [getattr(v, "parameter_path", str(v)) for v in variations]
    envelope.update(dict(training_envelope or {}))

    repo_root = Path(__file__).resolve().parents[1]
    prov = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "generated_by": "machine_learning.policy_card.build_policy_card",
        "git": _git_provenance(repo_root),
    }
    prov.update(dict(provenance or {}))

    card_notes = list(notes or [])
    if uses_privileged:
        card_notes.append(
            "This policy card includes privileged observation paths. Treat associated policy performance as an oracle/simulator baseline, not an operational autonomy claim."
        )

    return PolicyCard(
        policy_name=str(policy_name),
        policy_type=str(policy_type),
        generated_utc=datetime.now(timezone.utc).isoformat(),
        scenario_name=scenario_name,
        training_envelope=envelope,
        observation_source={
            "contract": label,
            "controlled_agents": controlled_agents,
            "by_agent": observation_by_agent,
        },
        privileged_data={
            "uses_privileged_data": bool(uses_privileged),
            "privileged_paths_by_agent": privileged_paths,
            "oracle_baseline": bool(uses_privileged),
        },
        reward=dict(reward or reward_default),
        safety_gates=dict(
            safety_gates
            or {
                "earth_impact_check": "Gym wrapper terminates on Earth impact radius.",
                "max_steps": getattr(env_cfg, "max_steps", None),
                "custom_termination": reward_default.get("termination", ""),
            }
        ),
        ood_checks=dict(
            ood_checks
            or {
                "status": "not_provided",
                "required_before_operational_claim": [
                    "initial-condition sweep",
                    "sensor-noise/dropout sweep",
                    "actuator-limit sweep",
                    "force-model/fidelity sweep",
                ],
            }
        ),
        provenance=prov,
        notes=card_notes,
    )


def write_policy_card(card: PolicyCard, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".md":
        path.write_text(policy_card_markdown(card), encoding="utf-8")
    else:
        path.write_text(json.dumps(card.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def policy_card_markdown(card: PolicyCard) -> str:
    data = card.to_dict()
    lines = [
        f"# Policy Card: {card.policy_name}",
        "",
        f"- Policy Type: {card.policy_type}",
        f"- Scenario: {card.scenario_name}",
        f"- Generated UTC: {card.generated_utc}",
        f"- Observation Contract: {data['observation_source']['contract']}",
        f"- Uses Privileged Data: {data['privileged_data']['uses_privileged_data']}",
        f"- Oracle Baseline: {data['privileged_data']['oracle_baseline']}",
        "",
        "## Observation Source",
        "",
    ]
    for agent_id, summary in dict(data["observation_source"]["by_agent"]).items():
        lines.append(f"### {agent_id}")
        for row in list(summary.get("fields", []) or []):
            lines.append(f"- `{row['path']}` ({row['source']}, scale={row['scale']})")
        lines.append("")
    lines.extend(
        [
            "## Training Envelope",
            "",
            "```json",
            json.dumps(data["training_envelope"], indent=2, sort_keys=True),
            "```",
            "",
            "## Reward",
            "",
            "```json",
            json.dumps(data["reward"], indent=2, sort_keys=True),
            "```",
            "",
            "## Safety Gates",
            "",
            "```json",
            json.dumps(data["safety_gates"], indent=2, sort_keys=True),
            "```",
            "",
            "## OOD Checks",
            "",
            "```json",
            json.dumps(data["ood_checks"], indent=2, sort_keys=True),
            "```",
            "",
            "## Provenance",
            "",
            "```json",
            json.dumps(data["provenance"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    if card.notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in card.notes)
        lines.append("")
    return "\n".join(lines)

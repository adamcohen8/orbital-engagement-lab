# ruff: noqa: F401,I001
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.config import SimulationScenarioConfig, configured_objects, default_pair_object_ids
from sim.reporting.ai_endpoint_security import resolve_ai_endpoint
from sim.security import ConfigPathPolicy
from sim.utils.io import write_json


@dataclass(frozen=True)
class ReportPayloadAdapter:
    payload_kind: str
    default_prompt_profile: str
    summary_filenames: tuple[str, ...]
    can_load: Any
    source_brief: Any


DEFAULT_PROMPT_PROFILES = {
    "campaign_report": """Write an AI-assisted campaign report for an engineering user.

Return only Markdown with exactly these top-level sections: Executive Summary, Figure Walk-through, Inferences Based on the Data. Use simulator_context and config_summary to explain what was simulated. Use figure_manifest for the figures requested in the config and for generated figure artifacts. Use payload for deterministic results and statistics. Follow report_rules exactly.""",
    "commander_summary": """Write a concise commander-facing campaign report.

Return only Markdown with exactly these top-level sections: Executive Summary, Figure Walk-through, Inferences Based on the Data. In Executive Summary, summarize what was simulated, key outcomes, major risks, pass/fail probability, resource margins, and recommended next actions. In Figure Walk-through, discuss each requested/generated figure from figure_manifest. In Inferences Based on the Data, explain the key results using payload statistics. Follow report_rules exactly.""",
    "analyst_report": """Write a technical analysis report for an engineering review.

Return only Markdown with exactly these top-level sections: Executive Summary, Figure Walk-through, Inferences Based on the Data. In Executive Summary, summarize the scenario and results. In Figure Walk-through, discuss each requested/generated figure from figure_manifest. In Inferences Based on the Data, explain aggregate statistics, parameter drivers, notable outliers, uncertainty, and recommended follow-up analysis. Follow report_rules exactly.""",
    "adversarial_advantage_assessment": """Write an adversarial advantage assessment for a chaser-target orbital engagement.

Return only Markdown with exactly these top-level sections: Executive Summary, Figure Walk-through, Inferences Based on the Data. In Executive Summary, state whether the supplied deterministic evidence better supports the chaser, the target, neither side, or an indeterminate assessment in achieving its configured or implied objectives. Infer objectives only from scenario description, configured roles, controllers, gates, metrics, and supplied results; if objectives are ambiguous, say so explicitly. In Figure Walk-through, discuss each requested/generated figure from figure_manifest and explain what it contributes to the advantage assessment without claiming visual inspection when pixels are unavailable. In Inferences Based on the Data, compare evidence for each side using pass/fail rates, closest approach or keepout outcomes, time and delta-v margins, control authority, knowledge/estimation evidence, failure modes, parameter drivers, and uncertainty. Identify what scenario changes or missing evidence could reverse the assessment. Follow report_rules exactly.""",
    "sensitivity_insights": """Write a sensitivity-analysis insight report.

Return only Markdown with exactly these top-level sections: Executive Summary, Figure Walk-through, Inferences Based on the Data. In Executive Summary, summarize the study setup and key sensitivity findings. In Figure Walk-through, discuss each requested/generated figure from figure_manifest. In Inferences Based on the Data, rank the strongest parameter effects, explain response curves or correlations, identify fragile regions, and recommend next experiments. Follow report_rules exactly.""",
    "sensitivity_analysis_report": """Write a sensitivity-analysis engineering report.

Return only Markdown with exactly these top-level sections: Executive Summary, Figure Walk-through, Inferences Based on the Data. In Executive Summary, identify the sensitivity method, varied parameters, metrics, baseline mode, successful/failed run counts, and the strongest observed drivers. In Figure Walk-through, discuss response-curve, LHS scatter, two-parameter grid heatmap, and ranking figures that appear in figure_manifest. In Inferences Based on the Data, compare OAAT deltas, LHS correlations, or two-parameter grid spans as appropriate; call out failed runs or missing metrics; identify fragile regions; and recommend the next sensitivity experiment. Follow report_rules exactly.""",
    "controller_bench_report": """Write a controller-benchmark engineering report.

Return only Markdown with exactly these top-level sections: Executive Summary, Figure Walk-through, Inferences Based on the Data. In Executive Summary, identify the controller target, variants, cases, pass rates, objective results, and strongest leaderboard findings. In Figure Walk-through, discuss generated benchmark figures that appear in figure_manifest. In Inferences Based on the Data, compare controller variants, explain which cases or objectives drove the ranking, call out weak evidence or missing metrics, and recommend follow-up benchmark runs. Follow report_rules exactly.""",
    "controller_selection_memo": """Write a controller-selection decision memo.

Return only Markdown with exactly these top-level sections: Executive Summary, Figure Walk-through, Inferences Based on the Data. In Executive Summary, recommend the best-supported controller choice from the supplied benchmark data and state the caveats. In Figure Walk-through, discuss generated benchmark figures that appear in figure_manifest. In Inferences Based on the Data, compare variants using pass rates, objective pass rates, metric means, leaderboard rows, failed cases, and optimization results where available. Follow report_rules exactly.""",
    "validation_evidence_summary": """Write a validation evidence summary for an engineering review.

Return only Markdown with exactly these top-level sections: Executive Summary, Figure Walk-through, Inferences Based on the Data. In Executive Summary, identify the validation suite, benchmark kinds, pass/fail counts, failed checks, tolerance or baseline comparisons, and decision-readiness caveats. In Figure Walk-through, discuss generated validation figures that appear in figure_manifest. In Inferences Based on the Data, explain what evidence passed, what failed, which tolerances or baselines drove the result, and what validation work remains. Follow report_rules exactly.""",
}


REPORT_MODE_PROMPT_PROFILES: dict[str, dict[str, str]] = {
    "engineering_report": {
        "default": "analyst_report",
        "monte_carlo": "analyst_report",
        "sensitivity": "sensitivity_analysis_report",
        "controller_bench": "controller_bench_report",
        "validation_harness": "validation_evidence_summary",
    },
    "decision_memo": {
        "default": "commander_summary",
        "monte_carlo": "commander_summary",
        "sensitivity": "sensitivity_insights",
        "controller_bench": "controller_selection_memo",
        "validation_harness": "validation_evidence_summary",
    },
    "adversarial_advantage": {
        "default": "adversarial_advantage_assessment",
        "monte_carlo": "adversarial_advantage_assessment",
        "sensitivity": "adversarial_advantage_assessment",
        "controller_bench": "adversarial_advantage_assessment",
        "validation_harness": "adversarial_advantage_assessment",
    },
    "executive_summary": {
        "default": "commander_summary",
        "monte_carlo": "commander_summary",
        "sensitivity": "sensitivity_insights",
        "controller_bench": "controller_selection_memo",
        "validation_harness": "validation_evidence_summary",
    },
    "technical_analysis": {
        "default": "analyst_report",
        "monte_carlo": "analyst_report",
        "sensitivity": "sensitivity_analysis_report",
        "controller_bench": "controller_bench_report",
        "validation_harness": "validation_evidence_summary",
    },
    "controller_selection_memo": {
        "default": "controller_selection_memo",
        "controller_bench": "controller_selection_memo",
    },
    "validation_evidence_summary": {
        "default": "validation_evidence_summary",
        "validation_harness": "validation_evidence_summary",
    },
}


SIMULATOR_CONTEXT = {
    "system_purpose": (
        "Orbital Engagement Lab simulates orbital, attitude, sensing, estimation, control, mission, rocket, "
        "and campaign-analysis workflows for spacecraft engagement and rendezvous scenarios."
    ),
    "campaign_flow": (
        "For a Monte Carlo campaign, the runner samples configured parameter variations, creates one scenario "
        "configuration per iteration, runs each scenario, then aggregates deterministic outputs into summaries, "
        "commander briefs, analyst packs, plots, and optional AI-written reports. For sensitivity analysis, it "
        "generates one-at-a-time, Latin hypercube, or two-parameter grid scenario variants and compares tracked metrics."
    ),
    "important_units": {
        "distance": "kilometers unless a field name says meters",
        "velocity": "kilometers per second unless a field name says meters per second",
        "delta_v": "meters per second in total_dv_m_s fields",
        "time": "seconds",
        "probabilities": "fractions from 0.0 to 1.0",
    },
    "relative_state_convention": (
        "relative_to_target_ric.state is [radial separation, in-track separation, cross-track separation, "
        "radial velocity, in-track velocity, cross-track velocity]. The frame may be rectangular RIC or "
        "curvilinear RIC depending on the config."
    ),
    "interpretation_rules": [
        "A pass/fail result comes from the configured gates and run assessment, not from independent judgment.",
        "closest_approach_km is the minimum chaser-target range found in a run when available.",
        "ZeroController means the object is not applying active orbit-control commands.",
        "Do not treat missing plots, disabled attitude, or disabled animations as failures unless a configured gate says so.",
        "Use the scenario description and config summary as the plain-English description of what was simulated.",
    ],
}


REPORT_RULES = [
    "Use only data present in the supplied packet. Do not invent values, thresholds, failures, plots, or causal explanations.",
    "Do not describe the JSON schema or say that you are summarizing a JSON payload.",
    "Clearly separate observed results from inferences or recommendations.",
    "If a requested metric, figure, or artifact is missing, say it was not available.",
    "A reported value of 0.0 is known data, not missing data. Do not call a risk or resource metric unknown when the source brief gives a numeric value.",
    "If no failure modes were reported, say no failure modes were observed in this campaign; do not recommend investigating failure modes unless recommending a broader or more stressful follow-up envelope.",
    "Do not claim to visually inspect a figure unless figure_manifest.image_pixels_available is true for that figure.",
    "For figure discussion, use figure_manifest.data_sources for numeric claims. Use image paths only as artifact references unless image pixels are available.",
    "When image pixels are not available, use figure_manifest.description and related numeric payload data to explain what the figure is intended to show.",
    "In the Figure Walk-through section, place each generated figure with a placeholder line exactly like [[FIGURE:figure_id]].",
    "Put each figure placeholder immediately before the paragraph that discusses that figure.",
    "Keep units attached to numeric claims.",
    "Treat pass/fail and risk probabilities as deterministic campaign-report outputs, not as independent judgments.",
    "Return Markdown only.",
]

DIRECT_AI_REPORT_POSTURE = {
    "product_posture": "optional_headless_provider_adapter",
    "preferred_workflow": "oel_packet_external_agent_authorship_then_oel_audit",
}


DEFAULT_AI_PRICE_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "google/gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini/gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "openai/gpt-5-mini": {"input": 0.25, "output": 2.00},
    "openai/gpt-5-nano": {"input": 0.05, "output": 0.40},
    "openai/gpt-5": {"input": 1.25, "output": 10.00},
    "openai/gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "openai/gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "openai/gpt-5.4": {"input": 2.50, "output": 15.00},
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-7-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "ollama/": {"input": 0.0, "output": 0.0},
}


FIGURE_DESCRIPTIONS = {
    "run_dashboard": "Single-run summary dashboard with core trajectory, range, and control/mission telemetry.",
    "rendezvous_summary": "Rendezvous-focused summary of relative motion and engagement geometry.",
    "rendezvous_summary_curvilinear": "Rendezvous-focused dashboard using curvilinear RIC projections, combined range/speed, and cumulative delta-v.",
    "orbit_eci": "Object orbit trajectory in the Earth-centered inertial frame.",
    "orbital_element_a": "Semi-major axis over time.",
    "orbital_element_ecc": "Eccentricity over time.",
    "orbital_element_inc": "Inclination over time.",
    "orbital_element_raan": "Right ascension of ascending node over time.",
    "orbital_element_argp": "Argument of perigee over time.",
    "orbital_element_true_anomaly": "True anomaly over time.",
    "orbital_elements_summary": "Six-panel classical orbital element history.",
    "orbital_elements_angles": "Inclination, RAAN, argument of perigee, and true anomaly over time.",
    "ground_track": "Object ground track over Earth latitude/longitude.",
    "ground_track_multi": "Multi-object ground tracks over Earth latitude/longitude.",
    "trajectory_ecef": "Object trajectory in the Earth-centered Earth-fixed frame.",
    "trajectory_ric_rect": "Object trajectory relative to a reference object in rectangular RIC coordinates.",
    "trajectory_ric_curv": "Object trajectory relative to a reference object in curvilinear RIC coordinates.",
    "trajectory_ric_rect_2d": "Two-dimensional rectangular-RIC relative trajectory projection.",
    "trajectory_ric_curv_2d": "Two-dimensional curvilinear-RIC relative trajectory projection.",
    "trajectory_eci_multi": "Multi-object trajectory in the Earth-centered inertial frame.",
    "trajectory_ecef_multi": "Multi-object trajectory in the Earth-centered Earth-fixed frame.",
    "trajectory_ric_rect_multi": "Multi-object rectangular-RIC trajectories relative to a reference object.",
    "trajectory_ric_curv_multi": "Multi-object curvilinear-RIC trajectories relative to a reference object.",
    "trajectory_ric_rect_2d_multi": "Multi-object two-dimensional rectangular-RIC trajectory projection.",
    "trajectory_ric_rect_2d_multi_target_burns": "Multi-object rectangular-RIC 2D projection with orange target-burn markers on the plotted trajectory.",
    "trajectory_ric_curv_2d_multi": "Multi-object two-dimensional curvilinear-RIC trajectory projection.",
    "trajectory_ric_curv_2d_multi_target_burns": "Multi-object curvilinear-RIC 2D projection with orange target-burn markers on the plotted trajectory.",
    "relative_range": "Range between relevant objects over time.",
    "control_effort": "Control effort or delta-v usage over time.",
    "control_thrust": "Per-object thrust commands over time.",
    "control_thrust_multi": "Multi-object thrust commands over time.",
    "control_thrust_ric": "Per-object thrust commands expressed in RIC coordinates.",
    "control_thrust_ric_multi": "Multi-object thrust commands expressed in RIC coordinates.",
    "satellite_delta_v_remaining": "Remaining satellite delta-v budget over time.",
    "estimation_error": "Estimator error summary over time.",
    "estimation_error_components": "Estimator error components over time.",
    "knowledge_filtering": "Truth, raw measurement, filtered estimate, and sensor residual distribution evidence for knowledge tracks.",
    "knowledge_timeline": "Knowledge and tracking timeline for configured observers/targets.",
    "sensor_access": "Sensor line-of-sight or access intervals.",
    "ground_station_access": "Ground-station access, elevation, and slant range over time.",
    "quaternion_error": "Attitude quaternion error over time.",
    "attitude_control_summary": "Combined attitude tracking, body-rate, thrust, and alignment diagnostics.",
    "attitude": "Attitude state/tumble visualization.",
    "quaternion_eci": "Quaternion components relative to ECI frame.",
    "quaternion_ric": "Quaternion components relative to RIC frame.",
    "rates_eci": "Body rates relative to ECI frame.",
    "rates_ric": "Body rates relative to RIC frame.",
    "rocket_ascent_diagnostics": "Rocket ascent performance and guidance diagnostics.",
    "rocket_gnc_diagnostics": "Rocket guidance, navigation, control, and targeting diagnostics.",
    "rocket_orbital_elements": "Rocket orbital element history.",
    "rocket_fuel_remaining": "Rocket fuel remaining over time.",
    "rocket_mission_timeline": "Rocket mission event timeline.",
    "rocket_downrange_altitude": "Rocket altitude versus downrange distance.",
    "rocket_maxq_throttle": "Rocket max-Q, throttle, Mach, and altitude behavior.",
    "rocket_tvc_aero_authority": "Rocket TVC, aero angle, aero load, dynamic pressure, and T/W authority.",
    "rocket_insertion_scorecard": "Rocket insertion target, final orbit, resource, and load scorecard.",
    "reentry_summary": "Atmospheric pass/re-entry altitude, dynamic pressure, g-load, and heat-rate overview.",
    "reentry_aero": "Atmospheric density, relative speed, dynamic pressure, and aero acceleration histories.",
    "reentry_thermal": "Atmospheric pass/re-entry heat-rate and integrated heat-load histories.",
    "atmospheric_pass": "Aero-assisted pass summary with altitude, lift/drag acceleration, cross-track response, heat load, and lift-axis alignment.",
    "thrust_alignment_error": "Thrust vector alignment error over time.",
    "master_monte_carlo_histograms": "Campaign-level histograms for duration, closest approach, delta-v, and related aggregate quantities.",
    "master_monte_carlo_relative_range_timeseries": "Campaign-level relative range time histories by Monte Carlo iteration.",
    "master_monte_carlo_ops_dashboard": "Campaign-level dashboard of closest approach, duration, delta-v, and failure mode counts.",
    "master_monte_carlo_initial_relative_state_vs_closest_approach": "Campaign-level relationship between initial relative RIC state samples and closest approach.",
    "master_monte_carlo_delta_v_remaining": "Campaign-level delta-v remaining distributions and run-by-run scatter plots.",
}

__all__ = [name for name in globals() if not name.startswith("__")]

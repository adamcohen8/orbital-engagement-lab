from __future__ import annotations

from sim.agent_task.models import AgentTaskRecipe

RECIPES: dict[str, AgentTaskRecipe] = {
    "quickstart_review": AgentTaskRecipe(
        recipe_id="quickstart_review",
        title="Quickstart Review Evidence",
        description="Validate and run the five-minute public quickstart, then package review-store evidence.",
        config_path="configs/quickstart_5min.yaml",
        maturity="supported",
        query_names=(
            "run_metadata",
            "objects",
            "rendezvous_metrics",
            "rendezvous_closest_approach",
            "burn_activity",
            "artifacts",
        ),
        plot_recipe_ids=("relative_range", "burn_activity"),
        semantic_metric_names=(
            "initial_range_km",
            "final_range_km",
            "closest_approach_km",
            "closest_approach_time_s",
            "burn_activity",
        ),
        tags=("public", "quickstart", "single_run", "rendezvous"),
        notes=("Uses the same deterministic scenario YAML as the public quickstart path.",),
    ),
    "flagship_ric_pd_review": AgentTaskRecipe(
        recipe_id="flagship_ric_pd_review",
        title="Flagship RIC PD Review Evidence",
        description="Run the flagship 10 km RIC PD rendezvous scenario and inspect review evidence.",
        config_path="configs/ric_pd_10km_experiment.yaml",
        maturity="supported",
        query_names=(
            "run_metadata",
            "rendezvous_metrics",
            "rendezvous_closest_approach",
            "relative_final_state",
            "burn_activity",
            "artifacts",
        ),
        plot_recipe_ids=("relative_range", "relative_range_rate", "burn_activity"),
        semantic_metric_names=(
            "final_range_km",
            "closest_approach_km",
            "closest_approach_time_s",
            "range_rate_km_s",
            "burn_activity",
        ),
        tags=("public", "flagship", "single_run", "rendezvous", "controller"),
    ),
    "mission_reconstitution_review": AgentTaskRecipe(
        recipe_id="mission_reconstitution_review",
        title="Mission Reconstitution Review Evidence",
        description="Run the public mission-reconstitution trade-space example and inspect planner evidence.",
        config_path="agents/examples/public_agent_mission_reconstitution_trade_space.yaml",
        maturity="supported",
        query_names=(
            "run_metadata",
            "mission_recovery_summary",
            "mission_recovery_candidates",
            "mission_recovery_burns",
            "artifacts",
        ),
        semantic_metric_names=("mission_recovery_delta_v_m_s",),
        tags=("public", "agent_example", "mission_recovery", "trade_space"),
    ),
    "ground_access_review": AgentTaskRecipe(
        recipe_id="ground_access_review",
        title="Ground Access Review Evidence",
        description="Run the public ground-access example and package station/object access evidence.",
        config_path="agents/examples/public_agent_ground_access.yaml",
        maturity="supported",
        query_names=(
            "run_metadata",
            "objects",
            "ground_access_summary",
            "ground_access_windows",
            "ground_access_no_access_reasons",
            "artifacts",
        ),
        semantic_metric_names=("ground_access",),
        tags=("public", "agent_example", "ground_access"),
    ),
    "attitude_hold_review": AgentTaskRecipe(
        recipe_id="attitude_hold_review",
        title="Public Attitude-Hold Review Evidence",
        description=(
            "Run the public disturbed attitude-hold example and package desired-versus-actual "
            "quaternion error, body-rate, controller, and plot evidence."
        ),
        config_path="examples/configs/public_attitude_hold_disturbance.yaml",
        maturity="supported",
        query_names=(
            "run_metadata",
            "objects",
            "attitude_error_first_last",
            "attitude_state_first_last",
            "artifacts",
        ),
        plot_recipe_ids=("attitude_error", "attitude_body_rates"),
        tags=("public", "agent_example", "attitude", "ideal_wrench", "control"),
        notes=(
            "The configured hardware profile is hardware.ideal_wrench.v1, not a physical reaction-wheel model.",
            "Results are deterministic simulation evidence, not flight-qualified ADCS performance.",
        ),
    ),
    "coverage_link_review": AgentTaskRecipe(
        recipe_id="coverage_link_review",
        title="Coverage and Directed-Link Review Evidence",
        description=(
            "Run the public whole-Earth coverage and directed-link example, then package "
            "coverage, availability-window, margin, and artifact evidence."
        ),
        config_path="examples/configs/public_coverage_and_link_analysis.yaml",
        maturity="supported",
        query_names=(
            "run_metadata",
            "objects",
            "coverage_summary",
            "coverage_transition_summary",
            "directed_link_summary",
            "directed_link_windows",
            "artifacts",
        ),
        plot_recipe_ids=("coverage_fraction", "directed_link_margin"),
        semantic_metric_names=("coverage_fraction", "directed_link_margin_db"),
        tags=("public", "agent_example", "coverage", "link_budget", "communications"),
        notes=(
            "Coverage and link results are evidence-only post-processing over one deterministic run.",
            "This workflow does not establish calibrated payload performance or operational RF assurance.",
        ),
    ),
    "ogp_sgp4_review": AgentTaskRecipe(
        recipe_id="ogp_sgp4_review",
        title="Public OGP-SGP4 Propagation Evidence",
        description=(
            "Validate and run a fixed public TLE through continuous passive OGP-SGP4, then package "
            "propagation provenance, canonical ECI state evidence, and a review-store plot."
        ),
        config_path="agents/examples/public_agent_ogp_sgp4_propagation.yaml",
        maturity="supported",
        query_names=(
            "run_metadata",
            "objects",
            "ogp_propagation_contract",
            "passive_final_state",
            "artifacts",
        ),
        plot_recipe_ids=("object_eci_radius",),
        tags=("public", "agent_example", "ogp", "sgp4", "propagation", "tle"),
        notes=(
            "Uses a fixed historical public TLE so the workflow is deterministic and offline.",
            "The OGP product is native TEME while review object-state histories remain canonical ECI evidence.",
            "This workflow does not establish current or operational ephemeris accuracy.",
        ),
    ),
    "ogp_sdp4_review": AgentTaskRecipe(
        recipe_id="ogp_sdp4_review",
        title="Public OGP-SDP4 Deep-Space Propagation Evidence",
        description=(
            "Validate and run a synthetic GEO-like TLE through continuous passive OGP-SDP4, then package "
            "resolved regime, period, frame, canonical ECI state, and radius evidence."
        ),
        config_path="agents/examples/public_agent_ogp_sdp4_propagation.yaml",
        maturity="supported",
        query_names=(
            "run_metadata",
            "objects",
            "ogp_propagation_contract",
            "object_final_state",
            "object_eci_radius_extrema",
            "artifacts",
        ),
        plot_recipe_ids=("object_eci_radius",),
        tags=("public", "agent_example", "ogp", "sdp4", "deep_space", "propagation", "tle"),
        notes=(
            "The fixed synthetic TLE has valid checksums and is not a real catalog object.",
            "The review contract must resolve OGP-SDP4, deep_space, and an orbital period above 225 minutes.",
            "This workflow does not establish current or operational ephemeris accuracy.",
        ),
    ),
}


def get_recipe(recipe_id: str) -> AgentTaskRecipe | None:
    return RECIPES.get(str(recipe_id or "").strip())


def list_recipes() -> list[AgentTaskRecipe]:
    return [RECIPES[key] for key in sorted(RECIPES)]


def is_public_mcp_recipe(recipe: AgentTaskRecipe | None) -> bool:
    """Return whether a recipe is part of the supported public MCP surface."""

    return bool(
        recipe is not None
        and recipe.maturity == "supported"
        and recipe.workflow == "scenario_run"
        and "public" in recipe.tags
    )

from __future__ import annotations

from sim.agent_task.models import AgentTaskRecipe

RECIPES: dict[str, AgentTaskRecipe] = {
    "quickstart_review": AgentTaskRecipe(
        recipe_id="quickstart_review",
        title="Quickstart Review Evidence",
        description="Validate and run the five-minute public quickstart, then package review-store evidence.",
        config_path="configs/quickstart_5min.yaml",
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
        query_names=("run_metadata", "objects", "ground_access_summary", "ground_access_no_access_reasons", "artifacts"),
        semantic_metric_names=("ground_access",),
        tags=("public", "agent_example", "ground_access"),
    ),
}


def get_recipe(recipe_id: str) -> AgentTaskRecipe | None:
    return RECIPES.get(str(recipe_id or "").strip())


def list_recipes() -> list[AgentTaskRecipe]:
    return [RECIPES[key] for key in sorted(RECIPES)]

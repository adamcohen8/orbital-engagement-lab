from __future__ import annotations

from sim.agent_task.models import AgentPlotRecipe
from sim.review.plotting import ReviewPlotSpec

PLOT_RECIPES: dict[str, AgentPlotRecipe] = {
    "object_eci_radius": AgentPlotRecipe(
        recipe_id="object_eci_radius",
        title="Canonical ECI radius",
        description="Plots canonical ECI radius from the object_state review table.",
        sql=(
            "SELECT time_s, object_id, "
            "sqrt(pos_x_eci_km * pos_x_eci_km + pos_y_eci_km * pos_y_eci_km + "
            "pos_z_eci_km * pos_z_eci_km) AS radius_km "
            "FROM object_state ORDER BY object_id, time_s"
        ),
        x_column="time_s",
        y_columns=("radius_km",),
        group_column="object_id",
        plot_type="line",
        x_label="Time (s)",
        y_label="ECI radius (km)",
        artifact_id="agent_object_eci_radius",
        supported_tables=("object_state",),
    ),
    "relative_range": AgentPlotRecipe(
        recipe_id="relative_range",
        title="Relative range over time",
        description="Plots deputy-chief range from the relative_state review table.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, deputy_id || ':' || chief_id AS pair_id, range_km "
            "FROM relative_state ORDER BY pair_id, time_s"
        ),
        x_column="time_s",
        y_columns=("range_km",),
        group_column="pair_id",
        plot_type="line",
        x_label="Time (s)",
        y_label="Range (km)",
        artifact_id="agent_relative_range",
        semantic_metric_names=("closest_approach_km", "final_range_km"),
        supported_tables=("relative_state",),
    ),
    "relative_range_rate": AgentPlotRecipe(
        recipe_id="relative_range_rate",
        title="Relative range rate over time",
        description="Plots relative range rate from the relative_state review table.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, deputy_id || ':' || chief_id AS pair_id, range_rate_km_s "
            "FROM relative_state ORDER BY pair_id, time_s"
        ),
        x_column="time_s",
        y_columns=("range_rate_km_s",),
        group_column="pair_id",
        plot_type="line",
        x_label="Time (s)",
        y_label="Range rate (km/s)",
        artifact_id="agent_relative_range_rate",
        semantic_metric_names=("range_rate_km_s",),
        supported_tables=("relative_state",),
    ),
    "burn_activity": AgentPlotRecipe(
        recipe_id="burn_activity",
        title="Burn activity by object",
        description="Plots active thrust samples by object.",
        sql=(
            "SELECT object_id, SUM(burn_active) AS active_samples "
            "FROM thrust GROUP BY object_id ORDER BY object_id"
        ),
        x_column="object_id",
        y_columns=("active_samples",),
        plot_type="bar",
        x_label="Object",
        y_label="Active thrust samples",
        artifact_id="agent_burn_activity",
        semantic_metric_names=("burn_activity",),
        supported_tables=("thrust",),
    ),
    "campaign_closest_approach": AgentPlotRecipe(
        recipe_id="campaign_closest_approach",
        title="Campaign closest approach by iteration",
        description="Plots Monte Carlo closest-approach results by iteration.",
        sql="SELECT iteration, closest_approach_km FROM campaign_runs ORDER BY iteration",
        x_column="iteration",
        y_columns=("closest_approach_km",),
        plot_type="scatter",
        x_label="Iteration",
        y_label="Closest approach (km)",
        artifact_id="agent_campaign_closest_approach",
        semantic_metric_names=("campaign_closest_approach_km",),
        supported_tables=("campaign_runs",),
    ),
    "sensitivity_effects": AgentPlotRecipe(
        recipe_id="sensitivity_effects",
        title="Sensitivity effect sizes",
        description="Plots ranked sensitivity effect sizes by parameter.",
        sql=(
            "SELECT parameter_path, effect_size FROM sensitivity_rankings "
            "ORDER BY rank, parameter_path, metric_path"
        ),
        x_column="parameter_path",
        y_columns=("effect_size",),
        plot_type="bar",
        x_label="Parameter",
        y_label="Effect size",
        artifact_id="agent_sensitivity_effects",
        semantic_metric_names=("sensitivity_effect_size",),
        supported_tables=("sensitivity_rankings",),
    ),
}


def get_plot_recipe(recipe_id: str) -> AgentPlotRecipe | None:
    return PLOT_RECIPES.get(str(recipe_id or "").strip())


def list_plot_recipes() -> list[AgentPlotRecipe]:
    return [PLOT_RECIPES[key] for key in sorted(PLOT_RECIPES)]


def review_plot_spec(
    recipe: AgentPlotRecipe,
    *,
    style_name: str = "oel_dark",
    file_format: str = "png",
    artifact_id: str = "",
) -> ReviewPlotSpec:
    return ReviewPlotSpec(
        sql=recipe.sql,
        x_column=recipe.x_column,
        y_columns=list(recipe.y_columns),
        plot_type=recipe.plot_type,
        group_column=recipe.group_column,
        style_name=style_name,
        title=recipe.title,
        x_label=recipe.x_label,
        y_label=recipe.y_label,
        artifact_id=artifact_id or recipe.artifact_id,
        file_format=file_format,
    )

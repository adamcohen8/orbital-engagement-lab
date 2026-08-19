from __future__ import annotations

from sim.review.plot_recipes import (
    REVIEW_PLOT_RECIPES,
    ReviewPlotRecipe,
    get_review_plot_recipe,
    list_review_plot_recipes,
)
from sim.review.plotting import ReviewPlotSpec

# Compatibility façade. Review plotting owns the authoritative recipe registry;
# agent-task and MCP callers consume the same definitions.
PLOT_RECIPES = REVIEW_PLOT_RECIPES


def get_plot_recipe(recipe_id: str) -> ReviewPlotRecipe | None:
    return get_review_plot_recipe(recipe_id)


def list_plot_recipes() -> list[ReviewPlotRecipe]:
    return list_review_plot_recipes()


def review_plot_spec(
    recipe: ReviewPlotRecipe,
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
        renderer_id=recipe.renderer_id,
        extra={
            "source": "oel_review_plot_api",
            "caller": "sim.agent_task",
            "recipe_id": recipe.recipe_id,
            "recipe_version": recipe.recipe_version,
        },
    )


__all__ = ["PLOT_RECIPES", "get_plot_recipe", "list_plot_recipes", "review_plot_spec"]

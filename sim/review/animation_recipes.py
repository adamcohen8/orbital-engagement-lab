"""Discoverable, allowlisted animation recipes over OEL review evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ANIMATION_RECIPE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReviewAnimationRecipe:
    recipe_id: str
    title: str
    description: str
    sql: str
    renderer_id: str
    supported_tables: tuple[str, ...]
    required_columns: tuple[str, ...]
    artifact_id: str
    maturity: str = "supported"
    recipe_version: int = 1
    natural_language_triggers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.maturity not in {"supported", "prototype", "experimental"}:
            raise ValueError(f"Unknown review animation recipe maturity {self.maturity!r}.")
        if not self.sql.lstrip().upper().startswith(("SELECT", "WITH")):
            raise ValueError(f"Review animation recipe {self.recipe_id!r} must use read-only SELECT/WITH SQL.")
        if self.recipe_version < 1:
            raise ValueError(f"Review animation recipe {self.recipe_id!r} must have a positive recipe_version.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REVIEW_ANIMATION_RECIPES: dict[str, ReviewAnimationRecipe] = {
    "relative_position_ric_2d": ReviewAnimationRecipe(
        recipe_id="relative_position_ric_2d",
        title="Relative trajectory in rectangular RIC",
        description=(
            "Animate professional I-R, I-C, and C-R projections from recorded rectangular-RIC review evidence."
        ),
        sql=(
            "SELECT time_s, deputy_id, chief_id, deputy_id || ':' || chief_id AS pair_id, "
            "r_radial_km, i_intrack_km, c_crosstrack_km "
            "FROM relative_state ORDER BY pair_id, time_s"
        ),
        renderer_id="ric_rectangular_2d",
        supported_tables=("relative_state",),
        required_columns=(
            "time_s",
            "pair_id",
            "r_radial_km",
            "i_intrack_km",
            "c_crosstrack_km",
        ),
        artifact_id="animation_relative_position_ric_2d",
        natural_language_triggers=(
            "animate the 2D RIC trajectory",
            "make a RIC trajectory movie",
            "animate radial in-track cross-track motion",
        ),
    ),
}


def get_review_animation_recipe(recipe_id: str) -> ReviewAnimationRecipe | None:
    return REVIEW_ANIMATION_RECIPES.get(str(recipe_id or "").strip())


def list_review_animation_recipes() -> list[ReviewAnimationRecipe]:
    return [REVIEW_ANIMATION_RECIPES[key] for key in sorted(REVIEW_ANIMATION_RECIPES)]


__all__ = [
    "ANIMATION_RECIPE_SCHEMA_VERSION",
    "REVIEW_ANIMATION_RECIPES",
    "ReviewAnimationRecipe",
    "get_review_animation_recipe",
    "list_review_animation_recipes",
]

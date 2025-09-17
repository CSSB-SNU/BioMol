from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from biomol.core.feature import Feature


Target = Mapping[str, type[Any]]  # (name, type)


@dataclass(frozen=True)
class Recipe:
    """A single step in a data processing recipe."""

    target: Target
    instruction: Callable  # old name: insturction or mapper
    inputs: dict[str, Any]


class RecipeBook:
    """Recipe Builder for defining data processing workflows."""

    def __init__(self):
        self.steps: list[Recipe] = []

    def add(self, target: Target, instruction: Callable, **inputs) -> "RecipeBook":
        """Add a new step to the recipe."""
        step = Recipe(target=target, instruction=instruction, inputs=inputs)
        self.steps.append(step)
        return self

    def targets(self) -> list[str]:
        """List all target names defined in the recipe."""
        return [name for step in self.steps for name in step.target]

    def __contains__(self, target_name: str) -> bool:
        """Check if a target is already defined in the recipe."""
        return any(target_name in step.target for step in self.steps)

    def __getitem__(self, target_name: str) -> Recipe:
        """Retrieve a recipe step by target name."""
        for step in self.steps:
            if target_name in step.target:
                return step
        msg = f"Recipe for target '{target_name}' not found."
        raise KeyError(msg)

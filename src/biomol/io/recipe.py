from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

Target = Mapping[str, type[Any]]  # (name, type)


@dataclass(frozen=True)
class Constant:
    """A node in the recipe input/output graph."""

    value: object

@dataclass(frozen=True)
class Recipe:
    """A single step in a data processing recipe."""

    target: Target
    instruction: Callable  # old name: insturction or mapper
    inputs: dict[str, Any]


class RecipeBook:
    """Recipe Builder for defining data processing workflows."""

    def __init__(self) -> None:
        self.steps: list[Recipe] = []

    def _check_duplicate_targets(self, target: Target) -> None:
        for name in target:
            if name in self:
                msg = f"Target '{name}' is already defined in the recipe."
                raise ValueError(msg)

    def add(
        self,
        target: Target | list[Target],
        instruction: Callable,
        *,
        group: bool = False,
        **inputs: object,
    ) -> "RecipeBook":
        """Add a new step to the recipe."""
        if group and isinstance(target, list):
            for _input in inputs.values():
                if not isinstance(_input, list) or len(_input) != len(target):
                    msg = (
                        "When 'group' is True, all inputs must be lists of the same "
                        "length as target."
                    )
                    raise ValueError(msg)
            for i, single_target in enumerate(target):
                single_inputs = {
                    k: v[i] if isinstance(v, list) else v for k, v in inputs.items()
                }
                self._check_duplicate_targets(single_target)
                step = Recipe(
                    target=single_target,
                    instruction=instruction,
                    inputs=single_inputs,
                )
                self.steps.append(step)
            return self
        self._check_duplicate_targets(target)
        step = Recipe(target=target, instruction=instruction, inputs=inputs)
        self.steps.append(step)
        return self

    def targets(self) -> list[str]:
        """List all target names defined in the recipe."""
        return [target_name for step in self.steps for target_name in step.target]

    def __contains__(self, target_name: str) -> bool:
        """Check if a target is already defined in the recipe."""
        return any(target_name in step.target for step in self.steps)

    def __getitem__(self, target_name: str) -> Recipe:
        """Retrieve a recipe step by target name."""
        for step in self.steps:
            if target_name in step.target:
                return step, step.target.keys()
        msg = f"Recipe for target '{target_name}' not found."
        raise KeyError(msg)

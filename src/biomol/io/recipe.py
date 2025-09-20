from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias, overload


@dataclass(frozen=True)
class Variable:
    """Represents a target variable in a recipe step."""

    name: str
    type: type[Any]


VariableSet: TypeAlias = tuple[Variable, ...]
VariableMap: TypeAlias = Mapping[str, Variable]

RawVariableSet: TypeAlias = tuple[tuple[str, type[Any]], ...]
RawVariableMap: TypeAlias = Mapping[str, tuple[str, type]]


@dataclass(frozen=True, slots=True)
class Inputs:
    """Represents input variables for a recipe step."""

    args: VariableSet = field(default_factory=tuple)
    kwargs: VariableMap = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Recipe:
    """A single step in a data processing recipe."""

    targets: VariableSet
    instruction: Callable
    inputs: Inputs


class RecipeError(KeyError):
    """Custom error for missing recipe targets."""


class RecipeBook:
    """Recipe Builder for defining data processing workflows."""

    def __init__(self) -> None:
        self.steps: list[Recipe] = []

    def _check_duplicate_targets(self, targets: VariableSet) -> None:
        for target in targets:
            if target.name in self:
                msg = f"Target '{target.name}' is already defined in the recipe."
                raise ValueError(msg)

    def _coerce_to_variable_set(self, varset: RawVariableSet | None) -> VariableSet:
        if not varset:
            return ()
        tuple(Variable(*t) for t in varset)
        return tuple(Variable(*t) for t in varset)

    def _coerce_to_variable_map(self, varset: RawVariableMap | None) -> VariableMap:
        if not varset:
            return {}
        return {key: Variable(*t) for key, t in varset.items()}

    def _single_add(
        self,
        targetset: RawVariableSet,
        instruction: Callable,
        inputs: dict[str, Any],
    ) -> None:
        arg_vars = self._coerce_to_variable_set(inputs.get("args"))
        kwarg_vars = self._coerce_to_variable_map(inputs.get("kwargs"))
        params_dict = inputs.get("params", {})
        final_inputs = Inputs(args=arg_vars, kwargs=kwarg_vars, params=params_dict)
        final_targetset = self._coerce_to_variable_set(targetset)

        self._check_duplicate_targets(final_targetset)
        step = Recipe(
            targets=final_targetset,
            instruction=instruction,
            inputs=final_inputs,
        )
        self.steps.append(step)

    @overload
    def add(
        self,
        targets: RawVariableSet,
        instruction: Callable,
        inputs: dict[str, Any],
    ) -> "RecipeBook": ...
    @overload
    def add(
        self,
        targets: list[RawVariableSet],
        instruction: Callable,
        inputs: list[dict[str, Any]],
    ) -> "RecipeBook": ...

    def add(
        self,
        targets: Any,
        instruction: Callable,
        inputs: Any,
    ) -> "RecipeBook":
        """Add a new step to the recipe."""
        if isinstance(targets, list) and isinstance(inputs, list):
            if len(targets) != len(inputs):
                msg = (
                    "When providing lists of targets and inputs, "
                    "they must be of equal length."
                )
                raise ValueError(msg)
            for targetset, input_bundle in zip(targets, inputs, strict=True):
                self._single_add(targetset, instruction, input_bundle)

        else:
            self._single_add(targets, instruction, inputs)

        return self

    def __contains__(self, target_name: str) -> bool:
        """Check if any step contains a target with this name."""
        return any(t.name == target_name for step in self.steps for t in step.targets)

    def __getitem__(self, target_name: str) -> Recipe:
        """Retrieve a recipe step by target name."""
        for step in self.steps:
            for t in step.targets:
                if t.name == target_name:
                    return step
        msg = f"Recipe for target '{target_name}' not found."
        raise RecipeError(msg)

    def targets(self) -> list[Variable]:
        """Return a list of all target names defined in the recipe book."""
        all_targets = []
        for step in self.steps:
            all_targets.extend(list(step.targets))
        return all_targets

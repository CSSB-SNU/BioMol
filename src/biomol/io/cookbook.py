from typing import Any

from biomol.io.context import ParsingCache
from biomol.io.recipe import RecipeBook


class Cooker:
    """
    Process input datas through a series of defined recipes.

    Args:
    parse_cache: An instance of ParsingCache to store intermediate and final results.
    recipebook: An instance of RecipeBook containing the recipes to execute.
    """

    def __init__(self, parse_cache: ParsingCache, recipebook: RecipeBook):
        self.parse_cache = parse_cache
        self.recipebook = recipebook

    def prep(self, data_dict: dict, fields: list[str]) -> None:
        """Prepare the context with initial data."""
        for field in fields:
            if field in data_dict:
                self.parse_cache.add_data(field, data_dict[field])
            else:
                msg = f"Field {field} not found in data_dict."
                raise ValueError(msg)

    def cook(self) -> None:
        """Execute all recipes in dependency order."""
        visited = set()

        def resolve(output: str):
            # Already computed
            if output in self.parse_cache:
                return self.parse_cache[output]
            # Prevent infinite recursion
            if output in visited:
                msg = f"Cyclic dependency detected at '{output}'"
                raise RuntimeError(msg)
            visited.add(output)

            recipe = self.recipebook[output]
            inputs = {}
            for arg_name, input in recipe.inputs.items():
                inputs[arg_name] = resolve(input)  # recursively resolve

            result = recipe.instruction(**inputs)
            self.parse_cache.add_data(output, result)
            return result

        # Try to compute all declared outputs
        for output in self.recipebook.targets():
            resolve(output)

    def serve(self, output: list[str]) -> dict[str, Any]:
        """Retrieve computed outputs."""
        results = {}
        for out in output:
            if out in self.parse_cache:
                results[out] = self.parse_cache[out]
            else:
                msg = f"Output '{out}' not found in context."
                raise KeyError(msg)
        return results

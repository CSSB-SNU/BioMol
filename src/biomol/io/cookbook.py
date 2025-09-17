from typing import Any

from biomol.io.context import ParsingContext


class CookBook:
    def __init__(self, parse_context: ParsingContext):
        self.parse_context = parse_context
        self.recipes: dict[str, Any] = {}

    def add_recipe(self, output:str, instructions: callable, inputs:list[str]|str, params : dict[str, Any] | None) -> None:
        recipe = {
            "instructions": instructions,
            "inputs": inputs,
            "params": params,
        }
        self.recipes[output] = recipe

    def prep(self, data_dict: dict, fields: list[str]) -> None:
        """Prepare the context with initial data."""
        for field in fields:
            if field in data_dict:
                self.parse_context.add_data(field, data_dict[field])
            else:
                msg = f"Field {field} not found in data_dict."
                raise ValueError(msg)

    def cook(self) -> None:
        """Execute all recipes in dependency order."""
        visited = set()

        def resolve(output: str):
            # Already computed
            if output in self.parse_context:
                return self.parse_context[output]
            # Prevent infinite recursion
            if output in visited:
                msg = f"Cyclic dependency detected at '{output}'"
                raise RuntimeError(msg)
            visited.add(output)

            if output not in self.recipes:
                msg = f"No recipe to compute '{output}'. Available: {list(self.recipes.keys())}"
                raise KeyError(msg)

            recipe = self.recipes[output]
            inputs = []
            for inp in recipe["inputs"]:
                inputs.append(resolve(inp))  # recursively resolve

            params = recipe["params"] or {}
            result = recipe["instructions"](*inputs, **params)
            self.parse_context.add_data(output, result)
            return result

        # Try to compute all declared outputs
        for output in list(self.recipes.keys()):
            if output not in self.parse_context:
                resolve(output)

    def serve(self, output: list[str]) -> dict[str, Any]:
        """Retrieve computed outputs."""
        results = {}
        for out in output:
            if out in self.parse_context:
                results[out] = self.parse_context[out]
            else:
                msg = f"Output '{out}' not found in context."
                raise KeyError(msg)
        return results
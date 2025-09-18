import importlib.util
from pathlib import Path
from typing import Any, overload

from biomol.io.cache import ParsingCache
from biomol.io.recipe import RecipeBook


class Cooker:
    """
    Process input datas through a series of defined recipes.

    Args:
    parse_cache: An instance of ParsingCache to store intermediate and final results.
    recipebook: An instance of RecipeBook containing the recipes to execute.
    """

    @overload
    def __init__(self, parse_cache: ParsingCache, recipebook: RecipeBook) -> None: ...
    @overload
    def __init__(self, parse_cache: ParsingCache, recipebook: str) -> None: ...

    def __init__(self, parse_cache: ParsingCache, recipebook: RecipeBook | str) -> None:
        self.parse_cache = parse_cache
        if isinstance(recipebook, str):
            self.recipebook = self._load_recipe(recipebook)
        else:
            self.recipebook = recipebook

    def _load_recipe(self, recipebook_strpath: str) -> RecipeBook:
        """Dynamically load a RecipeBook from a given path."""
        recipebook_path = Path(recipebook_strpath).resolve()
        if not recipebook_path.exists():
            msg = f"RecipeBook file '{recipebook_path}' does not exist."
            raise FileNotFoundError(msg)

        module_name = recipebook_path.stem
        spec = importlib.util.spec_from_file_location(module_name, recipebook_path)
        if spec is None or spec.loader is None:
            msg = f"Could not load module from '{recipebook_path}'."
            raise ImportError(msg)
        recipe_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recipe_module)
        recipebook = getattr(recipe_module, "RECIPE", None)
        if recipebook is None or not isinstance(recipebook, RecipeBook):
            msg = f"'RECIPE' not found or invalid in '{recipebook_path}'."
            raise AttributeError(msg)

        return recipebook

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
            target_type = recipe.target[output]
            self.parse_cache.add_data(output, (target_type)(result))
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

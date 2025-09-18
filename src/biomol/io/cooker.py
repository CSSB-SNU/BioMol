import importlib.util
import inspect
from pathlib import Path
from typing import Any, overload

from biomol.io.cache import ParsingCache
from biomol.io.recipe import Constant, RecipeBook


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

    def prep(self, data_dict: dict, fields: list[str] | None = None) -> None:
        """Prepare the context with initial data."""
        if fields is None:
            fields = list(data_dict.keys())
        for field in fields:
            if field in data_dict:
                self.parse_cache.add_data(field, data_dict[field])
            else:
                msg = f"Field {field} not found in data_dict."
                raise ValueError(msg)

    def cook(self) -> None:
        """Execute all recipes in dependency order."""
        visited = set()

        def resolve(target: str | Constant | type) -> object:
            """Recursively resolve dependencies and compute the target."""
            # Already computed
            if isinstance(target, Constant):
                return target.value
            if isinstance(target, type):
                return target
            if target in self.parse_cache:
                return self.parse_cache[target]
            # Prevent infinite recursion
            if target in visited:
                msg = f"Cyclic dependency detected at '{target}'"
                raise RuntimeError(msg)
            visited.add(target)

            recipe, target_list = self.recipebook[target]
            resolved_inputs = {k: resolve(v) for k, v in recipe.inputs.items()}

            sig = inspect.signature(recipe.instruction)
            params = list(sig.parameters.values())
            has_varpos = any(p.kind == p.VAR_POSITIONAL for p in params)

            if has_varpos:
                after_varpos = False
                kwonly_names: set[str] = set()
                for p in params:
                    if p.kind == p.VAR_POSITIONAL:
                        after_varpos = True
                        continue
                    if after_varpos:
                        kwonly_names.add(p.name)

                kwargs = {k: v for k, v in resolved_inputs.items() if k in kwonly_names}
                args = [v for k, v in resolved_inputs.items() if k not in kwonly_names]
                result = recipe.instruction(*args, **kwargs)
            else:
                result = recipe.instruction(**resolved_inputs)
            if isinstance(result, tuple) and len(result) != len(target_list):
                msg = (
                    f"Instruction for target '{target}' returned {len(result)} values, "
                    f"but {len(target_list)} were expected."
                )
                raise ValueError(msg)
            if not isinstance(result, tuple):
                self.parse_cache.add_data(target, result)
                return result
            output = None
            for tgt, res in zip(target_list, result, strict=True):
                if tgt == target:
                    output = res
                self.parse_cache.add_data(tgt, res)
            return output

        # Try to compute all declared targets
        for target in self.recipebook.targets():
            resolve(target)

    def serve(self, targets: list[str]) -> dict[str, Any]:
        """Retrieve computed targets."""
        results = {}
        for out in targets:
            if out in self.parse_cache:
                results[out] = self.parse_cache[out]
            else:
                msg = f"targets '{out}' not found in context."
                raise KeyError(msg)
        return results

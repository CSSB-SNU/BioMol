import fnmatch
import importlib.util
from pathlib import Path
from typing import Any, get_args, overload

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
            self.recipebook, self.targets = self._load_recipe(recipebook)
        else:
            self.recipebook, self.targets = recipebook

    def _load_recipe(
        self, recipebook_strpath: str
    ) -> tuple[RecipeBook, list[str] | str | None]:
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
        targets = getattr(recipe_module, "TARGETS", None)
        if recipebook is None or not isinstance(recipebook, RecipeBook):
            msg = f"'RECIPE' not found or invalid in '{recipebook_path}'."
            raise AttributeError(msg)

        return recipebook, targets

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

    def _expand_wildcard_args(self, pattern: str) -> list[tuple[str, Any]]:
        """
        Expand wildcard-based arguments by matching keys from the cache.

        Notes
        -----
        - This method applies *only* to args, not kwargs.
        - Wildcard matching uses fnmatch (glob-style, e.g. "input*").
        - Importantly, the search is performed **only on keys already present
          in the parse_cache**, not on the recipebook targets.
        """
        matches: list[tuple[str, Any]] = []
        for key in self.parse_cache:
            if fnmatch.fnmatch(key, pattern):
                matches.append((key, self.parse_cache[key]))
        return matches

    def cook(self) -> None:
        """Execute all recipes in dependency order."""
        visited = set()

        def resolve(target_name: str, target_type: type) -> object:
            """Recursively resolve dependencies and compute the target."""
            # Already computed
            if target_name in self.parse_cache:
                return self.parse_cache[target_name]
            # Prevent infinite recursion
            if target_name in visited:
                msg = f"Cyclic dependency detected at '{target_name}'"
                raise RuntimeError(msg)
            if target_name not in self.recipebook and type(None) in get_args(
                target_type,
            ):
                return None
            visited.add(target_name)
            recipe = self.recipebook[target_name]
            resolved_args: list[Any] = []
            for var in recipe.inputs.args:
                if any(ch in var.name for ch in ["*", "?", "["]):  # detect glob pattern
                    wildcard_matches = self._expand_wildcard_args(var.name)
                    for _match_name, match_value in wildcard_matches:
                        resolved_args.append(match_value)
                else:
                    resolved_args.append(resolve(var.name, var.type))

            resolved_kwargs = {
                key: resolve(var.name, var.type)
                for key, var in recipe.inputs.kwargs.items()
            }
            final_kwargs = {**recipe.inputs.params, **resolved_kwargs}
            result = recipe.instruction(*resolved_args, **final_kwargs)

            visited.remove(target_name)
            target_names = [t.name for t in recipe.targets]

            if len(target_names) == 1:
                self.parse_cache.add_data(target_names[0], result)
                return result
            if not isinstance(result, tuple) or len(result) != len(target_names):
                msg = (
                    f"Instruction for targets {target_names} returned {type(result)}, "
                    f"but a tuple of length {len(target_names)} was expected."
                )
                raise ValueError(msg)

            output = None
            for name, value in zip(target_names, result, strict=True):
                self.parse_cache.add_data(name, value)
                if name == target_name:
                    output = value

            return output

        # Try to compute all declared targets
        for target in self.recipebook.targets():
            if target.name not in self.parse_cache:
                resolve(target.name, target.type)

    def serve(self, targets: list[str] | str | None = None) -> dict[str, Any]:
        """Retrieve computed targets."""
        results = {}
        if targets is None:
            targets = self.targets
        if isinstance(targets, str):
            return self.parse_cache[targets]
        for out in targets:
            if out in self.parse_cache:
                results[out] = self.parse_cache[out]
            else:
                msg = f"targets '{out}' not found in context."
                raise KeyError(msg)
        return results

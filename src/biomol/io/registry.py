from typing import Callable, List, Dict, Any
from biomol.io.context import ParsingContext
from biomol.core.feature import Feature


class MapperRegistry:
    """
    Registry for mapping functions that convert parsed fields into features.

    each python main process has a global registry for mappers.
    """

    _mappers: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """
        Register callable for mapping fields to features.

        Example:
        @MapperRegistry.register("identity")
        def identity(x) -> dict[str, Feature]:
            return {"x": Feature(...)}
        """

        def wrapper(func: Callable):
            cls._mappers[name] = func
            return func

        return wrapper

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._mappers:
            raise ValueError(f"Mapper '{name}' not found.")
        return cls._mappers[name]

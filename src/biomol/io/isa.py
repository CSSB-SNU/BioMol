from collections.abc import Callable
from typing import Any


class InstructionSet:
    """
    Registry for mapping functions that convert parsed fields into features.

    each python main process has a global registry for mappers.
    """

    _mappers: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Register callable for mapping fields to features.

        Example:
        @MapperRegistry.register("identity")
        def identity(x) -> dict[str, Feature]:
            return {"x": Feature(...)}
        """

        def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
            cls._mappers[name] = func
            return func

        return wrapper

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        """
        Retrieve a registered mapper function by name.

        Args:
            name: The name of the mapper to retrieve.

        Returns
        -------
            The callable mapper function.

        Raises
        ------
            ValueError: If no mapper with the given name is found.
        """
        if name not in cls._mappers:
            msg = f"Mapper '{name}' not found."
            raise ValueError(msg)
        return cls._mappers[name]

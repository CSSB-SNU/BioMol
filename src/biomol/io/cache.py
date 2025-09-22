from __future__ import annotations
from typing import Union
from typing import Any, overload


class ParsingCache:
    """
    Store necessary parsing input & temporary output while parsing a data.

    it store any typed data and its name.
    old name: ParsingContext
    """

    def __init__(self) -> None:
        self._storage: dict[str, Any] = {}

    def add_data(self, name: str, data: Any):
        """Store Data with a given name."""
        if name in self._storage:
            msg = f"Data with name '{name}' already exists in context."
            raise KeyError(msg)
        self._storage[name] = data

    def __contains__(self, name: str) -> bool:
        """Return True if name exists in context."""
        return name in self._storage

    @overload
    def __getitem__(self, query: str) -> Any: ...
    @overload
    def __getitem__(self, query: type) -> Any: ...

    def __getitem__(self, query: type | str) -> Any:
        """Get data by name or type"""
        if isinstance(query, str):
            if query not in self._storage:
                msg = f"Data with name '{query}' not found in context."
                raise KeyError(msg)
            return self._storage[query]

        return {k: v for k, v in self._storage.items() if isinstance(v, query)}

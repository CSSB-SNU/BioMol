from __future__ import annotations

from typing import Any


class ParsingCache:
    """
    Store necessary parsing input & temporary output while parsing a data.

    it store any typed data and its name.
    old name: ParsingContext
    """

    def __init__(self) -> None:
        self._storage: dict[str, Any] = {}

    def add_data(self, name: str, data: object) -> None:
        """Store Data with a given name."""
        if name in self._storage:
            msg = f"Data with name '{name}' already exists in context."
            raise KeyError(msg)
        self._storage[name] = data

    def __contains__(self, name: str) -> bool:
        """Return True if name exists in context."""
        return name in self._storage

    def __getitem__(self, name: str) -> object | None:
        """Get data by name."""
        if name not in self._storage:
            msg = f"Data with name '{name}' not found in context."
            raise KeyError(msg)
        return self._storage[name]

from __future__ import annotations

from typing import Any, Callable


class ParsingCache:
    """
    Store necessary parsing input & temporary output while parsing a data.

    This cache uses a key_transform function to interpret string keys.
    By default it treats keys as flat strings, but custom transforms
    can allow nested structures (e.g. dot notation).

    """

    def __init__(
        self,
        key_transform: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        self._storage: dict[str, Any] = {}
        if key_transform is None:
            self._key_transform = lambda k: (k,)
        else:
            self._key_transform = key_transform

    def add_data(self, name: str, data: object) -> None:
        """Store Data with a given name."""
        parts = self._key_transform(name)
        cur = self._storage
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
            if not isinstance(cur, dict):
                msg = f"Cannot create nested key {name}, {part} is not a dict."
                raise TypeError(msg)
        if parts[-1] in cur:
            msg = f"Data with name '{name}' already exists in context."
            raise KeyError(msg)
        cur[parts[-1]] = data

    def __contains__(self, name: str) -> bool:
        """Return True if name exists in context."""
        parts = self._key_transform(name)
        cur: Any = self._storage
        for part in parts:
            if not isinstance(cur, dict) or part not in cur:
                return False
            cur = cur[part]
        return True

    def __getitem__(self, name: str) -> object:
        """Get data by name."""
        parts = self._key_transform(name)
        cur: Any = self._storage
        for part in parts:
            if not isinstance(cur, dict) or part not in cur:
                msg = f"Data with name '{name}' not found in context."
                raise KeyError(msg)
            cur = cur[part]
        return cur

    def keys(self) -> list[str]:
        """Return a list of all keys (flattened back to strings)."""
        result: list[str] = []

        def _collect(d: dict[str, Any], prefix: str = "") -> None:
            for k, v in d.items():
                new_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _collect(v, new_key)
                else:
                    result.append(new_key)

        _collect(self._storage)
        return result

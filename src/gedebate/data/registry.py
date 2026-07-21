"""A tiny name -> object registry.

Used for generators, encoders, and tasks so each is pluggable by name and adding
one is a single decorated definition. Configs reference components by their
registered name; nothing else in the codebase needs to change.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str):
        """Decorator: register the decorated object under `name`."""

        def deco(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' already registered")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} '{name}'; known: {self.names()}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

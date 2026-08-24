from __future__ import annotations

from loopgraph_supervisor.harness.base import HarnessAdapter


class HarnessRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, HarnessAdapter] = {}

    def register(self, adapter: HarnessAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"Harness {adapter.name!r} is already registered")
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> HarnessAdapter:
        try:
            return self._adapters[name]
        except KeyError as error:
            raise KeyError(f"Harness {name!r} is not registered") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


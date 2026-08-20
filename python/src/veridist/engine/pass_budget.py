"""Lazy, auditable enforcement of data-source pass budgets."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from threading import Lock
from types import MappingProxyType
from typing import TypeVar

Item = TypeVar("Item")


class PassBudgetError(Exception):
    """A stable, localization-independent pass-budget failure."""

    __slots__ = ("code", "context")

    def __init__(self, code: str, context: Mapping[str, object]) -> None:
        super().__init__(code)
        self.code = code
        self.context: Mapping[str, object] = MappingProxyType(dict(context))


class PassEnforcer:
    """Count source iterator acquisitions and reject excess passes pre-read."""

    __slots__ = ("_actual_pass_count", "_lock", "_max_passes")

    def __init__(self, *, max_passes: int) -> None:
        if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer")
        self._max_passes = max_passes
        self._actual_pass_count = 0
        self._lock = Lock()

    @property
    def max_passes(self) -> int:
        """Return the immutable declared pass budget."""

        return self._max_passes

    @property
    def actual_pass_count(self) -> int:
        """Return the number of source iterator acquisitions reserved so far."""

        with self._lock:
            return self._actual_pass_count

    @property
    def provenance(self) -> Mapping[str, int]:
        """Return an immutable snapshot of declared and actual pass counts."""

        with self._lock:
            return MappingProxyType(
                {
                    "max_passes": self._max_passes,
                    "actual_pass_count": self._actual_pass_count,
                }
            )

    def begin_pass(self, source: Iterable[Item]) -> Iterator[Item]:
        """Reserve a pass before acquiring and returning the source iterator."""

        with self._lock:
            attempted_pass = self._actual_pass_count + 1
            if attempted_pass > self._max_passes:
                raise PassBudgetError(
                    "PASS_BUDGET_EXCEEDED",
                    {
                        "max_passes": self._max_passes,
                        "actual_pass_count": self._actual_pass_count,
                        "attempted_pass": attempted_pass,
                    },
                )
            self._actual_pass_count = attempted_pass
        return iter(source)


__all__ = ["PassBudgetError", "PassEnforcer"]

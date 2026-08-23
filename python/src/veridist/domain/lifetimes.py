"""Typed, locale-neutral lifetime observations for supported fit verticals."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import isfinite


def _finite_non_negative_time(value: float | Decimal) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, Decimal, int)):
        raise TypeError("time must be a finite numeric value")
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError("time must be finite and non-negative")
    return numeric


@dataclass(frozen=True, slots=True)
class ExactLifetime:
    """An observed event time for a supported lifetime model."""

    time: float | Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _finite_non_negative_time(self.time))


@dataclass(frozen=True, slots=True)
class RightCensoredLifetime:
    """A right-censoring time under the declared independent-censoring assumption."""

    time: float | Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _finite_non_negative_time(self.time))


LifetimeObservation = ExactLifetime | RightCensoredLifetime


__all__ = ["ExactLifetime", "LifetimeObservation", "RightCensoredLifetime"]

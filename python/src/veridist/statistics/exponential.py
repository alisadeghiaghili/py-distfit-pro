"""Fixed-state reductions for the exponential lifetime vertical."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime


class _ReductionOverflow(Exception):
    """Internal overflow signal carrying only derived aggregate counts."""

    def __init__(self, observation_count: int, event_count: int) -> None:
        self.observation_count = observation_count
        self.event_count = event_count


def _require_count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ExponentialReductionState:
    """O(1) Neumaier-compensated sufficient statistics, with no raw observations."""

    observation_count: int
    event_count: int
    total_time: float
    compensation: float

    def __post_init__(self) -> None:
        _require_count(self.observation_count, "observation_count")
        _require_count(self.event_count, "event_count")
        if self.event_count > self.observation_count:
            raise ValueError("event_count cannot exceed observation_count")
        if not isfinite(self.total_time) or not isfinite(self.compensation):
            raise ValueError("reduction state must remain finite")

    @classmethod
    def empty(cls) -> ExponentialReductionState:
        """Return the neutral fixed-state reduction value."""

        return cls(0, 0, 0.0, 0.0)

    @property
    def summed_time(self) -> float:
        """Return the compensated total, failing if the final addition overflows."""

        value = self.total_time + self.compensation
        if not isfinite(value):
            raise _ReductionOverflow(self.observation_count, self.event_count)
        return value

    def _add_value(self, value: float, observation_increment: int, event_increment: int) -> ExponentialReductionState:
        updated_total = self.total_time + value
        if not isfinite(updated_total):
            raise _ReductionOverflow(
                self.observation_count + observation_increment,
                self.event_count + event_increment,
            )
        if abs(self.total_time) >= abs(value):
            updated_compensation = self.compensation + (self.total_time - updated_total + value)
        else:
            updated_compensation = self.compensation + (value - updated_total + self.total_time)
        if not isfinite(updated_compensation):
            raise _ReductionOverflow(
                self.observation_count + observation_increment,
                self.event_count + event_increment,
            )
        return ExponentialReductionState(
            self.observation_count + observation_increment,
            self.event_count + event_increment,
            updated_total,
            updated_compensation,
        )

    def add(self, observation: LifetimeObservation) -> ExponentialReductionState:
        """Return a new state after one validated exact or right-censored observation."""

        if type(observation) is ExactLifetime:
            return self._add_value(observation.time, 1, 1)
        if type(observation) is RightCensoredLifetime:
            return self._add_value(observation.time, 1, 0)
        raise TypeError("observation must be an exact or right-censored lifetime")

    def merge(self, other: ExponentialReductionState) -> ExponentialReductionState:
        """Merge two compatible states; partition order has a tolerance-only contract."""

        if type(other) is not ExponentialReductionState:
            raise TypeError("other must be an ExponentialReductionState")
        combined = self._add_value(other.total_time, other.observation_count, other.event_count)
        return combined._add_value(other.compensation, 0, 0)


def reduce_exponential_chunks(
    chunks: Iterable[Iterable[LifetimeObservation]],
) -> ExponentialReductionState:
    """Reduce ragged chunks in their caller-declared canonical input order."""

    if not isinstance(chunks, Iterable):
        raise TypeError("chunks must be an iterable of observation iterables")
    state = ExponentialReductionState.empty()
    for chunk in chunks:
        if not isinstance(chunk, Iterable):
            raise TypeError("each chunk must be an iterable of lifetime observations")
        for observation in chunk:
            state = state.add(observation)
    return state


__all__ = ["ExponentialReductionState", "reduce_exponential_chunks"]

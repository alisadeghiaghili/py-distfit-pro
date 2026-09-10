"""Shared validation and deterministic scalar optimization for reliability MLEs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import isfinite

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime
from veridist.engine.errors import CapabilityCode, CapabilityError


def admitted_observations(
    observations: Iterable[LifetimeObservation],
    *,
    frequency_weights: Iterable[int] | None,
    analytic_weights: object | None,
    censoring: str,
    truncation: object | None,
) -> tuple[LifetimeObservation, ...]:
    """Validate v1 semantics and expand integer frequencies exactly."""

    if analytic_weights is not None:
        raise CapabilityError(CapabilityCode.ANALYTIC_WEIGHTS_UNSUPPORTED)
    if truncation is not None:
        raise CapabilityError(CapabilityCode.TRUNCATION_UNSUPPORTED)
    unsupported = {"interval": CapabilityCode.INTERVAL_CENSORING_UNSUPPORTED,
                   "left": CapabilityCode.LEFT_CENSORING_UNSUPPORTED}
    if censoring in unsupported:
        raise CapabilityError(unsupported[censoring])
    if censoring != "right":
        raise ValueError("censoring must be 'right', 'left', or 'interval'")
    values = tuple(observations)
    if any(type(value) not in {ExactLifetime, RightCensoredLifetime} for value in values):
        raise TypeError("observations must be exact or independently right-censored lifetimes")
    if frequency_weights is None:
        return values
    weights = tuple(frequency_weights)
    if len(weights) != len(values):
        raise ValueError("frequency_weights must match observations")
    expanded: list[LifetimeObservation] = []
    for value, weight in zip(values, weights, strict=True):
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 0:
            raise TypeError("frequency_weights must be non-negative built-in integers")
        expanded.extend((value,) * weight)
    return tuple(expanded)


def bounded_maximize(
    objective: Callable[[float], float], *, lower: float, upper: float, steps: int = 80
) -> tuple[float, float]:
    """Maximize a finite scalar objective by deterministic golden-section search."""

    if not (isfinite(lower) and isfinite(upper) and lower < upper):
        raise ValueError("optimization bounds must be finite and ordered")
    ratio = (5.0**0.5 - 1.0) / 2.0
    left, right = lower, upper
    first, second = right - ratio * (right - left), left + ratio * (right - left)
    first_value, second_value = objective(first), objective(second)
    for _ in range(steps):
        if first_value < second_value:
            left, first, first_value = first, second, second_value
            second = left + ratio * (right - left)
            second_value = objective(second)
        else:
            right, second, second_value = second, first, first_value
            first = right - ratio * (right - left)
            first_value = objective(first)
    point = (left + right) / 2.0
    return point, objective(point)


def positive_support(values: tuple[LifetimeObservation, ...]) -> bool:
    """Return whether all fixed-location reliability times are strictly positive."""

    return all(isfinite(value.time) and value.time > 0.0 for value in values)


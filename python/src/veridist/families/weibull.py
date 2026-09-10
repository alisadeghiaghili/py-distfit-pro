"""Fixed-location Weibull-min MLE cell for exact/right-censored lifetimes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import exp, isfinite, log

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation
from veridist.families._reliability import admitted_observations, bounded_maximize, positive_support


class WeibullFitFailureCode(StrEnum):
    """Reasons a finite Weibull point estimate is unavailable."""

    EMPTY_SAMPLE = "EMPTY_SAMPLE"
    NO_OBSERVED_EVENTS = "NO_OBSERVED_EVENTS"
    INVALID_SUPPORT = "INVALID_SUPPORT"
    OPTIMIZER_EXHAUSTED = "OPTIMIZER_EXHAUSTED"


@dataclass(frozen=True, slots=True)
class WeibullFitFailure:
    code: WeibullFitFailureCode
    observation_count: int
    event_count: int
    censored_count: int
    complete: bool = False
    converged: bool = False
    restart_failures: int = 0


@dataclass(frozen=True, slots=True)
class WeibullFitSuccess:
    shape: float
    scale: float
    log_likelihood: float
    observation_count: int
    event_count: int
    censored_count: int
    converged: bool = True
    restart_failures: int = 0
    complete: bool = True
    family: str = "weibull_min"
    location: float = 0.0

    def __post_init__(self) -> None:
        if not (isfinite(self.shape) and self.shape > 0.0):
            raise ValueError("shape must be finite and positive")
        if not (isfinite(self.scale) and self.scale > 0.0 and isfinite(self.log_likelihood)):
            raise ValueError("scale and log likelihood must be finite")


WeibullFit = WeibullFitSuccess | WeibullFitFailure


def fit_weibull(
    observations: Iterable[LifetimeObservation],
    *,
    fixed_shape: float | None = None,
    frequency_weights: Iterable[int] | None = None,
    analytic_weights: object | None = None,
    censoring: str = "right",
    truncation: object | None = None,
) -> WeibullFit:
    """Fit a fixed-location Weibull-min MLE for admitted v1 observations."""

    values = admitted_observations(
        observations,
        frequency_weights=frequency_weights,
        analytic_weights=analytic_weights,
        censoring=censoring,
        truncation=truncation,
    )
    count = len(values)
    events = sum(type(value) is ExactLifetime for value in values)
    def failure(code: WeibullFitFailureCode) -> WeibullFitFailure:
        return WeibullFitFailure(code, count, events, count - events)
    if count == 0:
        return failure(WeibullFitFailureCode.EMPTY_SAMPLE)
    if events == 0:
        return failure(WeibullFitFailureCode.NO_OBSERVED_EVENTS)
    if not positive_support(values):
        return failure(WeibullFitFailureCode.INVALID_SUPPORT)
    log_times = tuple(log(value.time) for value in values)
    event_logs = tuple(log(value.time) for value in values if type(value) is ExactLifetime)

    def at_shape(shape: float) -> tuple[float, float]:
        total = sum(exp(shape * value) for value in log_times)
        scale = exp(log(total / events) / shape)
        likelihood = (
            events * log(shape)
            - events * shape * log(scale)
            + (shape - 1.0) * sum(event_logs)
            - total / scale**shape
        )
        return scale, likelihood

    try:
        if fixed_shape is None:
            def profile(log_shape: float) -> float:
                return at_shape(exp(log_shape))[1]

            log_shape, _ = bounded_maximize(profile, lower=-6.0, upper=6.0)
            shape = exp(log_shape)
        else:
            if isinstance(fixed_shape, bool) or not isinstance(fixed_shape, int | float):
                raise TypeError("fixed_shape must be a positive built-in real or None")
            shape = float(fixed_shape)
            if not isfinite(shape) or shape <= 0.0:
                raise ValueError("fixed_shape must be finite and positive")
        scale, likelihood = at_shape(shape)
    except (ArithmeticError, OverflowError, ValueError):
        return failure(WeibullFitFailureCode.OPTIMIZER_EXHAUSTED)
    if not (isfinite(scale) and scale > 0.0 and isfinite(likelihood)):
        return failure(WeibullFitFailureCode.OPTIMIZER_EXHAUSTED)
    return WeibullFitSuccess(shape, scale, likelihood, count, events, count - events)


__all__ = [
    "WeibullFit",
    "WeibullFitFailure",
    "WeibullFitFailureCode",
    "WeibullFitSuccess",
    "fit_weibull",
]

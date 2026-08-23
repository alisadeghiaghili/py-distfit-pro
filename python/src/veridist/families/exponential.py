"""Closed-form exponential rate MLE for exact and right-censored lifetimes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import fsum, log

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime


class ExponentialFitFailureCode(StrEnum):
    """Stable, locale-neutral reasons that no finite point estimate exists."""

    EMPTY_SAMPLE = "EMPTY_SAMPLE"
    NO_OBSERVED_EVENTS = "NO_OBSERVED_EVENTS"
    UNBOUNDED_LIKELIHOOD = "UNBOUNDED_LIKELIHOOD"


@dataclass(frozen=True, slots=True)
class ExponentialFitSuccess:
    """A finite rate-only exponential MLE with declared capability facts."""

    rate: float
    observation_count: int
    event_count: int
    total_time: float
    mean: float
    log_likelihood: float
    censored_count: int
    family: str = "exponential"
    parameterization: str = "rate"
    location: float = 0.0
    inference: str = "not_provided"
    censoring_assumption: str = "independent_right_censoring"


@dataclass(frozen=True, slots=True)
class ExponentialFitFailure:
    """A typed statistical non-estimate, distinct from an engine failure."""

    code: ExponentialFitFailureCode
    observation_count: int
    event_count: int
    total_time: float


ExponentialFit = ExponentialFitSuccess | ExponentialFitFailure


def fit_exponential(observations: Iterable[LifetimeObservation]) -> ExponentialFit:
    """Fit the fixed-location exponential rate by its closed-form likelihood MLE."""

    if not isinstance(observations, Iterable):
        raise TypeError("observations must be an iterable of lifetime observations")
    count = 0
    events = 0
    times: list[float] = []
    for observation in observations:
        if type(observation) is ExactLifetime:
            events += 1
        elif type(observation) is not RightCensoredLifetime:
            raise TypeError("observations must contain exact or right-censored lifetimes")
        count += 1
        times.append(observation.time)
    total_time = fsum(times)
    if count == 0:
        return ExponentialFitFailure(ExponentialFitFailureCode.EMPTY_SAMPLE, 0, 0, 0.0)
    if events == 0:
        return ExponentialFitFailure(
            ExponentialFitFailureCode.NO_OBSERVED_EVENTS, count, events, total_time
        )
    if total_time == 0.0:
        return ExponentialFitFailure(
            ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD, count, events, total_time
        )
    rate = events / total_time
    return ExponentialFitSuccess(
        rate,
        count,
        events,
        total_time,
        1.0 / rate,
        events * log(rate) - rate * total_time,
        count - events,
    )


__all__ = [
    "ExponentialFit",
    "ExponentialFitFailure",
    "ExponentialFitFailureCode",
    "ExponentialFitSuccess",
    "fit_exponential",
]

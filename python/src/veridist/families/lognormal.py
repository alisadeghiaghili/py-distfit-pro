"""Fixed-location Lognormal MLE cell for exact/right-censored lifetimes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import erfc, exp, isfinite, log, pi, sqrt

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation
from veridist.families._reliability import admitted_observations, bounded_maximize, positive_support


class LognormalFitFailureCode(StrEnum):
    """Reasons a finite Lognormal point estimate is unavailable."""

    EMPTY_SAMPLE = "EMPTY_SAMPLE"
    NO_OBSERVED_EVENTS = "NO_OBSERVED_EVENTS"
    INVALID_SUPPORT = "INVALID_SUPPORT"
    OPTIMIZER_EXHAUSTED = "OPTIMIZER_EXHAUSTED"


@dataclass(frozen=True, slots=True)
class LognormalFitFailure:
    code: LognormalFitFailureCode
    observation_count: int
    event_count: int
    censored_count: int
    complete: bool = False
    converged: bool = False
    restart_failures: int = 0


@dataclass(frozen=True, slots=True)
class LognormalFitSuccess:
    mu_log: float
    sigma_log: float
    log_likelihood: float
    observation_count: int
    event_count: int
    censored_count: int
    converged: bool = True
    restart_failures: int = 0
    complete: bool = True
    family: str = "lognormal"
    location: float = 0.0

    def __post_init__(self) -> None:
        if not (isfinite(self.mu_log) and isfinite(self.sigma_log) and self.sigma_log > 0.0):
            raise ValueError("lognormal location and scale must be finite, with positive scale")
        if not isfinite(self.log_likelihood):
            raise ValueError("log likelihood must be finite")


LognormalFit = LognormalFitSuccess | LognormalFitFailure


def _log_sf(time: float, mu: float, sigma: float) -> float:
    survival = erfc((log(time) - mu) / (sigma * sqrt(2.0))) / 2.0
    if survival <= 0.0:
        raise ValueError("right-censoring survival underflow")
    return log(survival)


def fit_lognormal(
    observations: Iterable[LifetimeObservation],
    *,
    frequency_weights: Iterable[int] | None = None,
    analytic_weights: object | None = None,
    censoring: str = "right",
    truncation: object | None = None,
) -> LognormalFit:
    """Fit a Lognormal MLE for admitted v1 exact/right-censored observations."""

    values = admitted_observations(
        observations,
        frequency_weights=frequency_weights,
        analytic_weights=analytic_weights,
        censoring=censoring,
        truncation=truncation,
    )
    count = len(values)
    events = sum(type(value) is ExactLifetime for value in values)
    def failure(code: LognormalFitFailureCode) -> LognormalFitFailure:
        return LognormalFitFailure(code, count, events, count - events)
    if count == 0:
        return failure(LognormalFitFailureCode.EMPTY_SAMPLE)
    if events == 0:
        return failure(LognormalFitFailureCode.NO_OBSERVED_EVENTS)
    if not positive_support(values):
        return failure(LognormalFitFailureCode.INVALID_SUPPORT)
    exact_logs = tuple(log(value.time) for value in values if type(value) is ExactLifetime)
    try:
        if events == count:
            mu = sum(exact_logs) / count
            sigma = sqrt(sum((value - mu) ** 2 for value in exact_logs) / count)
            if sigma <= 0.0:
                raise ValueError("degenerate lognormal sample")
            likelihood = sum(
                -log(value.time) - log(sigma) - 0.5 * log(2.0 * pi)
                - (log(value.time) - mu) ** 2 / (2.0 * sigma**2)
                for value in values
            )
        else:
            center = sum(exact_logs) / events

            def profile(mu: float) -> float:
                def at_log_sigma(log_sigma: float) -> float:
                    sigma = exp(log_sigma)
                    exact = sum(
                        -value - log(sigma) - 0.5 * log(2.0 * pi)
                        - (value - mu) ** 2 / (2.0 * sigma**2)
                        for value in exact_logs
                    )
                    censored = sum(
                        (_log_sf(float(value.time), mu, sigma)
                         for value in values if type(value) is not ExactLifetime),
                        0.0,
                    )
                    return exact + censored

                _, likelihood = bounded_maximize(at_log_sigma, lower=-6.0, upper=6.0)
                return likelihood

            mu, _ = bounded_maximize(profile, lower=center - 8.0, upper=center + 8.0)

            def final(log_sigma: float) -> float:
                sigma = exp(log_sigma)
                exact = sum(
                    -value - log(sigma) - 0.5 * log(2.0 * pi)
                    - (value - mu) ** 2 / (2.0 * sigma**2)
                    for value in exact_logs
                )
                censored = sum(
                    (_log_sf(float(value.time), mu, sigma)
                     for value in values if type(value) is not ExactLifetime),
                    0.0,
                )
                return exact + censored

            log_sigma, likelihood = bounded_maximize(final, lower=-6.0, upper=6.0)
            sigma = exp(log_sigma)
    except (ArithmeticError, OverflowError, ValueError):
        return failure(LognormalFitFailureCode.OPTIMIZER_EXHAUSTED)
    if not (isfinite(mu) and isfinite(sigma) and sigma > 0.0 and isfinite(likelihood)):
        return failure(LognormalFitFailureCode.OPTIMIZER_EXHAUSTED)
    return LognormalFitSuccess(mu, sigma, likelihood, count, events, count - events)


__all__ = [
    "LognormalFit",
    "LognormalFitFailure",
    "LognormalFitFailureCode",
    "LognormalFitSuccess",
    "fit_lognormal",
]

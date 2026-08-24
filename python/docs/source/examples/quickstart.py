"""Canonical executable exponential right-censoring documentation example."""

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.families.exponential import ExponentialFitSuccess, fit_exponential

FIT = fit_exponential((ExactLifetime(1.0), ExactLifetime(2.0), RightCensoredLifetime(1.0)))
assert isinstance(FIT, ExponentialFitSuccess)
EXAMPLE_RESULT = {
    "rate": repr(FIT.rate),
    "observation_count": str(FIT.observation_count),
    "event_count": str(FIT.event_count),
    "censored_count": str(FIT.censored_count),
}

if __name__ == "__main__":
    print(EXAMPLE_RESULT)

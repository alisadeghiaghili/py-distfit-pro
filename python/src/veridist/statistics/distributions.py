"""Scalar CDF, survival, quantile, and sampling operations for v1 families."""

from __future__ import annotations

from collections.abc import Mapping
from math import erfc, exp, expm1, isfinite, lgamma, log, log1p, sqrt
from typing import cast

from veridist.families.registry import FAMILY_REGISTRY

_SQRT_TWO = sqrt(2.0)
_SQRT_TWO_PI = sqrt(2.0 * 3.141592653589793)


def _parameters(family: str, parameters: Mapping[str, object]) -> Mapping[str, float]:
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    if family == "exponential":
        if set(parameters) != {"rate"}:
            raise TypeError("parameter keys must equal the canonical parameter tuple")
        rate = parameters["rate"]
        if type(rate) not in {int, float}:
            raise ValueError("rate must be finite and positive")
        numeric_rate = float(cast(int | float, rate))
        if not isfinite(numeric_rate) or numeric_rate <= 0.0:
            raise ValueError("rate must be finite and positive")
        return {"rate": numeric_rate}
    return FAMILY_REGISTRY.resolve(family).validate_parameters(**dict(parameters))


def _probability(value: float) -> None:
    if type(value) is not float or not isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            "probability must be a finite built-in float strictly between zero and one"
        )


def _finite_scalar(value: object) -> float:
    if type(value) not in {int, float}:
        raise TypeError("value must be a built-in real number")
    numeric = float(cast(int | float, value))
    if not isfinite(numeric):
        raise ValueError("value must be finite")
    return numeric


def _regularized_gamma(shape: float, value: float) -> float:
    """Regularized lower incomplete gamma using the convergent series/CF split."""

    if value <= 0.0:
        return 0.0
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        current = shape
        for _ in range(512):
            current += 1.0
            term *= value / current
            total += term
            if abs(term) <= abs(total) * 2e-16:
                break
        return min(1.0, total * exp(-value + shape * log(value) - lgamma(shape)))
    tiny = 1e-300
    denominator = value + 1.0 - shape
    continued = 1.0 / max(abs(denominator), tiny)
    result = continued
    for index in range(1, 512):
        coefficient = -index * (index - shape)
        denominator = coefficient * continued + value + 1.0 - shape + 2.0 * index
        denominator = max(abs(denominator), tiny)
        continued = 1.0 / denominator
        numerator = value + 1.0 - shape + 2.0 * index + coefficient / max(abs(result), tiny)
        numerator = max(abs(numerator), tiny)
        delta = numerator * continued
        result *= delta
        if abs(delta - 1.0) <= 2e-16:
            break
    upper = exp(-value + shape * log(value) - lgamma(shape)) * result
    return max(0.0, min(1.0, 1.0 - upper))


def cdf(family: str, value: object, parameters: Mapping[str, object]) -> float:
    """Return one finite scalar cumulative probability for a registered family."""

    point = _finite_scalar(value)
    parameter = _parameters(family, parameters)
    if family == "exponential":
        return 0.0 if point < 0.0 else -expm1(-parameter["rate"] * point)
    if family == "normal":
        return 0.5 * erfc(-(point - parameter["mu"]) / (parameter["sigma"] * _SQRT_TWO))
    if family == "gamma":
        return _regularized_gamma(parameter["shape"], point / parameter["scale"])
    if family == "weibull_min":
        if point <= 0.0:
            return 0.0
        return -expm1(-((point / parameter["scale"]) ** parameter["shape"]))
    if family == "lognormal":
        if point <= 0.0:
            return 0.0
        return 0.5 * erfc(
            -(log(point) - parameter["mu_log"]) / (parameter["sigma_log"] * _SQRT_TWO)
        )
    if family == "gumbel_right":
        return exp(-exp(-(point - parameter["location"]) / parameter["scale"]))
    raise AssertionError("registry resolution must reject unknown families")


def sf(family: str, value: object, parameters: Mapping[str, object]) -> float:
    """Return the stable scalar survival probability for a registered family."""

    point = _finite_scalar(value)
    parameter = _parameters(family, parameters)
    if family == "exponential":
        return 1.0 if point < 0.0 else exp(-parameter["rate"] * point)
    if family == "normal":
        return 0.5 * erfc((point - parameter["mu"]) / (parameter["sigma"] * _SQRT_TWO))
    if family == "gamma":
        return 1.0 - _regularized_gamma(parameter["shape"], point / parameter["scale"])
    if family == "weibull_min":
        return 1.0 if point <= 0.0 else exp(-((point / parameter["scale"]) ** parameter["shape"]))
    if family == "lognormal":
        return (
            1.0
            if point <= 0.0
            else 0.5
            * erfc((log(point) - parameter["mu_log"]) / (parameter["sigma_log"] * _SQRT_TWO))
        )
    if family == "gumbel_right":
        return -expm1(-exp(-(point - parameter["location"]) / parameter["scale"]))
    raise AssertionError("registry resolution must reject unknown families")


def _normal_ppf(probability: float) -> float:
    """Acklam's rational inverse-normal approximation, refined by Newton steps."""

    lower = 0.02425
    upper = 1.0 - lower
    a = (
        -39.69683028665376,
        220.9460984245205,
        -275.9285104469687,
        138.357751867269,
        -30.66479806614716,
        2.506628277459239,
    )
    b = (
        -54.47609879822406,
        161.5858368580409,
        -155.6989798598866,
        66.80131188771972,
        -13.28068155288572,
    )
    c = (
        -0.007784894002430293,
        -0.3223964580411365,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
    if probability < lower:
        q = sqrt(-2.0 * log(probability))
        numerator = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        result = numerator / denominator
    elif probability > upper:
        q = sqrt(-2.0 * log1p(-probability))
        numerator = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        result = -numerator / denominator
    else:
        q = probability - 0.5
        r = q * q
        numerator = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        denominator = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        result = numerator * q / denominator
    for _ in range(2):
        error = 0.5 * erfc(-result / _SQRT_TWO) - probability
        density = exp(-result * result / 2.0) / _SQRT_TWO_PI
        if density == 0.0:
            break
        result -= error / density
    return result


def _inverse_by_bisection(
    family: str, probability: float, parameters: Mapping[str, object]
) -> float:
    lower, upper = -1.0, 1.0
    while cdf(family, lower, parameters) > probability:
        lower *= 2.0
    while cdf(family, upper, parameters) < probability:
        upper *= 2.0
    for _ in range(120):
        midpoint = (lower + upper) / 2.0
        if cdf(family, midpoint, parameters) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def ppf(family: str, probability: float, parameters: Mapping[str, object]) -> float:
    """Return a scalar quantile for a strictly interior probability."""

    _probability(probability)
    parameter = _parameters(family, parameters)
    if family == "exponential":
        return -log1p(-probability) / parameter["rate"]
    if family == "normal":
        return parameter["mu"] + parameter["sigma"] * _normal_ppf(probability)
    if family == "weibull_min":
        return float(parameter["scale"] * (-log1p(-probability)) ** (1.0 / parameter["shape"]))
    if family == "lognormal":
        return exp(parameter["mu_log"] + parameter["sigma_log"] * _normal_ppf(probability))
    if family == "gumbel_right":
        return parameter["location"] - parameter["scale"] * log(-log(probability))
    return _inverse_by_bisection(family, probability, parameter)


def sample(family: str, size: int, parameters: Mapping[str, object], rng: object) -> object:
    """Sample with only the caller-owned NumPy generator as a randomness source."""

    import numpy as np

    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative built-in integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    parameter = _parameters(family, parameters)
    if family == "exponential":
        return rng.exponential(1.0 / parameter["rate"], size=size)
    if family == "normal":
        return rng.normal(parameter["mu"], parameter["sigma"], size=size)
    if family == "gamma":
        return rng.gamma(parameter["shape"], parameter["scale"], size=size)
    if family == "weibull_min":
        return parameter["scale"] * rng.weibull(parameter["shape"], size=size)
    if family == "lognormal":
        return rng.lognormal(parameter["mu_log"], parameter["sigma_log"], size=size)
    if family == "gumbel_right":
        return rng.gumbel(parameter["location"], parameter["scale"], size=size)
    raise AssertionError("registry resolution must reject unknown families")


__all__ = ["cdf", "ppf", "sample", "sf"]

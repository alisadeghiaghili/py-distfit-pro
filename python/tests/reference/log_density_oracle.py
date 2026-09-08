"""Independent exact-binary high-precision oracle used only by tests.

The oracle pins mpmath 1.3.0.  Every binary64 input is reconstructed from its
integer ratio, never decimal text.  Working precision is at least 100 decimal
digits and increases with the largest input exponent plus a 200-digit guard;
this gives 500+ digits for the targeted 1e308 cancellation cases.
"""

from __future__ import annotations

import math

import mpmath


def normal(observation: float, *, mu: float, sigma: float) -> float:
    with mpmath.workdps(_precision(observation, mu, sigma)):
        x, mean, spread = map(_mp, (observation, mu, sigma))
        z = (x - mean) / spread
        return float(-mpmath.log(2 * mpmath.pi) / 2 - mpmath.log(spread) - z**2 / 2)


def gamma(observation: float, *, shape: float, scale: float) -> float:
    with mpmath.workdps(_precision(observation, shape, scale)):
        x, a, theta = map(_mp, (observation, shape, scale))
        return float(
            (a - 1) * mpmath.log(x) - x / theta - mpmath.loggamma(a) - a * mpmath.log(theta)
        )


def weibull_min(observation: float, *, shape: float, scale: float) -> float:
    with mpmath.workdps(_precision(observation, shape, scale)):
        x, k, lam = map(_mp, (observation, shape, scale))
        ratio = x / lam
        return float(mpmath.log(k) - mpmath.log(lam) + (k - 1) * mpmath.log(ratio) - ratio**k)


def lognormal(observation: float, *, mu_log: float, sigma_log: float) -> float:
    with mpmath.workdps(_precision(observation, mu_log, sigma_log)):
        x, mean, spread = map(_mp, (observation, mu_log, sigma_log))
        z = (mpmath.log(x) - mean) / spread
        return float(
            -mpmath.log(x) - mpmath.log(spread) - mpmath.log(2 * mpmath.pi) / 2 - z**2 / 2
        )


def gumbel_right(observation: float, *, location: float, scale: float) -> float:
    with mpmath.workdps(_precision(observation, location, scale)):
        x, loc, spread = map(_mp, (observation, location, scale))
        z = (x - loc) / spread
        return float(-mpmath.log(spread) - z - mpmath.exp(-z))


def _mp(value: float) -> mpmath.mpf:
    numerator, denominator = value.as_integer_ratio()
    return mpmath.mpf(numerator) / denominator


def _precision(*values: float) -> int:
    exponents = (abs(math.floor(math.log10(abs(value)))) for value in values if value != 0.0)
    return max(100, 200 + max(exponents, default=0))

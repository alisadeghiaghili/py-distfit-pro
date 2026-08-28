"""Independent high-precision oracle used only by log-density tests.

The oracle uses mpmath 1.3.0 at 100 decimal digits.  It is deliberately
outside the production package and is not a legacy result or a production
formula fixture.  The test cases below are finite binary64 inputs; the oracle
receives their decimal text so its reference calculation is performed before
binary64 rounding by the implementation under test.
"""

from __future__ import annotations

import mpmath


mpmath.mp.dps = 100


def normal(observation: float, *, mu: float, sigma: float) -> float:
    x, mean, spread = map(_mp, (observation, mu, sigma))
    z = (x - mean) / spread
    return float(-mpmath.log(2 * mpmath.pi) / 2 - mpmath.log(spread) - z**2 / 2)


def gamma(observation: float, *, shape: float, scale: float) -> float:
    x, a, theta = map(_mp, (observation, shape, scale))
    return float((a - 1) * mpmath.log(x) - x / theta - mpmath.loggamma(a) - a * mpmath.log(theta))


def weibull_min(observation: float, *, shape: float, scale: float) -> float:
    x, k, lam = map(_mp, (observation, shape, scale))
    ratio = x / lam
    return float(mpmath.log(k) - mpmath.log(lam) + (k - 1) * mpmath.log(ratio) - ratio**k)


def lognormal(observation: float, *, mu_log: float, sigma_log: float) -> float:
    x, mean, spread = map(_mp, (observation, mu_log, sigma_log))
    z = (mpmath.log(x) - mean) / spread
    return float(-mpmath.log(x) - mpmath.log(spread) - mpmath.log(2 * mpmath.pi) / 2 - z**2 / 2)


def gumbel_right(observation: float, *, location: float, scale: float) -> float:
    x, loc, spread = map(_mp, (observation, location, scale))
    z = (x - loc) / spread
    return float(-mpmath.log(spread) - z - mpmath.exp(-z))


def _mp(value: float) -> mpmath.mpf:
    return mpmath.mpf(repr(value))

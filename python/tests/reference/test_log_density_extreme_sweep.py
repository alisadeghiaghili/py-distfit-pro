"""Deterministic adversarial sweep for scalar log-density numerical contracts."""

from __future__ import annotations

import math
import random
import unittest
from collections.abc import Callable
from typing import TypeAlias

from tests.reference import log_density_oracle as oracle
from veridist.families.registry import FamilyId

_Oracle: TypeAlias = Callable[..., float]
_Vector: TypeAlias = tuple[FamilyId, float, dict[str, float], _Oracle]


class LogDensityExtremeSweepTests(unittest.TestCase):
    """Probe cancellation, adjacency, and tails; this is not a mutation score."""

    def test_fixed_seed_ordinary_sweep_has_no_false_success_or_uncaught_error(self) -> None:
        from veridist.statistics.log_density import LogDensitySuccess, evaluate_log_density

        random_source = random.Random(904_120)
        for family, observation, parameters, reference in _finite_vectors(random_source):
            with self.subTest(family=family, observation=observation, parameters=parameters):
                expected = reference(observation, **parameters)
                self.assertTrue(math.isfinite(expected))
                result = evaluate_log_density(family, observation, **parameters)
                self.assertIsInstance(result, LogDensitySuccess)
                self.assertAlmostEqual(
                    result.log_density,
                    expected,
                    delta=_ordinary_tolerance(expected),
                )


def _finite_vectors(random_source: random.Random) -> tuple[_Vector, ...]:
    vectors: list[_Vector] = []
    exponents = (-250, -100, -20, -1, 0, 1, 20, 100, 250)
    for exponent in exponents:
        scale = math.ldexp(1.0 + random_source.random(), exponent)
        shape = 8.0 + 32.0 * random_source.random()
        gamma_x = math.nextafter(shape * scale, math.inf)
        vectors.append(
            (FamilyId.GAMMA, gamma_x, {"shape": shape, "scale": scale}, oracle.gamma)
        )
        weibull_x = math.nextafter(scale, math.inf)
        vectors.append(
            (
                FamilyId.WEIBULL_MIN,
                weibull_x,
                {"shape": 0.5 + random_source.random() * 4.0, "scale": scale},
                oracle.weibull_min,
            )
        )
        normal_scale = math.ldexp(1.0 + random_source.random(), max(-250, exponent // 2))
        normal_mu = math.ldexp(random_source.uniform(-2.0, 2.0), exponent)
        normal_x = math.nextafter(normal_mu, math.inf)
        vectors.append(
            (FamilyId.NORMAL, normal_x, {"mu": normal_mu, "sigma": normal_scale}, oracle.normal)
        )
        lognormal_x = math.ldexp(1.0 + random_source.random(), exponent)
        vectors.append(
            (
                FamilyId.LOGNORMAL,
                lognormal_x,
                {"mu_log": exponent / 4.0, "sigma_log": 1.5},
                oracle.lognormal,
            )
        )
        location = math.ldexp(random_source.uniform(-2.0, 2.0), exponent)
        gumbel_x = math.nextafter(location, math.inf)
        vectors.append(
            (
                FamilyId.GUMBEL_RIGHT,
                gumbel_x,
                {"location": location, "scale": normal_scale},
                oracle.gumbel_right,
            )
        )
    return tuple(vectors)


def _ordinary_tolerance(expected: float) -> float:
    return max(8.0 * math.ulp(expected), abs(expected) * 2.0e-14, 2.0e-14)

"""Independent numerical reference contracts for scalar log-density."""

from __future__ import annotations

import math
import unittest

from tests.reference import log_density_oracle as oracle
from veridist.families.registry import FamilyId


class LogDensityReferenceTests(unittest.TestCase):
    """Compare binary64 results to the test-only 100-digit oracle."""

    def test_reference_vectors_cover_interior_and_extreme_finite_cases(self) -> None:
        vectors = (
            (FamilyId.NORMAL, -30.0, {"mu": -2.5, "sigma": 3.25}, oracle.normal),
            (FamilyId.NORMAL, 1.0e150, {"mu": -1.0e149, "sigma": 2.5e149}, oracle.normal),
            (FamilyId.GAMMA, 1.0e-150, {"shape": 0.25, "scale": 2.0}, oracle.gamma),
            (FamilyId.GAMMA, 40.0, {"shape": 17.5, "scale": 0.75}, oracle.gamma),
            (FamilyId.WEIBULL_MIN, 1.0e-100, {"shape": 0.4, "scale": 3.0}, oracle.weibull_min),
            (FamilyId.WEIBULL_MIN, 15.0, {"shape": 2.75, "scale": 4.5}, oracle.weibull_min),
            (FamilyId.LOGNORMAL, 1.0e-150, {"mu_log": -20.0, "sigma_log": 2.0}, oracle.lognormal),
            (FamilyId.LOGNORMAL, 1.0e100, {"mu_log": 5.0, "sigma_log": 3.5}, oracle.lognormal),
            (FamilyId.GUMBEL_RIGHT, -20.0, {"location": -3.0, "scale": 4.0}, oracle.gumbel_right),
            (FamilyId.GUMBEL_RIGHT, 700.0, {"location": 0.5, "scale": 2.25}, oracle.gumbel_right),
        )
        for family, observation, parameters, reference in vectors:
            with self.subTest(family=family, observation=observation):
                actual = _success_value(family, observation, **parameters)
                expected = reference(observation, **parameters)
                self.assertTrue(math.isfinite(actual))
                self.assertAlmostEqual(actual, expected, delta=_tolerance(expected))

    def test_normal_location_scale_identity(self) -> None:
        x, mu, sigma = 8.5, -1.25, 2.75
        self.assertAlmostEqual(
            _success_value(FamilyId.NORMAL, x, mu=mu, sigma=sigma),
            _success_value(FamilyId.NORMAL, (x - mu) / sigma, mu=0.0, sigma=1.0) - math.log(sigma),
            delta=2.0e-14,
        )

    def test_gamma_exponential_and_special_shape_identities(self) -> None:
        x, scale = 3.75, 1.5
        self.assertAlmostEqual(
            _success_value(FamilyId.GAMMA, x, shape=1.0, scale=scale),
            -math.log(scale) - x / scale,
            delta=2.0e-14,
        )
        self.assertAlmostEqual(
            _success_value(FamilyId.GAMMA, x, shape=2.0, scale=scale),
            math.log(x) - 2.0 * math.log(scale) - x / scale,
            delta=2.0e-14,
        )
        self.assertAlmostEqual(
            _success_value(FamilyId.GAMMA, x, shape=0.5, scale=scale),
            -0.5 * math.log(math.pi * scale * x) - x / scale,
            delta=2.0e-14,
        )

    def test_gamma_mode_cancellation_uses_the_exact_binary64_inputs(self) -> None:
        cases = ((1.0e12, 1.0e12), (1.0e16, 1.0e16))
        for observation, shape in cases:
            with self.subTest(observation=observation):
                actual = _success_value(FamilyId.GAMMA, observation, shape=shape, scale=1.0)
                expected = oracle.gamma(observation, shape=shape, scale=1.0)
                self.assertAlmostEqual(actual, expected, delta=_tolerance(expected))

    def test_weibull_adjacent_ratio_with_huge_shape_is_never_a_false_finite_success(self) -> None:
        from veridist.statistics.log_density import (
            LogDensityErrorCode,
            LogDensityFailure,
            evaluate_log_density,
        )

        scale = 1.0e300
        observation = math.nextafter(scale, math.inf)
        result = evaluate_log_density(
            FamilyId.WEIBULL_MIN,
            observation,
            shape=1.0e308,
            scale=scale,
        )
        self.assertIsInstance(result, LogDensityFailure)
        self.assertEqual(result.code, LogDensityErrorCode.NUMERICAL_OVERFLOW)

    def test_weibull_shape_one_is_exponential(self) -> None:
        x, scale = 3.75, 1.5
        self.assertAlmostEqual(
            _success_value(FamilyId.WEIBULL_MIN, x, shape=1.0, scale=scale),
            -math.log(scale) - x / scale,
            delta=2.0e-14,
        )

    def test_lognormal_jacobian_identity(self) -> None:
        x, mu_log, sigma_log = 7.25, 0.75, 1.5
        self.assertAlmostEqual(
            _success_value(FamilyId.LOGNORMAL, x, mu_log=mu_log, sigma_log=sigma_log),
            _success_value(FamilyId.NORMAL, math.log(x), mu=mu_log, sigma=sigma_log) - math.log(x),
            delta=2.0e-14,
        )

    def test_gumbel_location_scale_and_mode_identity(self) -> None:
        x, location, scale = 9.0, 2.0, 3.5
        self.assertAlmostEqual(
            _success_value(FamilyId.GUMBEL_RIGHT, x, location=location, scale=scale),
            _success_value(FamilyId.GUMBEL_RIGHT, (x - location) / scale, location=0.0, scale=1.0)
            - math.log(scale),
            delta=2.0e-14,
        )
        self.assertAlmostEqual(
            _success_value(FamilyId.GUMBEL_RIGHT, location, location=location, scale=scale),
            -math.log(scale) - 1.0,
            delta=2.0e-14,
        )


def _success_value(family: FamilyId, observation: float, **parameters: float) -> float:
    from veridist.statistics.log_density import LogDensitySuccess, evaluate_log_density

    result = evaluate_log_density(family, observation, **parameters)
    if not isinstance(result, LogDensitySuccess):
        raise AssertionError(f"expected a finite success, got {result!r}")
    return result.log_density


def _tolerance(expected: float) -> float:
    return max(8.0 * math.ulp(expected), abs(expected) * 2.0e-14, 2.0e-14)

"""Exact-binary Gamma envelope for stable large-shape log-density evaluation."""

from __future__ import annotations

import math
import unittest

from tests.reference import log_density_oracle as oracle
from veridist.families.registry import FamilyId


class GammaEnvelopeTests(unittest.TestCase):
    """Finite Gamma truths retain the ordinary scalar tolerance everywhere tested."""

    def test_large_shape_delta_that_rounds_to_negative_one_is_typed_overflow(self) -> None:
        from veridist.statistics.log_density import (
            LogDensityErrorCode,
            LogDensityFailure,
            evaluate_log_density,
        )

        result = evaluate_log_density(
            FamilyId.GAMMA,
            float.fromhex("0x0.0000000000001p-1022"),
            shape=8.0,
            scale=1.0,
        )
        self.assertIsInstance(result, LogDensityFailure)
        self.assertEqual(result.code, LogDensityErrorCode.NUMERICAL_OVERFLOW)

    def test_large_shape_scale_delta_grid_matches_independent_oracle(self) -> None:
        from veridist.statistics.log_density import LogDensitySuccess, evaluate_log_density

        checked = 0
        for shape in (8.0, 1.0e4, 1.0e12, 1.0e16, 1.0e100, 1.0e307, 1.0e308):
            for scale in (1.0e-300, 1.0, 1.0e300):
                base = shape * scale
                if not math.isfinite(base) or base <= 0.0:
                    continue
                for delta in (
                    -0.9,
                    math.nextafter(-0.5, -math.inf),
                    -0.5,
                    0.0,
                    0.5,
                    math.nextafter(0.5, math.inf),
                    2.0,
                ):
                    observation = base * (1.0 + delta)
                    if not math.isfinite(observation) or observation <= 0.0:
                        continue
                    with self.subTest(shape=shape, scale=scale, delta=delta):
                        expected = oracle.gamma(observation, shape=shape, scale=scale)
                        self.assertTrue(math.isfinite(expected))
                        result = evaluate_log_density(
                            FamilyId.GAMMA,
                            observation,
                            shape=shape,
                            scale=scale,
                        )
                        self.assertIsInstance(result, LogDensitySuccess)
                        self.assertAlmostEqual(
                            result.log_density,
                            expected,
                            delta=_tolerance(expected),
                        )
                        checked += 1
        self.assertGreaterEqual(checked, 90)


def _tolerance(expected: float) -> float:
    return max(8.0 * math.ulp(expected), abs(expected) * 2.0e-14, 2.0e-14)

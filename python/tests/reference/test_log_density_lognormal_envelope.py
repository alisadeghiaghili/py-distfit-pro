"""Exact-binary Lognormal envelope for cancellation-sensitive tiny sigmas."""

from __future__ import annotations

import math
import unittest

from tests.reference import log_density_oracle as oracle
from veridist.families.registry import FamilyId


class LognormalEnvelopeTests(unittest.TestCase):
    """Classify finite and nonrepresentable exact-binary truths independently."""

    def test_nearby_log_centers_and_sigma_grid_match_or_overflow_deterministically(self) -> None:
        from veridist.statistics.log_density import (
            LogDensityErrorCode,
            LogDensityFailure,
            LogDensitySuccess,
            evaluate_log_density,
        )

        observations = (
            float.fromhex("0x1.0000000000000p-1022"),
            3.0,
            float.fromhex("0x1.fffffffffffffp+1023"),
        )
        sigmas = (float.fromhex("0x0.0000000000001p-1022"), 1.0e-18, 1.0e-16, 1.0e-12, 1.5)
        checked = 0
        finite_truths = 0
        overflow_truths = 0
        for x in observations:
            base = math.log(x)
            centers = (base, math.nextafter(base, -math.inf), math.nextafter(base, math.inf))
            for mean in centers:
                for sigma in sigmas:
                    with self.subTest(x=x, mean=mean, sigma=sigma):
                        expected = oracle.lognormal(x, mu_log=mean, sigma_log=sigma)
                        result = evaluate_log_density(
                            FamilyId.LOGNORMAL,
                            x,
                            mu_log=mean,
                            sigma_log=sigma,
                        )
                        checked += 1
                        if math.isfinite(expected):
                            self.assertIsInstance(result, LogDensitySuccess)
                            self.assertAlmostEqual(
                                result.log_density,
                                expected,
                                delta=_tolerance(expected),
                            )
                            finite_truths += 1
                        else:
                            self.assertIsInstance(result, LogDensityFailure)
                            self.assertEqual(result.code, LogDensityErrorCode.NUMERICAL_OVERFLOW)
                            overflow_truths += 1
        self.assertEqual(checked, len(observations) * 3 * len(sigmas))
        self.assertGreaterEqual(finite_truths, 20)
        self.assertGreaterEqual(overflow_truths, 9)


def _tolerance(expected: float) -> float:
    return max(8.0 * math.ulp(expected), abs(expected) * 2.0e-14, 2.0e-14)

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

        x = 3.0
        base = math.log(x)
        centers = (base, math.nextafter(base, -math.inf), math.nextafter(base, math.inf))
        sigmas = (float.fromhex("0x0.0000000000001p-1022"), 1.0e-18, 1.0e-16, 1.0e-12, 1.5)
        finite_truths = 0
        overflow_truths = 0
        for mean in centers:
            for sigma in sigmas:
                with self.subTest(mean=mean, sigma=sigma):
                    expected = oracle.lognormal(x, mu_log=mean, sigma_log=sigma)
                    result = evaluate_log_density(
                        FamilyId.LOGNORMAL,
                        x,
                        mu_log=mean,
                        sigma_log=sigma,
                    )
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
        self.assertGreaterEqual(finite_truths, 6)
        self.assertGreaterEqual(overflow_truths, 3)


def _tolerance(expected: float) -> float:
    return max(8.0 * math.ulp(expected), abs(expected) * 2.0e-14, 2.0e-14)

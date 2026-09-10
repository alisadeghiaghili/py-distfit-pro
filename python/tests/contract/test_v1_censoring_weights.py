"""Capability-boundary contracts for v1 censoring and weight semantics."""

from __future__ import annotations

import unittest

from veridist.domain.lifetimes import ExactLifetime


class V1CensoringWeightTests(unittest.TestCase):
    def test_frequency_weights_equal_explicit_replication(self) -> None:
        from veridist.families.weibull import fit_weibull

        weighted = fit_weibull(
            (ExactLifetime(1.0), ExactLifetime(4.0)), frequency_weights=(2, 1)
        )
        replicated = fit_weibull(
            (ExactLifetime(1.0), ExactLifetime(1.0), ExactLifetime(4.0))
        )
        self.assertAlmostEqual(weighted.shape, replicated.shape, places=12)
        self.assertAlmostEqual(weighted.scale, replicated.scale, places=12)
        self.assertAlmostEqual(weighted.log_likelihood, replicated.log_likelihood, places=12)

    def test_frequency_weights_reject_non_integral_negative_and_misaligned_values(self) -> None:
        from veridist.families.weibull import fit_weibull

        observations = (ExactLifetime(1.0), ExactLifetime(2.0))
        for weights, expected in (
            ((1,), ValueError),
            ((1, -1), TypeError),
            ((1, 0.5), TypeError),
            ((True, 1), TypeError),
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(expected):
                    fit_weibull(observations, frequency_weights=weights)

    def test_invalid_observation_and_optimizer_bounds_are_rejected(self) -> None:
        from veridist.families._reliability import bounded_maximize
        from veridist.families.weibull import fit_weibull

        with self.assertRaises(TypeError):
            fit_weibull((object(),))
        with self.assertRaises(ValueError):
            bounded_maximize(lambda value: value, lower=1.0, upper=1.0)

    def test_analytic_weights_and_undeclared_censoring_fail_loudly(self) -> None:
        from veridist.engine.errors import CapabilityError
        from veridist.families.weibull import fit_weibull

        with self.assertRaisesRegex(CapabilityError, "ANALYTIC_WEIGHTS_UNSUPPORTED"):
            fit_weibull((ExactLifetime(1.0),), analytic_weights=(0.5,))
        with self.assertRaisesRegex(CapabilityError, "INTERVAL_CENSORING_UNSUPPORTED"):
            fit_weibull(((1.0, 2.0),), censoring="interval")
        with self.assertRaisesRegex(CapabilityError, "LEFT_CENSORING_UNSUPPORTED"):
            fit_weibull(((1.0, 2.0),), censoring="left")
        with self.assertRaisesRegex(CapabilityError, "TRUNCATION_UNSUPPORTED"):
            fit_weibull((ExactLifetime(1.0),), truncation=(0.5, None))
        with self.assertRaises(ValueError):
            fit_weibull((ExactLifetime(1.0),), censoring="unknown")
        with self.assertRaises(TypeError):
            CapabilityError("ANALYTIC_WEIGHTS_UNSUPPORTED")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

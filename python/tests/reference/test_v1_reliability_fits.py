"""Independent invariants for the 0.7 reliability fitting cells."""

from __future__ import annotations

import math
import unittest

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime


class V1ReliabilityFitTests(unittest.TestCase):
    def test_weibull_shape_one_reduces_to_exponential_likelihood(self) -> None:
        from veridist.families.weibull import fit_weibull

        observations = (
            ExactLifetime(1.0),
            RightCensoredLifetime(2.0),
            ExactLifetime(3.0),
        )
        result = fit_weibull(observations, fixed_shape=1.0)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.shape, 1.0, places=15)
        self.assertAlmostEqual(result.scale, 3.0, places=15)
        expected = 2 * math.log(1.0 / 3.0) - 6.0 / 3.0
        self.assertAlmostEqual(result.log_likelihood, expected, places=14)
        self.assertEqual((result.event_count, result.censored_count), (2, 1))

    def test_lognormal_exact_fit_matches_log_space_closed_form(self) -> None:
        from veridist.families.lognormal import fit_lognormal

        observations = tuple(ExactLifetime(value) for value in (1.0, math.e, math.e**2))
        result = fit_lognormal(observations)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.mu_log, 1.0, places=14)
        self.assertAlmostEqual(result.sigma_log, math.sqrt(2.0 / 3.0), places=14)
        self.assertEqual(result.restart_failures, 0)

    def test_censored_cells_return_finite_declared_estimates(self) -> None:
        from veridist.families.lognormal import fit_lognormal
        from veridist.families.weibull import fit_weibull

        observations = (
            ExactLifetime(0.5),
            ExactLifetime(1.0),
            RightCensoredLifetime(2.0),
        )
        weibull = fit_weibull(observations)
        lognormal = fit_lognormal(observations)
        self.assertTrue(weibull.complete)
        self.assertTrue(lognormal.complete)
        self.assertTrue(math.isfinite(weibull.log_likelihood))
        self.assertTrue(math.isfinite(lognormal.log_likelihood))

    def test_typed_non_estimates_preserve_the_reason(self) -> None:
        from veridist.families.lognormal import LognormalFitFailureCode, fit_lognormal
        from veridist.families.weibull import WeibullFitFailureCode, fit_weibull

        all_censored = (RightCensoredLifetime(1.0), RightCensoredLifetime(2.0))
        zero_event = (ExactLifetime(0.0),)
        self.assertEqual(fit_weibull(()).code, WeibullFitFailureCode.EMPTY_SAMPLE)
        self.assertEqual(fit_lognormal(()).code, LognormalFitFailureCode.EMPTY_SAMPLE)
        self.assertEqual(
            fit_weibull(all_censored).code, WeibullFitFailureCode.NO_OBSERVED_EVENTS
        )
        self.assertEqual(
            fit_lognormal(all_censored).code, LognormalFitFailureCode.NO_OBSERVED_EVENTS
        )
        self.assertEqual(fit_weibull(zero_event).code, WeibullFitFailureCode.INVALID_SUPPORT)
        self.assertEqual(fit_lognormal(zero_event).code, LognormalFitFailureCode.INVALID_SUPPORT)

    def test_invalid_fixed_shape_and_degenerate_log_sample_are_non_estimates(self) -> None:
        from veridist.families.lognormal import LognormalFitFailureCode, fit_lognormal
        from veridist.families.weibull import WeibullFitFailureCode, fit_weibull

        self.assertEqual(
            fit_weibull((ExactLifetime(1.0),), fixed_shape=-1.0).code,
            WeibullFitFailureCode.OPTIMIZER_EXHAUSTED,
        )
        with self.assertRaises(TypeError):
            fit_weibull((ExactLifetime(1.0),), fixed_shape="one")
        self.assertEqual(
            fit_lognormal((ExactLifetime(1.0), ExactLifetime(1.0))).code,
            LognormalFitFailureCode.OPTIMIZER_EXHAUSTED,
        )

    def test_estimate_value_objects_reject_non_finite_parameters(self) -> None:
        from veridist.families.lognormal import LognormalFitSuccess
        from veridist.families.weibull import WeibullFitSuccess

        with self.assertRaises(ValueError):
            WeibullFitSuccess(0.0, 1.0, 0.0, 1, 1, 0)
        with self.assertRaises(ValueError):
            WeibullFitSuccess(1.0, float("inf"), 0.0, 1, 1, 0)
        with self.assertRaises(ValueError):
            LognormalFitSuccess(0.0, 0.0, 0.0, 1, 1, 0)
        with self.assertRaises(ValueError):
            LognormalFitSuccess(0.0, 1.0, float("nan"), 1, 1, 0)

    def test_right_tail_underflow_remains_visible_to_the_optimizer(self) -> None:
        from veridist.families.lognormal import _log_sf

        with self.assertRaises(ValueError):
            _log_sf(1e300, 0.0, 1e-6)


if __name__ == "__main__":
    unittest.main()

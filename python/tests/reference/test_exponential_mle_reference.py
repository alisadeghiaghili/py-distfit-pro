"""Independent reference contracts for the first exponential fit vertical."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from math import inf, nan
import unittest

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.families.exponential import (
    ExponentialFitFailure,
    ExponentialFitFailureCode,
    ExponentialFitSuccess,
    fit_exponential,
)


def rational_rate(events: int, total_time: str) -> float:
    """Compute an independent high-precision reference from a rational formula."""

    with localcontext() as context:
        context.prec = 50
        return float(Decimal(events) / Decimal(total_time))


def log_likelihood(events: int, total_time: str, rate: float) -> float:
    """Calculate the declared likelihood independently in Decimal arithmetic."""

    with localcontext() as context:
        context.prec = 50
        decimal_rate = Decimal(str(rate))
        return float(Decimal(events) * decimal_rate.ln() - decimal_rate * Decimal(total_time))


class ExponentialReferenceContracts(unittest.TestCase):
    """EXP01-EXP08: closed-form point-estimate and boundary contracts."""

    def test_exp01_complete_sample_uses_independent_rational_golden(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("1.5")), ExactLifetime(Decimal("2.5"))))

        self.assertIsInstance(result, ExponentialFitSuccess)
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.rate, rational_rate(2, "4.0"))
        self.assertEqual(result.mean, 2.0)
        self.assertEqual(result.log_likelihood, log_likelihood(2, "4.0", result.rate))
        self.assertEqual((result.observation_count, result.event_count, result.censored_count), (2, 2, 0))

    def test_exp02_right_censoring_contributes_time_not_event(self) -> None:
        result = fit_exponential(
            (ExactLifetime(Decimal("1")), RightCensoredLifetime(Decimal("3")))
        )

        self.assertIsInstance(result, ExponentialFitSuccess)
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.rate, rational_rate(1, "4"))
        self.assertEqual(result.log_likelihood, log_likelihood(1, "4", result.rate))
        self.assertEqual((result.observation_count, result.event_count, result.censored_count), (2, 1, 1))

    def test_exp03_mixed_sample_has_closed_form_mle(self) -> None:
        result = fit_exponential(
            (
                ExactLifetime(Decimal("0.25")),
                RightCensoredLifetime(Decimal("1.75")),
                ExactLifetime(Decimal("2")),
                RightCensoredLifetime(Decimal("6")),
            )
        )

        self.assertIsInstance(result, ExponentialFitSuccess)
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.rate, rational_rate(2, "10"))
        self.assertEqual(result.mean, 5.0)
        self.assertEqual(result.log_likelihood, log_likelihood(2, "10", result.rate))
        self.assertEqual((result.observation_count, result.event_count, result.censored_count), (4, 2, 2))

    def test_exp04_canonical_parameterization_is_rate_with_fixed_zero_location(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("2")),))

        self.assertIsInstance(result, ExponentialFitSuccess)
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.family, "exponential")
        self.assertEqual(result.parameterization, "rate")
        self.assertEqual(result.location, 0.0)
        self.assertEqual(result.mean, 2.0)

    def test_exp05_all_censored_is_a_typed_non_estimate(self) -> None:
        result = fit_exponential(
            (RightCensoredLifetime(Decimal("2")), RightCensoredLifetime(Decimal("5")))
        )

        self.assertEqual(
            result,
            ExponentialFitFailure(
                code=ExponentialFitFailureCode.NO_OBSERVED_EVENTS,
                observation_count=2,
                event_count=0,
                total_time=7.0,
            ),
        )

    def test_exp06_zero_total_time_event_is_typed_unbounded_likelihood(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("0")),))

        self.assertEqual(
            result,
            ExponentialFitFailure(
                code=ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD,
                observation_count=1,
                event_count=1,
                total_time=0.0,
            ),
        )

    def test_exp07_point_estimate_explicitly_does_not_provide_inference(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("4")),))

        self.assertIsInstance(result, ExponentialFitSuccess)
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.inference, "not_provided")
        self.assertEqual(result.censoring_assumption, "independent_right_censoring")

    def test_exp08_empty_input_is_a_typed_non_estimate(self) -> None:
        result = fit_exponential(())

        self.assertEqual(
            result,
            ExponentialFitFailure(
                code=ExponentialFitFailureCode.EMPTY_SAMPLE,
                observation_count=0,
                event_count=0,
                total_time=0.0,
            ),
        )

    def test_exp08_result_and_failure_shapes_are_frozen_and_slotted(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("1")),))
        failure = fit_exponential(())

        for value in (result, failure):
            self.assertTrue(hasattr(type(value), "__slots__"))
            with self.assertRaises((FrozenInstanceError, TypeError)):
                value.observation_count = 99  # type: ignore[misc]

    def test_exp08_validation_rejects_invalid_times_without_echoing_input(self) -> None:
        for value in (-1, nan, inf, -inf, True, "1"):
            with self.assertRaises((TypeError, ValueError)) as captured:
                ExactLifetime(value)  # type: ignore[arg-type]
            self.assertNotIn(str(value), str(captured.exception))
        with self.assertRaises(TypeError):
            fit_exponential((object(),))  # type: ignore[arg-type]

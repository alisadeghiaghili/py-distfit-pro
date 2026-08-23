"""Independent reference contracts for the first exponential fit vertical."""

from __future__ import annotations

from decimal import Decimal, localcontext
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


class ExponentialReferenceContracts(unittest.TestCase):
    """EXP01-EXP08: closed-form point-estimate and boundary contracts."""

    def test_exp01_complete_sample_uses_independent_rational_golden(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("1.5")), ExactLifetime(Decimal("2.5"))))

        self.assertIsInstance(result, ExponentialFitSuccess)
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.rate, rational_rate(2, "4.0"))

    def test_exp02_right_censoring_contributes_time_not_event(self) -> None:
        result = fit_exponential(
            (ExactLifetime(Decimal("1")), RightCensoredLifetime(Decimal("3")))
        )

        self.assertIsInstance(result, ExponentialFitSuccess)
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.rate, rational_rate(1, "4"))

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

    def test_exp04_canonical_parameterization_is_rate_with_fixed_zero_location(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("2")),))

        self.assertIsInstance(result, ExponentialFitSuccess)
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.family, "exponential")
        self.assertEqual(result.parameterization, "rate")
        self.assertEqual(result.location, 0.0)

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


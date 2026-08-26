"""RED execution contracts staged after the CSV adapter behavior is green."""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.contract.test_csv_lifetime_adapter_contract import (
    SCHEMA,
    SOURCE_ID,
    CsvLifetimeAdapterContracts,
)
from veridist.adapters.csv_lifetimes import CsvLifetimeLimits
from veridist.engine.errors import FailureCode
from veridist.engine.outcome import FailedOutcome, UnknownMissingRanges
from veridist.execution import (
    ExponentialSourceFitResult,
    fit_exponential_csv,
    fit_exponential_source,
)
from veridist.families.exponential import ExponentialFitFailure, ExponentialFitFailureCode


class CsvLifetimeExecutionContracts(unittest.TestCase):
    """The adapter is family-neutral; only this layer creates scientific fits."""

    def test_csv_exec01_header_only_is_complete_empty_sample(self) -> None:
        helper = CsvLifetimeAdapterContracts()
        adapter, source = helper.adapter(b"time,event_observed\n")

        result = fit_exponential_source(adapter)

        self.assertIsInstance(result, ExponentialSourceFitResult)
        assert isinstance(result, ExponentialSourceFitResult)
        self.assertIsInstance(result.fit, ExponentialFitFailure)
        assert isinstance(result.fit, ExponentialFitFailure)
        self.assertEqual(result.fit.code, ExponentialFitFailureCode.EMPTY_SAMPLE)
        self.assertTrue(result.execution.outcome.complete)
        self.assertEqual(source.close_count, 1)

    def test_csv_exec02_open_failure_is_closed_failed_outcome_without_fit(self) -> None:
        result = fit_exponential_csv(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
        )

        self.assertIsInstance(result, ExponentialSourceFitResult)
        assert isinstance(result, ExponentialSourceFitResult)
        self.assertIsNone(result.fit)
        self.assertIsInstance(result.execution.outcome, FailedOutcome)
        assert isinstance(result.execution.outcome, FailedOutcome)
        self.assertEqual(result.execution.outcome.failure.code, FailureCode.SOURCE_OPEN_FAILED)
        self.assertEqual(result.execution.outcome.failure.stage.value, "preflight")
        self.assertIsInstance(result.execution.outcome.coverage, UnknownMissingRanges)


if __name__ == "__main__":
    unittest.main()

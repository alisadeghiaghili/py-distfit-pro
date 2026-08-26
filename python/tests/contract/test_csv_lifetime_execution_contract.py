"""RED execution contracts staged after the CSV adapter behavior is green."""

from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path

from veridist.adapters.csv_lifetimes import (
    CsvLifetimeAdapter,
    CsvLifetimeLimits,
    CsvLifetimeSchema,
)
from veridist.engine.errors import FailureCode
from veridist.engine.outcome import FailedOutcome, UnknownMissingRanges
from veridist.engine.provenance import PublicSourceId
from veridist.execution import (
    ExponentialSourceFitResult,
    fit_exponential_csv,
    fit_exponential_source,
)
from veridist.families.exponential import ExponentialFitFailure, ExponentialFitFailureCode

SOURCE_ID = PublicSourceId("src_0123456789abcdef0123456789abcdef")
SCHEMA = CsvLifetimeSchema(time_column="time", event_observed_column="event_observed")


class _TrackingSource:
    """Local execution fixture; importing another test class duplicates collection."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.open_count = 0
        self.close_count = 0

    def identity(self, path: Path) -> object:
        del path
        return "fixture-revision"

    def open_binary(self, path: Path) -> BytesIO:
        del path
        self.open_count += 1
        return _CountingBytesIO(self.payload, self)


class _CountingBytesIO(BytesIO):
    def __init__(self, payload: bytes, owner: _TrackingSource) -> None:
        super().__init__(payload)
        self.owner = owner

    def close(self) -> None:
        if not self.closed:
            self.owner.close_count += 1
        super().close()


def _adapter(payload: bytes) -> tuple[CsvLifetimeAdapter, _TrackingSource]:
    source = _TrackingSource(payload)
    return (
        CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            SCHEMA,
            SOURCE_ID,
            CsvLifetimeLimits(2048, 4096),
            source,
        ),
        source,
    )


class CsvLifetimeExecutionContracts(unittest.TestCase):
    """The adapter is family-neutral; only this layer creates scientific fits."""

    def test_csv_exec01_header_only_is_complete_empty_sample(self) -> None:
        adapter, source = _adapter(b"time,event_observed\n")

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

    def test_csv_exec03_overflow_consumes_the_full_source_and_is_a_complete_nonestimate(
        self,
    ) -> None:
        adapter, source = _adapter(b"time,event_observed\n1e308,1\n1e308,0\n1,1\n")

        result = fit_exponential_source(adapter)

        self.assertTrue(result.execution.outcome.complete)
        self.assertIsInstance(result.fit, ExponentialFitFailure)
        assert isinstance(result.fit, ExponentialFitFailure)
        self.assertEqual(result.fit.code, ExponentialFitFailureCode.NUMERICAL_OVERFLOW)
        self.assertEqual((result.fit.observation_count, result.fit.event_count), (3, 2))
        self.assertEqual(source.open_count, 1)
        self.assertEqual(source.close_count, 1)


if __name__ == "__main__":
    unittest.main()

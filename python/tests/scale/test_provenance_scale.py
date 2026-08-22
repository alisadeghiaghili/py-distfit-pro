"""DS-12 structural scale evidence for provenance metadata."""

from __future__ import annotations

import unittest

from tests.contract.test_execution_provenance import provenance
from veridist.engine.errors import FailureCode
from veridist.engine.outcome import (
    CompleteOutcome,
    FailureRecord,
    FailureStage,
    KnownCoverage,
    KnownExtent,
    PartialOutcome,
    RowRange,
)
from veridist.engine.provenance import ExecutionReport, to_canonical_json_bytes


class ProvenanceScaleTests(unittest.TestCase):
    def test_ds12_sequential_prefix_does_not_retain_per_chunk_metadata(self) -> None:
        reports = []
        for logical_chunks in (10, 1_000_000):
            coverage = KnownCoverage(
                KnownExtent(0, logical_chunks),
                (RowRange(0, logical_chunks),),
                logical_chunks,
                0,
            )
            reports.append(ExecutionReport(CompleteOutcome(coverage), provenance()))

        small, large = reports
        self.assertEqual(len(small.outcome.coverage.processed_ranges), 1)
        self.assertEqual(len(large.outcome.coverage.processed_ranges), 1)
        small_size = len(to_canonical_json_bytes(small))
        large_size = len(to_canonical_json_bytes(large))
        digit_growth = len(str(1_000_000)) - len(str(10))
        self.assertLessEqual(large_size - small_size, 4 * digit_growth)

    def test_ds12_arbitrary_gaps_are_explicitly_linear_in_gap_count(self) -> None:
        small_ranges = (RowRange(0, 1),)
        large_ranges = tuple(RowRange(index * 2, index * 2 + 1) for index in range(100))
        failure = FailureRecord(FailureCode.MISSING_CHUNK, FailureStage.DELIVERY)
        small = ExecutionReport(
            PartialOutcome(KnownCoverage(KnownExtent(0, 2), small_ranges, 1, 0), failure),
            provenance(),
        )
        large = ExecutionReport(
            PartialOutcome(KnownCoverage(KnownExtent(0, 200), large_ranges, 100, 0), failure),
            provenance(),
        )

        self.assertEqual(len(large.outcome.coverage.processed_ranges), 100)
        self.assertEqual(len(large.outcome.coverage.missing_ranges), 100)
        self.assertGreater(
            len(to_canonical_json_bytes(large)),
            len(to_canonical_json_bytes(small)) * 2,
        )


if __name__ == "__main__":
    unittest.main()

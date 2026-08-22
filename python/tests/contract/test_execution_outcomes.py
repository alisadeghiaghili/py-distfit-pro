"""DS-11 contracts for immutable, honestly labelled execution outcomes."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from math import inf

from veridist.engine.errors import FailureCode
from veridist.engine.outcome import (
    CompleteOutcome,
    FailedOutcome,
    FailureRecord,
    FailureStage,
    KnownCoverage,
    KnownExtent,
    PartialOutcome,
    RowRange,
    UnknownMissingRanges,
    classify_execution_outcome,
)


def known(
    stop: int,
    *processed: RowRange,
    accepted_chunk_count: int = 1,
    empty_chunk_count: int = 0,
) -> KnownCoverage:
    return KnownCoverage(
        extent=KnownExtent(0, stop),
        processed_ranges=processed,
        accepted_chunk_count=accepted_chunk_count,
        empty_chunk_count=empty_chunk_count,
    )


class ExecutionOutcomeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cancelled = FailureRecord(FailureCode.CANCELLED, FailureStage.CANCELLATION)

    def test_ds11_ranges_are_nonempty_half_open_integer_intervals(self) -> None:
        self.assertEqual(RowRange(2, 5).row_count, 3)
        self.assertEqual(KnownExtent(2, 2).row_count, 0)

        invalid_ranges = (
            (-1, 1),
            (1, 1),
            (2, 1),
            (False, 1),
            (0, True),
            (0, inf),
            (None, 1),
        )
        for start, stop in invalid_ranges:
            with self.subTest(start=start, stop=stop), self.assertRaises((TypeError, ValueError)):
                RowRange(start, stop)  # type: ignore[arg-type]

        invalid_extents = ((-1, 0), (2, 1), (False, 0), (0, True), (0, inf))
        for start, stop in invalid_extents:
            with self.subTest(start=start, stop=stop), self.assertRaises((TypeError, ValueError)):
                KnownExtent(start, stop)  # type: ignore[arg-type]

    def test_ds11_processed_ranges_are_canonical_without_silent_overlap_repair(self) -> None:
        coverage = known(8, RowRange(0, 2), RowRange(2, 5), RowRange(6, 8))
        self.assertEqual(coverage.processed_ranges, (RowRange(0, 5), RowRange(6, 8)))

        invalid = (
            (RowRange(3, 4), RowRange(0, 2)),
            (RowRange(0, 3), RowRange(2, 4)),
            (RowRange(0, 2), RowRange(0, 2)),
        )
        for ranges in invalid:
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                known(5, *ranges)

        with self.assertRaises(TypeError):
            KnownCoverage(KnownExtent(0, 2), [RowRange(0, 1)], 1, 0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            KnownCoverage(KnownExtent(0, 2), (object(),), 1, 0)  # type: ignore[arg-type]

    def test_ds11_known_coverage_derives_the_exact_complement(self) -> None:
        coverage = known(
            10,
            RowRange(0, 2),
            RowRange(4, 6),
            RowRange(8, 10),
            accepted_chunk_count=3,
        )

        self.assertEqual(coverage.processed_row_count, 6)
        self.assertEqual(coverage.missing_ranges, (RowRange(2, 4), RowRange(6, 8)))
        self.assertEqual(coverage.missing_row_count, 4)

        offset_coverage = KnownCoverage(
            KnownExtent(3, 10),
            (RowRange(4, 6),),
            1,
            0,
        )
        self.assertEqual(
            offset_coverage.missing_ranges,
            (RowRange(3, 4), RowRange(6, 10)),
        )

    def test_ds11_ranges_must_remain_inside_the_known_extent(self) -> None:
        for ranges in (
            (RowRange(0, 2),),
            (RowRange(3, 6),),
        ):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                KnownCoverage(KnownExtent(1, 5), ranges, 1, 0)
        with self.assertRaises(TypeError):
            KnownCoverage("0:5", (), 0, 0)  # type: ignore[arg-type]

    def test_ds11_empty_source_can_complete_without_fabricating_a_range(self) -> None:
        coverage = known(0, accepted_chunk_count=1, empty_chunk_count=1)
        outcome = CompleteOutcome(coverage)

        self.assertTrue(outcome.complete)
        self.assertEqual(coverage.processed_ranges, ())
        self.assertEqual(coverage.missing_ranges, ())

        no_chunk_outcome = CompleteOutcome(known(0, accepted_chunk_count=0))
        self.assertEqual(no_chunk_outcome.coverage.accepted_chunk_count, 0)

    def test_ds11_complete_requires_no_missing_rows(self) -> None:
        coverage = known(3, RowRange(0, 3))
        outcome = CompleteOutcome(coverage)

        self.assertTrue(outcome.complete)
        self.assertEqual(outcome.status.value, "complete")
        self.assertFalse(hasattr(outcome, "failure"))
        with self.assertRaises(ValueError):
            CompleteOutcome(known(3, RowRange(0, 2)))
        with self.assertRaises(TypeError):
            CompleteOutcome(UnknownMissingRanges((), 0, 0))  # type: ignore[arg-type]

    def test_ds11_partial_is_labelled_and_has_no_scientific_value(self) -> None:
        coverage = known(5, RowRange(0, 2))
        outcome = PartialOutcome(coverage, self.cancelled)

        self.assertFalse(outcome.complete)
        self.assertEqual(outcome.status.value, "partial")
        self.assertEqual(outcome.coverage.missing_ranges, (RowRange(2, 5),))
        self.assertIs(outcome.failure, self.cancelled)
        self.assertFalse(hasattr(outcome, "value"))
        with self.assertRaises(TypeError):
            PartialOutcome(coverage, "CANCELLED")  # type: ignore[arg-type]

    def test_ds11_partial_rejects_zero_progress_or_fully_processed_data(self) -> None:
        for coverage in (known(5), known(5, RowRange(0, 5))):
            with self.subTest(coverage=coverage), self.assertRaises(ValueError):
                PartialOutcome(coverage, self.cancelled)

    def test_ds11_failure_after_all_data_is_failed_not_partial(self) -> None:
        coverage = known(5, RowRange(0, 5))
        outcome = classify_execution_outcome(coverage, self.cancelled)

        self.assertIsInstance(outcome, FailedOutcome)
        self.assertFalse(outcome.complete)
        self.assertEqual(outcome.status.value, "failed")
        self.assertEqual(outcome.coverage.missing_ranges, ())

    def test_ds11_unknown_extent_is_failed_with_both_typed_reasons(self) -> None:
        coverage = UnknownMissingRanges(
            processed_ranges=(RowRange(0, 4),),
            accepted_chunk_count=2,
            empty_chunk_count=1,
        )
        outcome = classify_execution_outcome(coverage, self.cancelled)

        self.assertIsInstance(outcome, FailedOutcome)
        self.assertIs(outcome.failure.code, FailureCode.CANCELLED)
        self.assertIs(coverage.reason, FailureCode.MISSING_RANGE_UNKNOWN)
        self.assertFalse(hasattr(coverage, "missing_ranges"))
        self.assertFalse(hasattr(coverage, "expected_row_count"))

    def test_ds11_unknown_extent_cannot_complete_or_be_partial(self) -> None:
        coverage = UnknownMissingRanges((), 0, 0)

        with self.assertRaises(ValueError):
            classify_execution_outcome(coverage, None)
        with self.assertRaises(TypeError):
            PartialOutcome(coverage, self.cancelled)  # type: ignore[arg-type]

    def test_ds11_failure_record_is_typed_frozen_and_context_free(self) -> None:
        failure = FailureRecord(FailureCode.MISSING_CHUNK, FailureStage.DELIVERY)

        self.assertIs(failure.code, FailureCode.MISSING_CHUNK)
        self.assertIs(failure.stage, FailureStage.DELIVERY)
        self.assertFalse(hasattr(failure, "context"))
        self.assertFalse(hasattr(failure, "__dict__"))
        with self.assertRaises(TypeError):
            FailureRecord("MISSING_CHUNK", FailureStage.DELIVERY)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            FailureRecord(FailureCode.MISSING_CHUNK, "delivery")  # type: ignore[arg-type]
        # CPython's generated frozen/slotted dataclass setter differs for a
        # subclass instance across supported versions.  The public record must
        # still be immutable; its exact implementation exception is not data.
        with self.assertRaises((FrozenInstanceError, TypeError)):
            failure.code = FailureCode.CANCELLED  # type: ignore[misc]

        class ExtendedFailure(FailureRecord):
            pass

        extended = ExtendedFailure(FailureCode.MISSING_CHUNK, FailureStage.DELIVERY)
        # Build an adversarial subclass payload without depending on the
        # version-specific generated __setattr__ implementation above.
        object.__setattr__(extended, "context", {"payload": "private"})
        with self.assertRaises(TypeError):
            FailedOutcome(known(1), extended)

    def test_ds11_coverage_counts_are_validated_and_derived(self) -> None:
        invalid_counts = ((-1, 0), (0, 1), (True, 0), (1, False))
        for accepted, empty in invalid_counts:
            with self.subTest(accepted=accepted, empty=empty), self.assertRaises(
                (TypeError, ValueError)
            ):
                known(
                    1,
                    accepted_chunk_count=accepted,  # type: ignore[arg-type]
                    empty_chunk_count=empty,  # type: ignore[arg-type]
                )

        unknown = UnknownMissingRanges((RowRange(2, 5),), 2, 1)
        self.assertEqual(unknown.processed_row_count, 3)
        with self.assertRaises(ValueError):
            known(1, RowRange(0, 1), accepted_chunk_count=0)
        with self.assertRaises(ValueError):
            UnknownMissingRanges((RowRange(0, 1),), 0, 0)

    def test_ds11_outcome_owns_the_single_coverage_instance_for_future_provenance(self) -> None:
        coverage = known(2, RowRange(0, 1))
        outcome = PartialOutcome(coverage, self.cancelled)

        self.assertIs(outcome.coverage, coverage)
        # Frozen/slotted dataclasses reject direct mutation on every supported
        # CPython version.  Their generated setter can surface either
        # FrozenInstanceError or TypeError, so the exception class is not part
        # of the public outcome contract.
        with self.assertRaises((FrozenInstanceError, TypeError)):
            outcome.coverage = known(2, RowRange(0, 2))  # type: ignore[misc]
        with self.assertRaises((FrozenInstanceError, TypeError)):
            coverage.processed_ranges = ()  # type: ignore[misc]

    def test_ds11_classifier_table_is_exhaustive_for_known_final_states(self) -> None:
        cases = (
            (known(2, RowRange(0, 2)), None, CompleteOutcome),
            (known(2, RowRange(0, 1)), self.cancelled, PartialOutcome),
            (known(2), self.cancelled, FailedOutcome),
            (known(2, RowRange(0, 2)), self.cancelled, FailedOutcome),
            (UnknownMissingRanges((RowRange(0, 1),), 1, 0), self.cancelled, FailedOutcome),
        )
        for coverage, failure, expected_type in cases:
            with self.subTest(coverage=coverage, failure=failure):
                self.assertIsInstance(
                    classify_execution_outcome(coverage, failure),
                    expected_type,
                )

        with self.assertRaises(ValueError):
            classify_execution_outcome(known(2, RowRange(0, 1)), None)
        with self.assertRaises(TypeError):
            classify_execution_outcome(known(2), "CANCELLED")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            classify_execution_outcome(object(), None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            FailedOutcome(object(), self.cancelled)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

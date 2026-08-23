"""EXP09-EXP14 contracts for the fixed-state exponential reduction seam."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from math import isclose
import tracemalloc
import unittest

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.families.exponential import (
    ExponentialFitFailure,
    ExponentialFitFailureCode,
    ExponentialFitSuccess,
    fit_exponential,
    fit_exponential_chunks,
)
from veridist.statistics.exponential import ExponentialReductionState, reduce_exponential_chunks


class ExponentialReducerContracts(unittest.TestCase):
    """Streaming, reproducibility, state, and provenance constraints."""

    def test_exp09_ragged_and_empty_chunks_match_canonical_observation_order(self) -> None:
        chunks = (
            (ExactLifetime(Decimal("1")),),
            (),
            (RightCensoredLifetime(Decimal("2")), ExactLifetime(Decimal("3"))),
        )

        streaming = fit_exponential_chunks(chunks)
        canonical = fit_exponential(tuple(item for chunk in chunks for item in chunk))

        self.assertEqual(streaming, canonical)

    def test_exp10_reduction_state_is_frozen_slotted_and_fixed_shape(self) -> None:
        state = ExponentialReductionState.empty().add(ExactLifetime(Decimal("1")))

        self.assertEqual(
            tuple(ExponentialReductionState.__slots__),
            ("observation_count", "event_count", "total_time", "compensation"),
        )
        with self.assertRaises((FrozenInstanceError, TypeError)):
            state.event_count = 99  # type: ignore[misc]
        self.assertFalse(hasattr(state, "raw_rows"))

    def test_exp11_compensated_sum_preserves_small_term_in_adversarial_order(self) -> None:
        state = ExponentialReductionState.empty()
        for time in (Decimal("10000000000000000"), Decimal("1"), Decimal("1")):
            state = state.add(ExactLifetime(time))

        self.assertEqual(state.summed_time, 10_000_000_000_000_002.0)
        self.assertEqual(state.event_count, 3)

    def test_exp11_merge_preserves_compensation_and_declares_only_tolerance_across_partitions(self) -> None:
        left = reduce_exponential_chunks(((ExactLifetime(Decimal("1e16")), ExactLifetime(Decimal("1"))),))
        right = reduce_exponential_chunks(((ExactLifetime(Decimal("1")), ExactLifetime(Decimal("1"))),))
        merged = left.merge(right)
        direct = reduce_exponential_chunks(
            ((ExactLifetime(Decimal("1e16")), ExactLifetime(Decimal("1")), ExactLifetime(Decimal("1")), ExactLifetime(Decimal("1"))),)
        )

        self.assertEqual((merged.observation_count, merged.event_count), (4, 4))
        self.assertEqual(merged.summed_time, direct.summed_time)
        self.assertTrue(isclose(merged.summed_time, direct.summed_time, rel_tol=1e-15, abs_tol=0.0))

    def test_exp12_partition_order_is_tolerance_claim_not_bit_identical_contract(self) -> None:
        observations = tuple(ExactLifetime(Decimal("0.1")) for _ in range(10_001))
        canonical = fit_exponential(observations)
        partitioned = fit_exponential_chunks((observations[:3333], observations[3333:7777], observations[7777:]))

        self.assertIsInstance(canonical, ExponentialFitSuccess)
        self.assertIsInstance(partitioned, ExponentialFitSuccess)
        assert isinstance(canonical, ExponentialFitSuccess)
        assert isinstance(partitioned, ExponentialFitSuccess)
        self.assertTrue(isclose(canonical.rate, partitioned.rate, rel_tol=1e-15, abs_tol=0.0))
        self.assertEqual(canonical.provenance.reduction_order, "canonical_input_order")
        self.assertEqual(canonical.provenance.partition_order_contract, "tolerance_only")

    def test_exp13_provenance_is_closed_and_never_retains_source_locator_or_rows(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("2")),))

        self.assertIsInstance(result, ExponentialFitSuccess)
        assert isinstance(result, ExponentialFitSuccess)
        provenance = result.provenance
        self.assertEqual(provenance.accumulator_schema_version, "1")
        self.assertEqual(provenance.state_complexity, "O(1)")
        self.assertFalse(provenance.raw_data_retained)
        self.assertFalse(hasattr(provenance, "__dict__"))
        self.assertEqual(tuple(type(provenance).__slots__), (
            "accumulator_schema_version",
            "state_complexity",
            "reduction_order",
            "partition_order_contract",
            "raw_data_retained",
        ))

    def test_exp14_large_generator_uses_fixed_state_and_never_materializes_input(self) -> None:
        generated = (ExactLifetime(Decimal("2")) for _ in range(20_000))
        state = reduce_exponential_chunks((generated,))

        self.assertEqual((state.observation_count, state.event_count, state.total_time), (20_000, 20_000, 40_000.0))
        self.assertEqual(tuple(ExponentialReductionState.__slots__), (
            "observation_count", "event_count", "total_time", "compensation"
        ))
        self.assertFalse(hasattr(state, "__dict__"))

    def test_exp14_memory_growth_is_bounded_for_unique_generated_observations(self) -> None:
        def peak_for(size: int) -> int:
            tracemalloc.start()
            try:
                result = fit_exponential(ExactLifetime(Decimal(index + 1)) for index in range(size))
                self.assertIsInstance(result, ExponentialFitSuccess)
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        small_peak = peak_for(256)
        large_peak = peak_for(30_000)
        self.assertLess(
            large_peak - small_peak,
            1_500_000,
            "the fixed-cardinality reducer must not retain generated observations",
        )

    def test_exp14_aggregate_overflow_is_a_typed_non_estimate(self) -> None:
        result = fit_exponential((ExactLifetime(1e308), ExactLifetime(1e308)))

        self.assertEqual(
            result,
            ExponentialFitFailure(
                code=ExponentialFitFailureCode.NUMERICAL_OVERFLOW,
                observation_count=2,
                event_count=2,
                total_time=None,
            ),
        )
        chunked = fit_exponential_chunks(((ExactLifetime(1e308),), (ExactLifetime(1e308),)))
        self.assertEqual(chunked, result)

    def test_exp14_public_constructors_reject_inconsistent_facts_and_have_no_dict(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            ExponentialFitSuccess(
                rate=0.0,
                observation_count=1,
                event_count=1,
                total_time=1.0,
                mean=1.0,
                log_likelihood=0.0,
                censored_count=0,
            )
        with self.assertRaises((TypeError, ValueError)):
            ExponentialFitSuccess(
                rate=2.0,
                observation_count=1,
                event_count=1,
                total_time=1.0,
                mean=0.5,
                log_likelihood=0.0,
                censored_count=0,
                family="spoofed",
            )
        with self.assertRaises((TypeError, ValueError)):
            ExponentialFitFailure(
                code=ExponentialFitFailureCode.NO_OBSERVED_EVENTS,
                observation_count=1,
                event_count=1,
                total_time=1.0,
            )
        result = fit_exponential((ExactLifetime(Decimal("1")),))
        self.assertFalse(hasattr(result, "__dict__"))

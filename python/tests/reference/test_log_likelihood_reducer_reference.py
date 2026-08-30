"""Independent exact-integer reference checks for the likelihood reducer."""

from __future__ import annotations

import unittest
from fractions import Fraction
from math import copysign, isnan
from unittest.mock import patch

from veridist.families.registry import FamilyId


class LogLikelihoodReducerReferenceTests(unittest.TestCase):
    """Exercise boundaries that ordinary scalar formula grids cannot cover."""

    def test_exact_integer_oracle_handles_cancellation_subnormal_and_maximum(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodState

        values = (
            float.fromhex("0x1.fffffffffffffp+1023"),
            -float.fromhex("0x1.fffffffffffffp+1023"),
            5.0e-324,
            -0.0,
            0.0,
        )
        state = LogLikelihoodState.empty(FamilyId.NORMAL, mu=-0.0, sigma=1.0)
        oracle_units = 0
        for value in values:
            numerator, denominator = value.as_integer_ratio()
            oracle_units += numerator * ((1 << 1074) // denominator)
            state = state.add_log_density(value)
        self.assertEqual(state.total_units, oracle_units)
        self.assertEqual(state.finalize(), float(Fraction(oracle_units, 1 << 1074)))
        self.assertEqual(copysign(1.0, state.finalize()), 1.0)

    def test_final_overflow_and_count_cap_have_distinct_typed_failures(self) -> None:
        from veridist.statistics.log_density import LogDensitySuccess
        from veridist.statistics.log_likelihood import (
            MAX_OBSERVATION_COUNT,
            LogLikelihoodErrorCode,
            LogLikelihoodState,
            _FinalTotalNotRepresentable,
            _ObservationLimitExceeded,
            reduce_log_likelihood_chunks,
        )

        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        overflowing = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0)
        overflowing = overflowing.add_log_density(maximum).add_log_density(maximum)
        with self.assertRaises(_FinalTotalNotRepresentable):
            overflowing.finalize()
        capped = LogLikelihoodState.restore(
            FamilyId.NORMAL,
            observation_count=MAX_OBSERVATION_COUNT,
            total_units=0,
            mu=0.0,
            sigma=1.0,
        )
        one = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0).add_log_density(0.0)
        with self.assertRaises(_ObservationLimitExceeded):
            capped.merge(one)
        with patch(
            "veridist.statistics.log_likelihood._evaluate_validated_log_density",
            return_value=LogDensitySuccess(FamilyId.NORMAL, maximum),
        ):
            result = reduce_log_likelihood_chunks(FamilyId.NORMAL, ((0.0, 0.0),), mu=0.0, sigma=1.0)
        self.assertIs(result.code, LogLikelihoodErrorCode.FINAL_TOTAL_NOT_REPRESENTABLE)

    def test_lazy_source_is_not_read_ahead_after_closed_scalar_failure(self) -> None:
        from veridist.statistics.log_likelihood import (
            LogLikelihoodErrorCode,
            reduce_log_likelihood_chunks,
        )

        yielded: list[float] = []

        def source() -> object:
            for value in (0.0, float("nan"), 2.0):
                yielded.append(value)
                yield value

        result = reduce_log_likelihood_chunks(FamilyId.NORMAL, (source(),), mu=0.0, sigma=1.0)
        self.assertIs(result.code, LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE)
        self.assertEqual(yielded[0], 0.0)
        self.assertEqual(len(yielded), 2)
        self.assertTrue(isnan(yielded[1]))

    def test_restore_rejects_invalid_public_state_and_does_not_mutate_source(self) -> None:
        from veridist.statistics.log_likelihood import (
            MAX_OBSERVATION_COUNT,
            LogLikelihoodState,
        )

        source = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0)
        cases = (
            {"observation_count": True, "total_units": 0},
            {"observation_count": -1, "total_units": 0},
            {"observation_count": MAX_OBSERVATION_COUNT + 1, "total_units": 0},
            {"observation_count": 0, "total_units": 1},
            {"observation_count": 1, "total_units": "0"},
            {"observation_count": 1, "total_units": 1 << 2100},
        )
        for state_fields in cases:
            with self.subTest(state_fields=state_fields):
                with self.assertRaises((TypeError, ValueError)):
                    LogLikelihoodState.restore(FamilyId.NORMAL, mu=0, sigma=1, **state_fields)
                self.assertEqual(
                    source, LogLikelihoodState.empty(FamilyId.NORMAL, mu=-0.0, sigma=1)
                )

    def test_restore_identity_is_private_exact_and_merge_uses_it_not_sha_metadata(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodState

        positive = LogLikelihoodState.restore(
            FamilyId.GUMBEL_RIGHT, observation_count=0, total_units=0, location=0, scale=1
        )
        signed_zero = LogLikelihoodState.restore(
            FamilyId.GUMBEL_RIGHT, observation_count=0, total_units=0, location=-0.0, scale=1.0
        )
        self.assertEqual(positive, signed_zero)
        self.assertEqual(positive.parameter_fingerprint, signed_zero.parameter_fingerprint)
        self.assertNotIn("_canonical_identity", repr(positive))
        self.assertNotIn("0x", repr(positive))
        with self.assertRaises(ValueError):
            positive.merge(
                LogLikelihoodState.restore(
                    FamilyId.GUMBEL_RIGHT,
                    observation_count=0,
                    total_units=0,
                    location=1,
                    scale=1,
                )
            )

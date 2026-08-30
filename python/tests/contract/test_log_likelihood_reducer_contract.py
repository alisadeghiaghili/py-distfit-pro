"""LLR-01--LLR-05 contracts for exact-state streaming log likelihoods."""

from __future__ import annotations

from fractions import Fraction
from math import isfinite
import unittest
from dataclasses import FrozenInstanceError

from veridist.families.registry import FamilyId


class LogLikelihoodReducerContracts(unittest.TestCase):
    """The reducer retains only compatible exact binary64-output state."""

    def test_llr01_state_is_frozen_slotted_and_exact_units_are_bounded(self) -> None:
        from veridist.statistics.log_likelihood import (
            MAX_OBSERVATION_COUNT,
            MAX_TOTAL_UNITS,
            LogLikelihoodState,
        )

        state = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0)
        self.assertEqual(
            tuple(LogLikelihoodState.__slots__),
            ("family", "parameter_signature", "observation_count", "total_units"),
        )
        self.assertEqual((state.observation_count, state.total_units), (0, 0))
        self.assertEqual(MAX_OBSERVATION_COUNT, (1 << 64) - 1)
        self.assertEqual(MAX_TOTAL_UNITS.bit_length(), 2162)
        with self.assertRaises((FrozenInstanceError, TypeError)):
            state.total_units = 1  # type: ignore[misc]
        self.assertFalse(hasattr(state, "__dict__"))
        self.assertFalse(hasattr(state, "observations"))

    def test_llr01_finalization_matches_independent_fraction_oracle_and_rounds_once(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodState

        state = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0)
        # This deliberately cancels a large pair before retaining a subnormal unit.
        state = state.add_log_density(float.fromhex("0x1.0000000000000p+0"))
        state = state.add_log_density(-float.fromhex("0x1.0000000000000p+0"))
        state = state.add_log_density(float.fromhex("0x0.0000000000001p-1022"))
        expected = float(Fraction(state.total_units, 1 << 1074))
        self.assertEqual(state.finalize(), expected)
        self.assertEqual(state.finalize(), float.fromhex("0x0.0000000000001p-1022"))

    def test_llr02_merge_is_exact_and_rejects_incompatible_states(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodState

        left = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0).add_log_density(1.0)
        right = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0).add_log_density(-1.0)
        self.assertEqual(left.merge(right).total_units, 0)
        self.assertEqual(left.merge(right), right.merge(left))
        other_parameters = LogLikelihoodState.empty(FamilyId.NORMAL, mu=1.0, sigma=1.0)
        other_family = LogLikelihoodState.empty(FamilyId.GAMMA, shape=1.0, scale=1.0)
        with self.assertRaises(ValueError):
            left.merge(other_parameters)
        with self.assertRaises(ValueError):
            left.merge(other_family)
        with self.assertRaises(TypeError):
            left.merge(object())  # type: ignore[arg-type]

    def test_llr03_order_chunks_and_merge_tree_are_bit_identical(self) -> None:
        from veridist.statistics.log_likelihood import reduce_log_likelihood_chunks

        observations = (0.0, 1.0, -1.0, 0.5, -0.5)
        direct = reduce_log_likelihood_chunks(
            FamilyId.NORMAL, (observations,), mu=0.0, sigma=1.0
        )
        chunked = reduce_log_likelihood_chunks(
            FamilyId.NORMAL, (observations[:2], (), observations[2:]), mu=0.0, sigma=1.0
        )
        reverse = reduce_log_likelihood_chunks(
            FamilyId.NORMAL, (tuple(reversed(observations)),), mu=0.0, sigma=1.0
        )
        self.assertEqual(direct, chunked)
        self.assertEqual(direct, reverse)

    def test_llr04_all_families_dispatch_and_scalar_failures_are_closed(self) -> None:
        from veridist.statistics.log_likelihood import (
            LogLikelihoodErrorCode,
            LogLikelihoodFailure,
            LogLikelihoodSuccess,
            reduce_log_likelihood_chunks,
        )

        configurations = (
            (FamilyId.NORMAL, (0.0,), {"mu": 0.0, "sigma": 1.0}),
            (FamilyId.GAMMA, (1.0,), {"shape": 1.0, "scale": 1.0}),
            (FamilyId.WEIBULL_MIN, (1.0,), {"shape": 1.0, "scale": 1.0}),
            (FamilyId.LOGNORMAL, (1.0,), {"mu_log": 0.0, "sigma_log": 1.0}),
            (FamilyId.GUMBEL_RIGHT, (0.0,), {"location": 0.0, "scale": 1.0}),
        )
        for family, observations, parameters in configurations:
            with self.subTest(family=family):
                result = reduce_log_likelihood_chunks(family, (observations,), **parameters)
                self.assertIsInstance(result, LogLikelihoodSuccess)
                assert isinstance(result, LogLikelihoodSuccess)
                self.assertTrue(isfinite(result.total_log_likelihood))
        failed = reduce_log_likelihood_chunks(
            FamilyId.GAMMA, ((1.0, 0.0),), shape=1.0, scale=1.0
        )
        self.assertIsInstance(failed, LogLikelihoodFailure)
        assert isinstance(failed, LogLikelihoodFailure)
        self.assertIs(failed.code, LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE)
        self.assertEqual(failed.processed_count, 1)
        self.assertNotIn("0.0", failed.to_json())

    def test_llr05_empty_ragged_and_lazy_inputs_are_one_pass_and_unmaterialized(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodSuccess, reduce_log_likelihood_chunks

        yielded: list[int] = []

        def source() -> object:
            for value in (0.0, 1.0, 2.0):
                yielded.append(1)
                yield value

        empty = reduce_log_likelihood_chunks(FamilyId.NORMAL, ((), ()), mu=0.0, sigma=1.0)
        self.assertIsInstance(empty, LogLikelihoodSuccess)
        assert isinstance(empty, LogLikelihoodSuccess)
        self.assertEqual((empty.observation_count, empty.total_log_likelihood), (0, 0.0))
        result = reduce_log_likelihood_chunks(FamilyId.NORMAL, (source(),), mu=0.0, sigma=1.0)
        self.assertIsInstance(result, LogLikelihoodSuccess)
        self.assertEqual(yielded, [1, 1, 1])
        with self.assertRaises(TypeError):
            reduce_log_likelihood_chunks(FamilyId.NORMAL, 1, mu=0.0, sigma=1.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            reduce_log_likelihood_chunks(FamilyId.NORMAL, (1,), mu=0.0, sigma=1.0)  # type: ignore[arg-type]

"""LLR-01--LLR-05 contracts for exact-state streaming log likelihoods."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction
from hashlib import sha256
from math import isfinite
from unittest.mock import patch

from veridist.families.registry import FamilyId, FamilySpec


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
            (
                "family",
                "parameter_fingerprint",
                "observation_count",
                "total_units",
                "_canonical_identity",
            ),
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

    def test_llr02_fingerprint_is_opaque_and_normalizes_signed_zero(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodState

        positive = LogLikelihoodState.empty(FamilyId.GUMBEL_RIGHT, location=0.0, scale=1.0)
        negative = LogLikelihoodState.empty(FamilyId.GUMBEL_RIGHT, location=-0.0, scale=1)
        normal = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0)
        self.assertEqual(positive.parameter_fingerprint, negative.parameter_fingerprint)
        self.assertNotEqual(positive.parameter_fingerprint, normal.parameter_fingerprint)
        self.assertEqual(len(positive.parameter_fingerprint), sha256().digest_size * 2)
        self.assertTrue(
            all(character in "0123456789abcdef" for character in positive.parameter_fingerprint)
        )
        self.assertNotIn("0x", repr(positive))
        self.assertNotIn("location", repr(positive))

    def test_llr02_state_constructor_is_closed_and_restore_derives_exact_identity(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodState

        honest = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0)
        with self.assertRaises(TypeError):
            LogLikelihoodState(  # type: ignore[call-arg]
                FamilyId.NORMAL, honest.parameter_fingerprint, 0, 0
            )
        restored = LogLikelihoodState.restore(
            FamilyId.NORMAL, observation_count=0, total_units=0, mu=0, sigma=1
        )
        self.assertEqual(honest, restored)
        transplanted = LogLikelihoodState.restore(
            FamilyId.GAMMA, observation_count=0, total_units=0, shape=1.0, scale=1.0
        )
        with self.assertRaises(ValueError):
            honest.merge(transplanted)

    def test_llr03_order_chunks_and_merge_tree_are_bit_identical(self) -> None:
        from veridist.statistics.log_likelihood import reduce_log_likelihood_chunks

        observations = (0.0, 1.0, -1.0, 0.5, -0.5)
        direct = reduce_log_likelihood_chunks(FamilyId.NORMAL, (observations,), mu=0.0, sigma=1.0)
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
        failed = reduce_log_likelihood_chunks(FamilyId.GAMMA, ((1.0, 0.0),), shape=1.0, scale=1.0)
        self.assertIsInstance(failed, LogLikelihoodFailure)
        assert isinstance(failed, LogLikelihoodFailure)
        self.assertIs(failed.code, LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE)
        self.assertIsNotNone(failed.scalar_error_code)
        self.assertEqual(failed.processed_count, 1)
        self.assertNotIn("0.0", failed.to_json())
        self.assertNotIn("parameter", failed.to_json())
        self.assertIn('"scalar_error_code":"support_violation"', failed.to_json())

    def test_llr04_scalar_failure_preserves_closed_scalar_taxonomy(self) -> None:
        from veridist.statistics.log_density import LogDensityErrorCode
        from veridist.statistics.log_likelihood import (
            LogLikelihoodFailure,
            reduce_log_likelihood_chunks,
        )

        cases = (
            (
                FamilyId.NORMAL,
                (float("nan"),),
                {"mu": 0.0, "sigma": 1.0},
                LogDensityErrorCode.NONFINITE_OBSERVATION,
            ),
            (
                FamilyId.GAMMA,
                (0.0,),
                {"shape": 1.0, "scale": 1.0},
                LogDensityErrorCode.SUPPORT_VIOLATION,
            ),
            (
                FamilyId.NORMAL,
                (1e308,),
                {"mu": -1e308, "sigma": 1.0},
                LogDensityErrorCode.NUMERICAL_OVERFLOW,
            ),
        )
        for family, observations, parameters, expected in cases:
            with self.subTest(expected=expected):
                result = reduce_log_likelihood_chunks(family, (observations,), **parameters)
                self.assertIsInstance(result, LogLikelihoodFailure)
                assert isinstance(result, LogLikelihoodFailure)
                self.assertIs(result.scalar_error_code, expected)
                self.assertNotIn("0x", repr(result))

    def test_llr05_empty_ragged_and_lazy_inputs_are_one_pass_and_unmaterialized(self) -> None:
        from veridist.statistics.log_likelihood import (
            LogLikelihoodSuccess,
            reduce_log_likelihood_chunks,
        )

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

    def test_llr05_parameters_are_validated_once_before_the_stream_is_consumed(self) -> None:
        from veridist.statistics.log_likelihood import reduce_log_likelihood_chunks

        original = FamilySpec.validate_parameters
        with patch.object(
            FamilySpec, "validate_parameters", autospec=True, wraps=original
        ) as validated:
            result = reduce_log_likelihood_chunks(
                FamilyId.NORMAL, ((0.0, 1.0, 2.0),), mu=0.0, sigma=1.0
            )
        self.assertEqual(result.observation_count, 3)
        self.assertEqual(validated.call_count, 1)

    def test_llr03_independent_state_merge_trees_are_bit_identical(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodState

        states = []
        for chunk in ((1.0, -1.0), (0.5,), (-0.5,)):
            state = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0)
            for value in chunk:
                state = state.add_log_density(value)
            states.append(state)
        self.assertEqual(
            states[0].merge(states[1]).merge(states[2]),
            states[0].merge(states[1].merge(states[2])),
        )

    def test_llr05_fingerprint_and_state_checks_are_not_on_the_observation_hot_path(self) -> None:
        from veridist.statistics import log_likelihood

        with (
            patch.object(
                log_likelihood,
                "_identity_and_fingerprint",
                wraps=log_likelihood._identity_and_fingerprint,
            ) as fingerprint,
            patch.object(
                log_likelihood, "_validate_fingerprint", wraps=log_likelihood._validate_fingerprint
            ) as checked,
        ):
            result = log_likelihood.reduce_log_likelihood_chunks(
                FamilyId.NORMAL, ((0.0, 1.0, 2.0),), mu=0.0, sigma=1.0
            )
        self.assertEqual(result.observation_count, 3)
        self.assertEqual(fingerprint.call_count, 1)
        self.assertLessEqual(checked.call_count, 2)

    def test_llr06_closed_constructor_and_terminal_failure_boundaries(self) -> None:
        from veridist.statistics import log_likelihood
        from veridist.statistics.log_density import LogDensityErrorCode, LogDensityFailure
        from veridist.statistics.log_likelihood import (
            MAX_OBSERVATION_COUNT,
            LogLikelihoodErrorCode,
            LogLikelihoodFailure,
            LogLikelihoodState,
            LogLikelihoodSuccess,
            _ExactAccumulator,
            _FinalTotalNotRepresentable,
            _ObservationLimitExceeded,
            _validate_family_and_parameters,
            _validate_fingerprint,
            reduce_log_likelihood_chunks,
        )

        state = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0)
        with self.assertRaises(ValueError):
            state.add_log_density(1)
        capped = LogLikelihoodState.restore(
            FamilyId.NORMAL,
            observation_count=MAX_OBSERVATION_COUNT,
            total_units=0,
            mu=0.0,
            sigma=1.0,
        )
        with self.assertRaises(_ObservationLimitExceeded):
            capped.add_log_density(0.0)
        with self.assertRaises(TypeError):
            LogLikelihoodState._create("normal", state.parameter_fingerprint, 0, 0, ())
        with self.assertRaises(ValueError):
            LogLikelihoodState._create(FamilyId.NORMAL, state.parameter_fingerprint, 0, 1, ())
        with patch.object(log_likelihood, "isfinite", return_value=False):
            with self.assertRaises(_FinalTotalNotRepresentable):
                state.finalize()
        with self.assertRaises(TypeError):
            LogLikelihoodSuccess("normal", state.parameter_fingerprint, 0, 0.0)
        with self.assertRaises(ValueError):
            LogLikelihoodSuccess(FamilyId.NORMAL, state.parameter_fingerprint, -1, 0.0)
        with self.assertRaises(ValueError):
            LogLikelihoodSuccess(FamilyId.NORMAL, state.parameter_fingerprint, 0, 0)
        self.assertIn(
            "success",
            LogLikelihoodSuccess(FamilyId.NORMAL, state.parameter_fingerprint, 0, 0.0).to_json(),
        )
        with self.assertRaises(TypeError):
            LogLikelihoodFailure(
                "normal", LogLikelihoodErrorCode.OBSERVATION_LIMIT_EXCEEDED, 0, None
            )
        with self.assertRaises(TypeError):
            LogLikelihoodFailure(FamilyId.NORMAL, "bad", 0, None)
        with self.assertRaises(ValueError):
            LogLikelihoodFailure(
                FamilyId.NORMAL, LogLikelihoodErrorCode.OBSERVATION_LIMIT_EXCEEDED, -1, None
            )
        with self.assertRaises(ValueError):
            LogLikelihoodFailure(
                FamilyId.NORMAL, LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE, 0, None
            )
        with self.assertRaises(ValueError):
            LogLikelihoodFailure(
                FamilyId.NORMAL,
                LogLikelihoodErrorCode.OBSERVATION_LIMIT_EXCEEDED,
                0,
                LogDensityErrorCode.SUPPORT_VIOLATION,
            )
        self.assertIn(
            "failure",
            LogLikelihoodFailure(
                FamilyId.NORMAL, LogLikelihoodErrorCode.OBSERVATION_LIMIT_EXCEEDED, 0, None
            ).to_json(),
        )
        accumulator = _ExactAccumulator(MAX_OBSERVATION_COUNT, 0)
        with self.assertRaises(_ObservationLimitExceeded):
            accumulator.add(0.0)
        failure = LogDensityFailure(FamilyId.NORMAL, LogDensityErrorCode.SUPPORT_VIOLATION)
        with patch.object(log_likelihood, "_evaluate_validated_log_density", return_value=failure):
            result = reduce_log_likelihood_chunks(FamilyId.NORMAL, ((0.0,),), mu=0.0, sigma=1.0)
        self.assertIsInstance(result, LogLikelihoodFailure)
        with patch.object(log_likelihood, "MAX_OBSERVATION_COUNT", 0):
            result = reduce_log_likelihood_chunks(FamilyId.NORMAL, ((0.0,),), mu=0.0, sigma=1.0)
        self.assertIsInstance(result, LogLikelihoodFailure)
        assert isinstance(result, LogLikelihoodFailure)
        self.assertIs(result.code, LogLikelihoodErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        with self.assertRaises(TypeError):
            _validate_family_and_parameters("normal", {})
        with self.assertRaises(TypeError):
            _validate_fingerprint("bad")
        with self.assertRaises(ValueError):
            _validate_fingerprint("g" * 64)

    def test_llr02_fingerprint_matches_golden_canonical_binary64_contract(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodState

        # SHA-256 of the documented v1 wire fixture: gumbel_right,
        # location=0.0, scale=1.5 in canonical registry parameter order.
        expected = "c8195a6a5cff9506cb48094ffc9a42c60330fd1294e48f2b124bc07beba40df5"
        canonical_order = {"location": 0.0, "scale": 1.5}
        reverse_insertion_order = {"scale": 1.5, "location": 0.0}
        signed_zero = {"location": -0.0, "scale": 1.5}

        self.assertEqual(
            LogLikelihoodState.empty(
                FamilyId.GUMBEL_RIGHT, **canonical_order
            ).parameter_fingerprint,
            expected,
        )
        self.assertEqual(
            LogLikelihoodState.empty(
                FamilyId.GUMBEL_RIGHT, **reverse_insertion_order
            ).parameter_fingerprint,
            expected,
        )
        self.assertEqual(
            LogLikelihoodState.empty(FamilyId.GUMBEL_RIGHT, **signed_zero).parameter_fingerprint,
            expected,
        )

    def test_llr02_public_results_reject_noncanonical_fingerprint_types(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodSuccess

        valid = "a" * 64

        class StringSubclass(str):
            pass

        type_errors = (StringSubclass(valid), "a" * 63, "a" * 65, b"a" * 64, None)
        for fingerprint in type_errors:
            with self.subTest(fingerprint=fingerprint), self.assertRaisesRegex(
                TypeError, "^parameter_fingerprint must be a SHA-256 hex string$"
            ):
                LogLikelihoodSuccess(FamilyId.NORMAL, fingerprint, 0, 0.0)  # type: ignore[arg-type]
        for fingerprint in ("A" * 64, "g" * 64):
            with self.subTest(fingerprint=fingerprint), self.assertRaisesRegex(
                ValueError, "^parameter_fingerprint must be lowercase hexadecimal$"
            ):
                LogLikelihoodSuccess(FamilyId.NORMAL, fingerprint, 0, 0.0)

    def test_llr04_first_scalar_failure_stops_the_input_tail_exactly(self) -> None:
        from veridist.statistics.log_density import LogDensityErrorCode
        from veridist.statistics.log_likelihood import (
            LogLikelihoodErrorCode,
            LogLikelihoodFailure,
            reduce_log_likelihood_chunks,
        )

        consumed: list[float] = []

        def observations() -> object:
            for value in (1.0, 0.0, 7.0):
                consumed.append(value)
                yield value

        result = reduce_log_likelihood_chunks(
            FamilyId.GAMMA,
            (observations(),),
            shape=1.0,
            scale=1.0,
        )
        self.assertIsInstance(result, LogLikelihoodFailure)
        assert isinstance(result, LogLikelihoodFailure)
        self.assertIs(result.code, LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE)
        self.assertIs(result.scalar_error_code, LogDensityErrorCode.SUPPORT_VIOLATION)
        self.assertEqual(result.processed_count, 1)
        self.assertEqual(consumed, [1.0, 0.0])

    def test_llr05_invalid_setup_never_consumes_a_generator(self) -> None:
        from veridist.statistics.log_likelihood import reduce_log_likelihood_chunks

        consumed: list[str] = []

        def chunks() -> object:
            consumed.append("outer")
            yield (0.0,)

        with self.assertRaisesRegex(TypeError, "^family must be a FamilyId$"):
            reduce_log_likelihood_chunks("normal", chunks(), mu=0.0, sigma=1.0)  # type: ignore[arg-type]
        self.assertEqual(consumed, [])
        with self.assertRaisesRegex(ValueError, "^sigma must be positive$"):
            reduce_log_likelihood_chunks(FamilyId.NORMAL, chunks(), mu=0.0, sigma=0.0)
        self.assertEqual(consumed, [])

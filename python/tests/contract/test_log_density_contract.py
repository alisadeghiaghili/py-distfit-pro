"""Closed API and failure contracts for scalar log-density evaluation."""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import FrozenInstanceError

from veridist.families.registry import FAMILY_REGISTRY, FamilyId, Operation


class LogDensityContractTests(unittest.TestCase):
    """No raw input escapes through typed scalar evaluation outcomes."""

    def test_every_canonical_family_supports_logpdf_after_dispatch_lands(self) -> None:
        for family in FAMILY_REGISTRY.list():
            with self.subTest(family=family.id):
                self.assertTrue(family.plans(Operation.LOGPDF))
                self.assertTrue(family.supports(Operation.LOGPDF))

    def test_success_is_frozen_finite_and_deterministically_serialized(self) -> None:
        from veridist.statistics.log_density import LogDensitySuccess, evaluate_log_density

        result = evaluate_log_density(FamilyId.NORMAL, 1.5, mu=0.0, sigma=2.0)
        self.assertIsInstance(result, LogDensitySuccess)
        self.assertTrue(math.isfinite(result.log_density))
        self.assertEqual(
            result.to_json(),
            json.dumps(
                {"family": "normal", "log_density": result.log_density, "outcome": "success"},
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            result.log_density = 0.0  # type: ignore[misc]

    def test_parameter_keys_are_exact_canonical_names(self) -> None:
        from veridist.statistics.log_density import evaluate_log_density

        for keyword_values in (
            {"mu": 0.0},
            {"mu": 0.0, "sigma": 1.0, "unused": 1.0},
            {"loc": 0.0, "sigma": 1.0},
            {"location": 0.0, "sigma": 1.0},
        ):
            with self.subTest(keyword_values=keyword_values):
                with self.assertRaises(TypeError):
                    evaluate_log_density(FamilyId.NORMAL, 1.0, **keyword_values)

    def test_dispatch_rejects_aliases_strings_and_non_family_ids(self) -> None:
        from veridist.statistics.log_density import evaluate_log_density

        for family in ("normal", "gaussian", 1, None):
            with self.subTest(family=family):
                with self.assertRaises(TypeError):
                    evaluate_log_density(family, 1.0, mu=0.0, sigma=1.0)  # type: ignore[arg-type]

    def test_nonfinite_and_bool_observations_are_typed_failures_without_raw_leakage(self) -> None:
        from veridist.statistics.log_density import (
            LogDensityErrorCode,
            LogDensityFailure,
            evaluate_log_density,
        )

        for observation in (True, float("nan"), float("inf"), float("-inf")):
            with self.subTest(observation=observation):
                result = evaluate_log_density(FamilyId.NORMAL, observation, mu=0.0, sigma=1.0)
                self.assertIsInstance(result, LogDensityFailure)
                self.assertEqual(result.code, LogDensityErrorCode.NONFINITE_OBSERVATION)
                payload = result.to_json()
                self.assertEqual(payload, '{"code":"nonfinite_observation","family":"normal","outcome":"failure"}')
                self.assertNotIn('"observation":', payload)
                self.assertNotIn("nan", payload.lower())
                self.assertNotIn("inf", payload.lower())

    def test_positive_support_is_strict_and_boundary_is_typed(self) -> None:
        from veridist.statistics.log_density import (
            LogDensityErrorCode,
            LogDensityFailure,
            evaluate_log_density,
        )

        configurations = (
            (FamilyId.GAMMA, {"shape": 0.5, "scale": 2.0}),
            (FamilyId.WEIBULL_MIN, {"shape": 0.5, "scale": 2.0}),
            (FamilyId.LOGNORMAL, {"mu_log": 0.0, "sigma_log": 2.0}),
        )
        for family, parameters in configurations:
            for observation in (0.0, -1.0):
                with self.subTest(family=family, observation=observation):
                    result = evaluate_log_density(family, observation, **parameters)
                    self.assertIsInstance(result, LogDensityFailure)
                    self.assertEqual(result.code, LogDensityErrorCode.SUPPORT_VIOLATION)

    def test_parameter_contract_misuse_is_not_recast_as_evaluation_failure(self) -> None:
        from veridist.statistics.log_density import evaluate_log_density

        with self.assertRaises(ValueError):
            evaluate_log_density(FamilyId.GAMMA, 1.0, shape=0.0, scale=1.0)
        with self.assertRaises(TypeError):
            evaluate_log_density(FamilyId.GAMMA, 1.0, shape=True, scale=1.0)

    def test_required_nonrepresentable_intermediates_are_typed_overflow_failures(self) -> None:
        from veridist.statistics.log_density import (
            LogDensityErrorCode,
            LogDensityFailure,
            evaluate_log_density,
        )

        cases = (
            (FamilyId.NORMAL, 1.0e308, {"mu": -1.0e308, "sigma": 1.0}),
            (FamilyId.GAMMA, 1.0e308, {"shape": 2.0, "scale": 1.0e-308}),
            (FamilyId.WEIBULL_MIN, 1.0e308, {"shape": 2.0, "scale": 1.0e-308}),
            (FamilyId.LOGNORMAL, 1.0e308, {"mu_log": -1.0e308, "sigma_log": 1.0e-308}),
            (FamilyId.GUMBEL_RIGHT, -1.0e308, {"location": 1.0e308, "scale": 1.0}),
        )
        for family, observation, parameters in cases:
            with self.subTest(family=family):
                result = evaluate_log_density(family, observation, **parameters)
                self.assertIsInstance(result, LogDensityFailure)
                self.assertEqual(result.code, LogDensityErrorCode.NUMERICAL_OVERFLOW)

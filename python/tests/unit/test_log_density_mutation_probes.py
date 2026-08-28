"""Adversarial probes that document the scalar evaluator's mutation targets."""

from __future__ import annotations

import math
import unittest
from types import MappingProxyType

from veridist.families.registry import FAMILY_REGISTRY, FamilyId


class LogDensityMutationProbeTests(unittest.TestCase):
    """Kill formula, dispatch, support, and non-finite leakage mutations."""

    def test_formula_sign_and_required_term_probes_are_distinguishing(self) -> None:
        from veridist.statistics.log_density import LogDensitySuccess, evaluate_log_density

        cases = (
            (FamilyId.NORMAL, 4.0, {"mu": 0.0, "sigma": 1.0}, -8.918938533204672),
            (FamilyId.GAMMA, 3.0, {"shape": 2.5, "scale": 1.25}, -1.594623315756279),
            (FamilyId.WEIBULL_MIN, 5.0, {"shape": 2.0, "scale": 3.0}, -2.6724172621199513),
            (FamilyId.LOGNORMAL, 4.0, {"mu_log": 0.25, "sigma_log": 0.75}, -3.165252933084214),
            (FamilyId.GUMBEL_RIGHT, -3.0, {"location": 1.0, "scale": 2.0}, -6.082203279490596),
        )
        for family, observation, parameters, expected in cases:
            with self.subTest(family=family):
                result = evaluate_log_density(family, observation, **parameters)
                self.assertIsInstance(result, LogDensitySuccess)
                self.assertAlmostEqual(result.log_density, expected, delta=2.0e-13)
                self.assertTrue(math.isfinite(result.log_density))

    def test_dispatch_is_closed_and_does_not_leak_nonfinite_successes(self) -> None:
        from veridist.statistics.log_density import LogDensityFailure, evaluate_log_density

        for family, parameters in (
            (FamilyId.NORMAL, {"mu": 0.0, "sigma": 1.0}),
            (FamilyId.GAMMA, {"shape": 2.0, "scale": 1.0}),
            (FamilyId.WEIBULL_MIN, {"shape": 2.0, "scale": 1.0}),
            (FamilyId.LOGNORMAL, {"mu_log": 0.0, "sigma_log": 1.0}),
            (FamilyId.GUMBEL_RIGHT, {"location": 0.0, "scale": 1.0}),
        ):
            with self.subTest(family=family):
                result = evaluate_log_density(family, float("inf"), **parameters)
                self.assertIsInstance(result, LogDensityFailure)

    def test_registry_dispatch_parity_rejects_missing_and_extra_evaluators(self) -> None:
        from veridist.statistics import log_density

        with self.assertRaisesRegex(RuntimeError, "exactly match"):
            log_density._verify_registry_dispatch(FAMILY_REGISTRY.families, MappingProxyType({}))
        with self.assertRaisesRegex(RuntimeError, "exactly match"):
            log_density._verify_registry_dispatch(
                FAMILY_REGISTRY.families,
                MappingProxyType({**log_density._DISPATCH, "unexpected": log_density._normal}),
            )

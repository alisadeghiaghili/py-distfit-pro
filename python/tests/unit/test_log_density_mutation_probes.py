"""Adversarial probes that document the scalar evaluator's mutation targets."""

from __future__ import annotations

import math
import unittest
from types import MappingProxyType
from unittest.mock import patch

from veridist.families.registry import (
    FAMILY_REGISTRY,
    FamilyId,
    FamilySpec,
    Operation,
    ParameterRole,
    ParameterSpec,
)


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

    def test_registry_dispatch_rejects_advertised_but_unavailable_logpdf(self) -> None:
        from veridist.statistics import log_density

        unavailable = FamilySpec(
            id=FamilyId.NORMAL,
            aliases=(),
            parameters=(ParameterSpec("scale", ParameterRole.POSITIVE),),
            fixed_location=None,
            planned_operations=frozenset({Operation.LOGPDF}),
            available_operations=frozenset(),
        )
        registry = MappingProxyType(
            {family: specification for family, specification in FAMILY_REGISTRY.families.items()}
        )
        registry = MappingProxyType({**registry, FamilyId.NORMAL: unavailable})
        with self.assertRaisesRegex(RuntimeError, "cannot exceed"):
            log_density._verify_registry_dispatch(registry, log_density._DISPATCH)

    def test_evaluator_domain_and_nonfinite_outputs_become_typed_failures(self) -> None:
        from veridist.statistics import log_density
        from veridist.statistics.log_density import (
            LogDensityErrorCode,
            LogDensityFailure,
            evaluate_log_density,
        )

        def value_error(_: float, __: object) -> float:
            raise ValueError("synthetic domain error")

        def nonfinite(_: float, __: object) -> float:
            return float("nan")

        normal_parameters = {"mu": 0.0, "sigma": 1.0}
        with patch.object(
            log_density,
            "_DISPATCH",
            MappingProxyType({**log_density._DISPATCH, FamilyId.NORMAL: value_error}),
        ):
            domain_result = evaluate_log_density(FamilyId.NORMAL, 1.0, **normal_parameters)
        with patch.object(
            log_density,
            "_DISPATCH",
            MappingProxyType({**log_density._DISPATCH, FamilyId.NORMAL: nonfinite}),
        ):
            nonfinite_result = evaluate_log_density(FamilyId.NORMAL, 1.0, **normal_parameters)
        self.assertIsInstance(domain_result, LogDensityFailure)
        self.assertEqual(domain_result.code, LogDensityErrorCode.NONFINITE_LOG_DENSITY)
        self.assertIsInstance(nonfinite_result, LogDensityFailure)
        self.assertEqual(nonfinite_result.code, LogDensityErrorCode.NONFINITE_LOG_DENSITY)

    def test_dispatch_binds_every_available_registry_family_without_reverse_import(self) -> None:
        from veridist.statistics import log_density

        self.assertEqual(set(log_density._DISPATCH), set(FAMILY_REGISTRY.families))
        self.assertTrue(
            all(spec.supports(Operation.LOGPDF) for spec in FAMILY_REGISTRY.families.values())
        )

    def test_exact_ratio_helpers_do_not_turn_unrepresentable_values_into_successes(self) -> None:
        from veridist.statistics import log_density

        with self.assertRaises(log_density._NumericalOverflow):
            log_density._integer_ratio_to_float(10**10000, 1)

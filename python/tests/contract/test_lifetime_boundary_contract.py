"""Boundary contracts that prevent silent duration reinterpretation."""

from __future__ import annotations

from decimal import Decimal
import unittest

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime


class LifetimeBoundaryContracts(unittest.TestCase):
    """EXP14: conversion must not turn a positive duration into zero or infinity."""

    def test_positive_decimal_underflow_is_rejected_not_silently_changed_to_zero(self) -> None:
        for observation_type in (ExactLifetime, RightCensoredLifetime):
            with self.assertRaises((TypeError, ValueError)) as captured:
                observation_type(Decimal("1e-10000"))
            self.assertNotIn("1E-10000", str(captured.exception))

    def test_huge_integer_and_decimal_overflow_are_rejected_without_value_echo(self) -> None:
        for value in (10**10000, Decimal("1e10000")):
            with self.assertRaises((TypeError, ValueError)) as captured:
                ExactLifetime(value)
            self.assertNotIn(str(value), str(captured.exception))


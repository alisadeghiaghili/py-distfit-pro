"""Conformance checks for v1 distribution operations and caller-owned RNGs."""

from __future__ import annotations

import math
import unittest

import numpy as np


class V1FamilyOperationTests(unittest.TestCase):
    def test_normal_cdf_sf_ppf_reference_points(self) -> None:
        from veridist.statistics.distributions import cdf, ppf, sf

        parameters = {"mu": 0.0, "sigma": 1.0}
        self.assertEqual(cdf("normal", 0.0, parameters), 0.5)
        self.assertEqual(sf("normal", 0.0, parameters), 0.5)
        self.assertAlmostEqual(ppf("normal", 0.5, parameters), 0.0, places=15)

    def test_exponential_tail_uses_stable_survival_path(self) -> None:
        from veridist.statistics.distributions import cdf, sf

        parameters = {"rate": 1.0}
        tail = sf("exponential", 700.0, parameters)
        self.assertGreater(tail, 0.0)
        self.assertAlmostEqual(math.log(tail), -700.0, places=12)
        self.assertEqual(cdf("exponential", -1.0, parameters), 0.0)

    def test_sampling_uses_only_the_caller_generator(self) -> None:
        from veridist.statistics.distributions import sample

        legacy_state = np.random.get_state()
        first = sample("normal", 8, {"mu": 0.0, "sigma": 1.0}, np.random.default_rng(42))
        second = sample("normal", 8, {"mu": 0.0, "sigma": 1.0}, np.random.default_rng(42))
        self.assertEqual(first.tolist(), second.tolist())
        current_state = np.random.get_state()
        self.assertEqual(legacy_state[0], current_state[0])
        self.assertTrue((legacy_state[1] == current_state[1]).all())

    def test_sampling_rejects_non_generator_without_constructing_rng(self) -> None:
        from veridist.statistics.distributions import sample

        with self.assertRaises(TypeError):
            sample("normal", 1, {"mu": 0.0, "sigma": 1.0}, object())

    def test_invalid_operation_inputs_fail_loudly(self) -> None:
        from veridist.statistics.distributions import cdf, ppf, sample

        with self.assertRaises(ValueError):
            ppf("normal", 0.0, {"mu": 0.0, "sigma": 1.0})
        with self.assertRaises(TypeError):
            cdf("normal", object(), {"mu": 0.0, "sigma": 1.0})
        with self.assertRaises(ValueError):
            sample("normal", -1, {"mu": 0.0, "sigma": 1.0}, np.random.default_rng(4))


if __name__ == "__main__":
    unittest.main()

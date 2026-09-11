"""RED contracts for refit GoF, bootstrap and model selection."""

from __future__ import annotations

import math
import unittest


class V1InferenceSelectionTests(unittest.TestCase):
    def test_information_criteria_use_declared_free_parameters(self) -> None:
        from veridist.inference import information_criteria

        result = information_criteria(log_likelihood=-10.0, sample_size=100, free_parameters=2)
        self.assertAlmostEqual(result.aic, 24.0)
        self.assertAlmostEqual(result.bic, 20.0 + 2.0 * math.log(100.0))

    def test_refit_monte_carlo_records_failures_and_uncertainty(self) -> None:
        import numpy as np

        from veridist.inference import GofStatistic, refit_monte_carlo_gof

        result = refit_monte_carlo_gof(
            observations=(0.2, 0.4, 0.8, 1.6),
            family="exponential",
            statistics=frozenset({GofStatistic.KS, GofStatistic.AD, GofStatistic.CVM}),
            replicates=40,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(result.requested_replicates, 40)
        self.assertEqual(result.successful_replicates + result.failed_replicates, 40)
        self.assertGreater(result.successful_replicates, 0)
        self.assertGreater(result.monte_carlo_standard_error, 0.0)
        self.assertEqual(result.method, "refit_monte_carlo")
        self.assertEqual(result.rng_policy, "caller_owned_generator")

    def test_model_selection_returns_none_adequate(self) -> None:
        from veridist.inference import SelectionCode, compare_models

        result = compare_models(
            candidates=(
                {"family": "normal", "aic": 10.0, "bic": 12.0, "p_value": 0.001},
                {"family": "lognormal", "aic": 11.0, "bic": 13.0, "p_value": 0.01},
            ),
            adequacy_threshold=0.05,
        )
        self.assertEqual(result.code, SelectionCode.NONE_ADEQUATE)
        self.assertIsNone(result.selected_family)


if __name__ == "__main__":
    unittest.main()

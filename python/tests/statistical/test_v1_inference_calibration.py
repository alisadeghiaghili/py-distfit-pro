"""Small deterministic RED calibration smoke test; full grids run at release."""

from __future__ import annotations

import unittest


class V1InferenceCalibrationTests(unittest.TestCase):
    def test_calibration_summary_reports_binomial_uncertainty(self) -> None:
        from veridist.inference import summarize_calibration

        result = summarize_calibration(rejections=10, replicates=200, nominal_alpha=0.05)
        self.assertEqual(result.rejection_rate, 0.05)
        self.assertAlmostEqual(result.standard_error, (0.05 * 0.95 / 200) ** 0.5, places=15)
        self.assertEqual(result.scope, "declared_grid_only")

    def test_calibration_summary_rejects_invalid_counts_and_probability(self) -> None:
        from veridist.inference import summarize_calibration

        invalid = (
            {"rejections": -1, "replicates": 10, "nominal_alpha": 0.05},
            {"rejections": 1, "replicates": 0, "nominal_alpha": 0.05},
            {"rejections": 11, "replicates": 10, "nominal_alpha": 0.05},
            {"rejections": 1, "replicates": 10, "nominal_alpha": 0.0},
            {"rejections": 1, "replicates": 10, "nominal_alpha": float("nan")},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                summarize_calibration(**arguments)


if __name__ == "__main__":
    unittest.main()

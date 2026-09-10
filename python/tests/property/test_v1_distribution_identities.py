"""Property checks for v1 CDF, SF and PPF semantics."""

from __future__ import annotations

import unittest


class V1DistributionIdentityTests(unittest.TestCase):
    def test_registered_continuous_families_are_monotone_and_invertible(self) -> None:
        from veridist.statistics.distributions import cdf, ppf, sf

        cases = {
            "normal": {"mu": 1.0, "sigma": 2.0},
            "gamma": {"shape": 2.0, "scale": 3.0},
            "weibull_min": {"shape": 1.5, "scale": 4.0},
            "lognormal": {"mu_log": 0.5, "sigma_log": 0.75},
            "gumbel_right": {"location": 0.0, "scale": 2.0},
        }
        for family, parameters in cases.items():
            with self.subTest(family=family):
                quantiles = (0.01, 0.1, 0.5, 0.9, 0.99)
                values = tuple(ppf(family, probability, parameters) for probability in quantiles)
                self.assertEqual(values, tuple(sorted(values)))
                for probability, value in zip(quantiles, values, strict=True):
                    self.assertAlmostEqual(cdf(family, value, parameters), probability, places=10)
                    self.assertAlmostEqual(
                        cdf(family, value, parameters) + sf(family, value, parameters),
                        1.0,
                        places=14,
                    )


if __name__ == "__main__":
    unittest.main()

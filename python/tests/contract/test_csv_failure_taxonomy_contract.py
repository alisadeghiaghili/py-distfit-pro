"""RED taxonomy contract for the first CSV source adapter."""

from __future__ import annotations

import unittest

from veridist.engine.errors import FailureCode


class CsvFailureTaxonomyContracts(unittest.TestCase):
    """CSV failures extend the one closed engine taxonomy exactly once."""

    def test_csv_failure_codes_are_closed_engine_codes(self) -> None:
        expected = {
            "SOURCE_OPEN_FAILED",
            "SOURCE_DECODE_FAILED",
            "SOURCE_SCHEMA_INVALID",
            "SOURCE_ROW_INVALID",
        }
        actual = {
            code.value
            for code in FailureCode
            if code.value
            in {
                "SOURCE_OPEN_FAILED",
                "SOURCE_DECODE_FAILED",
                "SOURCE_SCHEMA_INVALID",
                "SOURCE_ROW_INVALID",
            }
        }

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

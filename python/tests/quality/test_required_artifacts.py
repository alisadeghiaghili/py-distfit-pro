"""Regression checks for required, version-controlled quality artifacts."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PYTHON_ROOT.parent
MANIFEST = PYTHON_ROOT / "quality" / "coverage-manifest.json"
CAPABILITY_MATRIX = REPOSITORY_ROOT / "docs" / "capability-matrix.md"


class RequiredQualityArtifactTests(unittest.TestCase):
    def test_capability_matrix_declares_the_only_callable_statistical_cell(self) -> None:
        content = CAPABILITY_MATRIX.read_text(encoding="utf-8")
        for required in (
            "0.0.0.dev0",
            "rate-only exponential MLE",
            "exact and independent right-censoring",
            "inference=not_provided",
            "fixed O(1) reducer state",
            "no production adapter or out-of-core claim",
        ):
            with self.subTest(required=required):
                self.assertIn(required, content)

    def test_coverage_manifest_exists_is_valid_and_is_not_gitignored(self) -> None:
        self.assertTrue(MANIFEST.is_file(), "required coverage manifest is missing")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["production_root"], "src/veridist")
        self.assertIn("production_files", manifest)
        self.assertIn("expected_denominators", manifest)

        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={PYTHON_ROOT.parent.as_posix()}",
                "check-ignore",
                "-q",
                "quality/coverage-manifest.json",
            ],
            cwd=PYTHON_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            1,
            "quality/coverage-manifest.json must remain visible to Git",
        )


if __name__ == "__main__":
    unittest.main()

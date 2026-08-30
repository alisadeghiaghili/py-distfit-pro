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
READINESS = REPOSITORY_ROOT / "docs" / "v1-readiness.md"
EVALUATED_FAMILY_ADR = REPOSITORY_ROOT / "docs" / "adr" / "ADR-0019-evaluated-family-kernel.md"


class RequiredQualityArtifactTests(unittest.TestCase):
    def test_readiness_uses_the_authoritative_production_file_count(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        production_count = len(manifest["production_files"])
        self.assertEqual(production_count, 24)
        readiness = " ".join(READINESS.read_text(encoding="utf-8").split())
        self.assertIn(
            f"accepted all {production_count} enumerated production files",
            readiness,
        )

    def test_readiness_separates_historical_snapshot_from_current_evidence(
        self,
    ) -> None:
        readiness = READINESS.read_text(encoding="utf-8")
        self.assertIn("Historical snapshot: `bfb496d` (preserved verbatim)", readiness)
        self.assertIn("accepted all 23 enumerated\n  production files", readiness)
        self.assertIn("Current unmerged family-kernel candidate", readiness)
        self.assertNotIn("coverage.json` SHA-256", readiness)

    def test_evaluated_family_adr_uses_a_primary_immutable_math_source(self) -> None:
        adr = EVALUATED_FAMILY_ADR.read_text(encoding="utf-8")
        self.assertIn("NIST DLMF §5.11", adr)
        self.assertNotIn("github.com/wch/r-source/blob/trunk", adr)

    def test_capability_matrix_declares_the_only_callable_statistical_cell(self) -> None:
        content = CAPABILITY_MATRIX.read_text(encoding="utf-8")
        for required in (
            "0.0.0.dev0",
            "rate-only exponential MLE",
            "exact and independent right-censoring",
            "inference=not_provided",
            "fixed O(1) reducer state",
            "one CSV iterator and one pass",
            "SCALE-CSV-EXP-01",
            "formal mutation runner remains **NOT IMPLEMENTED**",
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

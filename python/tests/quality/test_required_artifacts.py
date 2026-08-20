"""Regression checks for required, version-controlled quality artifacts."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PYTHON_ROOT / "quality" / "coverage-manifest.json"


class RequiredQualityArtifactTests(unittest.TestCase):
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

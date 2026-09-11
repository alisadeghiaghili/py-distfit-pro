"""Adversarial checks for cross-file release metadata."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.check_release_metadata import validate

ROOT = Path(__file__).resolve().parents[3]


class ReleaseMetadataTests(unittest.TestCase):
    def _copy_metadata(self, target: Path) -> None:
        (target / "python").mkdir()
        (target / "conda-forge-recipe").mkdir()
        for relative in (
            "CITATION.cff",
            ".zenodo.json",
            "python/pyproject.toml",
            "conda-forge-recipe/meta.yaml",
        ):
            destination = target / relative
            shutil.copyfile(ROOT / relative, destination)

    def test_repository_release_metadata_is_aligned(self) -> None:
        self.assertEqual(validate(ROOT), [])

    def test_rejects_version_drift_and_placeholder_recipe_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_metadata(root)
            zenodo_path = root / ".zenodo.json"
            zenodo = json.loads(zenodo_path.read_text("utf-8"))
            zenodo["version"] = "9.9.9"
            zenodo_path.write_text(json.dumps(zenodo), encoding="utf-8")
            recipe_path = root / "conda-forge-recipe/meta.yaml"
            recipe_path.write_text(
                recipe_path.read_text("utf-8").replace(
                    "e778525dcc2536fb9748d626487137c57ccdc6c65a2a5adeee2ce773ba585fee",
                    "replace-me",
                ),
                encoding="utf-8",
            )
            errors = " ".join(validate(root))
            self.assertIn("Zenodo version", errors)
            self.assertIn("SHA-256", errors)


if __name__ == "__main__":
    unittest.main()
